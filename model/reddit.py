from __future__ import print_function
import os
os.environ['PYTHONHASHSEED'] = '0'
import collections
import datetime
import gzip
import json
import pickle
import random
import time

import numpy as np
import sklearn.metrics
import tensorflow as tf
import keras.backend as K
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Embedding, Input, TimeDistributed, Activation, Masking, Convolution1D, MaxPooling1D, Flatten, AveragePooling1D, GlobalAveragePooling1D

import sacred
from sacred.utils import apply_backspaces_and_linefeeds
ex = sacred.Experiment('train')
ex.path = 'train'
sacred.SETTINGS.HOST_INFO.CAPTURED_ENV.append('CUDA_VISIBLE_DEVICES')
sacred.SETTINGS.HOST_INFO.CAPTURED_ENV.append('USER')
ex.captured_out_filter = apply_backspaces_and_linefeeds

from redutil import datagen, config, ValMetrics
config = ex.config(config)


def build_model(p):
    """ build a Keras model using the parameters in p """
    max_posts = p['max_posts']
    max_length = p['max_length']
    filters = p['filters']
    filtlen = p['filtlen']
    poollen = p ['poollen']
    densed = p['densed']
    embed_size = p['embed_size']
    batch = p['batch']

    random.seed(p['seed'])
    np.random.seed(p['seed'])
    # https://github.com/fchollet/keras/issues/2280
    tf.reset_default_graph()
    if len(tf.get_default_graph()._nodes_by_id.keys()) > 0:
        raise RuntimeError("Seeding is not supported after building part of the graph. "
                            "Please move set_seed to the beginning of your code.")
    tf.set_random_seed(p['seed'])
    sess = tf.Session()
    K.set_session(sess)
        
    nb_words, genf, tok = datagen(max_posts, max_length, stype='training', batch_size=batch,
                                  randposts=p['randposts'], mintf=p['mintf'], mindf=p['mindf'],
                                  noempty=p['noempty'], prep=p['prep'],
                                  returntok=True)

    n_classes = 2
    inp = Input(shape=(max_posts, max_length), dtype='int32')
    nextin = inp

    if p['cosine']:
        from keras.constraints import unitnorm
        wconstrain = unitnorm()
    else:
        wconstrain = None
        
    if p['w2v']:
        embeddings_index = {}
        fn = 'data/w2v_50_sg_export.txt'
        with open(fn) as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                if word in tok.word_index:
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
        print('Found %s word vectors.' % len(embeddings_index))

        embedding_matrix = np.zeros((nb_words, embed_size))
        for word, i in tok.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                embedding_matrix[i] = np.random.uniform(-0.2, 0.2, size=embed_size)

        emb = Embedding(nb_words, embed_size, mask_zero=True,
                        input_length=max_length, W_constraint=wconstrain,
                        weights=[embedding_matrix], trainable=p['etrain'])
        if not p['etrain']:
            print("making not trainable")
            emb.trainable = False
    else:
        assert p['etrain'], "must have etrain=True with w2v=False"
        emb = Embedding(nb_words, embed_size, mask_zero=True, W_constraint=wconstrain)

    embedded = TimeDistributed(emb)(nextin)

    if not p['etrain']:
        emb.trainable = False
        embedded.trainable = False

    conv = Sequential()
    conv.add(Convolution1D(nb_filter=filters, filter_length=filtlen, border_mode='valid', W_constraint=wconstrain,
                           activation='linear', subsample_length=1, input_shape=(max_length, embed_size)))
    conv.add(Activation(p['af']))
    conv.add(GlobalAveragePooling1D())

    posts = TimeDistributed(conv)(embedded)
    combined = Convolution1D(nb_filter=filters, filter_length=p['acl'], border_mode='valid',
                             activation=p['af'], subsample_length=p['acl'])(posts)
    combined = Flatten()(combined)

    if densed != 0:
        combined = Dense(densed, activation=p['af'])(combined)
    outlayer = Dense(2, activation='softmax')(combined)

    model = Model(inp, outlayer)
    return model, genf


@ex.automain
def main(_log, _config):
    p = _config
    max_posts = p['max_posts']
    max_length = p['max_length']
    filters = p['filters']
    filtlen = p['filtlen']
    poollen = p ['poollen']
    densed = p['densed']
    embed_size = p['embed_size']
    batch = p['batch']
    
    model, genf = build_model(p)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=p['lr']))
    print(model.summary())
    #from keras.utils.visualize_util import plot, model_to_dot
    #plot(model, to_file='model-r.png', show_shapes=True)
    #plot(conv, to_file='model-r2.png', show_shapes=True)

    cw = None
    from keras.callbacks import ModelCheckpoint
    modelfn = "tmp/w/%s_%s_%s_%s_%s_%s-" % (filters, filtlen, poollen, densed, embed_size, max_posts) + "{epoch:02d}"

    outdir = "tmp/w/" + ".".join(["%s:%s" % (k, v) for k, v in sorted(p.items())])
    if os.path.exists(outdir):
        import time
        outdir += "_%d" % time.time()
    cb = [ValMetrics(outdir, p)]
    hist = model.fit_generator(genf(), samples_per_epoch=3072, nb_epoch=p['epochs'], max_q_size=20, class_weight=cw, callbacks=cb)

    K.clear_session()
