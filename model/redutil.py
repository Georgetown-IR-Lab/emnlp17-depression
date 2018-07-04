# reddit util methods
import collections
import datetime
import gzip
import json
import os
import pickle
import random
import time

import numpy as np
import sklearn.metrics
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical, categorical_probas_to_classes
from keras.callbacks import Callback


def config():
    max_posts = 400 # maximum posts per user
    max_length = 100 # maximum tokens per post
    mintf = 3 # min term frequency for tokens
    mindf = 2 # min doc frequency for tokens
    randposts = False # shuffle posts? (if False, posts will be ordered by date)
    noempty = False # remove empty posts? (e.g., posts with only a title)
    prep = None # other post pre-processing (e.g., rev to reverse post order)
    
    filters = 25 # post CNN filters
    filtlen = 3 # post CNN n-gram size
    poollen = 25
    ptype = 'acnn'
    acl = 15
    densed = 50
    embed_size = 50

    w2v = False # initialize embedding layer with pre-trained embeddings
    etrain = True # embeddings trainable?
    epochs = 25 # iterations to run for
    batch = 64 # batch size
    af = 'relu' # activation function used
    lr = 0.001
    seed = 123456 # random seed
    cosine = False
    
    
def evalon(stype, model, batch=32, X=None, y_true=None):
    print('\n-------- %s -------' % stype)
    print(datetime.datetime.now(), "predicting")
    y_pred = categorical_probas_to_classes(model.predict(X, batch_size=batch))

    print(datetime.datetime.now())
    posf1 = sklearn.metrics.f1_score(y_true, y_pred, pos_label=1, average='binary')
    posp = sklearn.metrics.precision_score(y_true, y_pred, pos_label=1, average='binary')
    posr = sklearn.metrics.recall_score(y_true, y_pred, pos_label=1, average='binary')
    print(sklearn.metrics.classification_report(y_true, y_pred))
    print("pred", collections.Counter(y_pred))
    print("true", collections.Counter(y_true))

    return posf1, posp, posr


class ValMetrics(Callback):
    def __init__(self, outdir, p):
        if os.path.exists(outdir):
            if len(os.listdir(outdir)) > 0:
                raise RuntimeError("callback outdir already exists: %s" % outdir)
        else:
            os.makedirs(outdir)
            
        self.outdir = outdir
        self.p = p
        self.losses = []
        self.bestf1 = 0.0

        stype = 'validation'
        cache = "%s_%s_%s_mintf%s_df%s_%s" % (stype, self.p['max_posts'], self.p['max_length'],
                                            self.p['mintf'], self.p['mindf'], self.p['noempty'])
        if self.p['prep'] is not None:
            cache += "_prep-%s" % self.p['prep']
            
        if os.path.exists("data/redcache/%s_X.npy" % cache):
            self.val_X = np.load("data/redcache/%s_X.npy" % cache)
            self.val_y = categorical_probas_to_classes(np.load("data/redcache/%s_y.npy" % cache))
        else:
            _, genf = datagen(self.p['max_posts'], self.p['max_length'], stype=stype,
                              force_full=True, mintf=self.p['mintf'], mindf=self.p['mindf'],
                              noempty=self.p['noempty'], prep=self.p['prep'], batch_size=9999999999)
            for i, (X, y) in enumerate(genf()):
                assert i == 0
            self.val_X, self.val_y = X, categorical_probas_to_classes(y)
            np.save("data/redcache/%s_X.npy" % cache, X)
            np.save("data/redcache/%s_y.npy" % cache, y)
    
    def on_epoch_end(self, epoch, logs=None):
        if len(self.losses) > 0 and logs['loss'] > 7.99 and self.losses[-1] > 7.99:
            raise RuntimeError("high loss indicates bad initialization: %s" % logs['loss'])
        self.losses.append(logs['loss'])

        f1, p, r = evalon('validation', self.model, X=self.val_X, y_true=self.val_y)
        print("epoch", epoch)
        if epoch == 1 or (f1 >= self.bestf1 and f1 >= 0.45):
            fn = os.path.join(self.outdir, "%s-%0.2f_%0.2f_%0.2f" % (epoch, f1, p, r))
            self.model.save_weights(fn, overwrite=True)
        if f1 >= self.bestf1:
            self.bestf1 = f1


def datagen(max_posts, max_length, stype='training', batch_size=32, force_full=False, randposts=False, mintf=1, mindf=2, noempty=False, prep=None, returntok=False, balbatch=True):
    assert stype in ['training', 'validation', 'testing']
    looponce = force_full or stype != 'training'
    fn = 'rsdd_posts/%s.gz' % stype
    
    print("loading %s posts" % stype)
    f = gzip.open(fn, 'rt')
    labels = {}
    allposts = {}
    for i, line in enumerate(f):
        user = str(i)
        d = json.loads(line)[0]
        if d['label'] == 'control':
            labels[user] = np.array([1, 0], dtype=np.float32)
        elif d['label'] == 'depression':
            labels[user] = np.array([0, 1], dtype=np.float32)
        elif d['label'] is None:
            continue
        else:
            raise RuntimeError("unknown label: %s" % d['label'])
        allposts[user] = [post for dt, post in d['posts']]
    f.close()

    tokfn = "tok_tf%s_df%s.p" % (mintf, mindf)
    load_tokenizer = looponce or os.path.exists(tokfn)

    if load_tokenizer:
        print("loading tokenizer")
        tok = pickle.load(open(tokfn, 'rb'))
    else:
        assert stype == 'training', "cannot fit tokenizer on validation or testing data"
        print("tokenizing %s users" % len(allposts))
        tok = Tokenizer(nb_words=None)
        tok.fit_on_texts(post for uposts in allposts.values() for post in uposts)

        # remove all tokens with a low DF or TF
        removed = 0
        for term in list(tok.word_index.keys()):
            if tok.word_docs[term] < mindf or tok.word_counts[term] < mintf:
                removed += 1
                del tok.word_docs[term]
                del tok.word_counts[term]
                del tok.word_index[term]
        tok.index_docs = None
        idxs = {}
        nexti = 1
        for term, oldi in sorted(tok.word_index.items()):
            idxs[term] = nexti
            nexti += 1
        assert len(tok.word_index) == len(idxs)
        tok.word_index = idxs

        print("terms removed: %s; remaining: %s" % (removed, len(tok.word_index)))
        pickle.dump(tok, open(tokfn, 'wb'), protocol=-1)

    nb_words = len(tok.word_index) + 1

    # remove empty posts
    if noempty:
        noempty_cache = "noempty_tf%s_df%s_%s_mp%s_ml%s.p" % (mintf, mindf, max_posts, max_length, stype)
        if os.path.exists(noempty_cache):
            print("loading cached noempty posts")
            allposts, before, after = pickle.load(open(noempty_cache, 'rb'))
        else:
            print("removing empty posts")
            before, after = [], []
            for user in list(allposts.keys()):
                before.append(len(allposts[user]))
                kept = []
                for upost in allposts[user]:
                    skip = True
                    for term in text_to_word_sequence(upost):
                        if term in tok.word_index:
                            skip = False
                            break

                    if not skip:
                        kept.append(upost)

                if len(kept) > 0:
                    allposts[user] = kept
                    after.append(len(allposts[user]))
                else:
                    del allposts[user]

            import scipy.stats
            print("posts before noempty:", scipy.stats.describe(before))
            print("posts after  noempty:", scipy.stats.describe(after))
            print("#users before vs. after: %s vs. %s" % (len(before), len(after)))
            pickle.dump((allposts, before, after), open(noempty_cache, 'wb'), protocol=-1)
            
    print("found %s words; generator ready" % nb_words)
    def vecify(uposts):
        assert prep is None or not randposts, "incompatible"
        if randposts or prep == 'bran':
            idxs = np.random.permutation(min(max_posts, len(uposts)))
            chosen = [uposts[idx] for idx in idxs]
        elif prep == 'dist':
            if max_posts >= len(uposts):
                chosen = uposts[:max_posts]
            else:
                idxs = np.linspace(0, len(uposts)-1, num=max_posts, dtype=np.int)
                chosen = [uposts[idx] for idx in idxs]
        elif prep == 'rev':
            chosen = uposts[-max_posts:]
        else:
            chosen = uposts[:max_posts]
            
        seqs = pad_sequences(tok.texts_to_sequences(chosen), maxlen=max_length)
        if len(seqs) < max_posts:
            seqs = np.pad(seqs, ((0, max_posts - len(seqs)), (0, 0)), mode='constant')
        return seqs

    if looponce:
        def gen(meta=False):
            X, y = [], []
            extra = []
            while True:
                for user, uposts in allposts.items():
                    X.append(vecify(uposts))
                    y.append(labels[user])
                    if meta:
                        extra.append((user, len(uposts)))

                    if len(X) == batch_size:
                        X, y = np.array(X), np.array(y)
                        print("...shouldn't happen")
                        yield (X, y)
                        X, y = [], []

                if looponce and len(X) > 0:
                    X, y = np.array(X), np.array(y)
                    if meta:
                        yield (X, y, extra)
                        X, y, extra = [], [], []
                    else:
                        yield (X, y)
                        X, y = [], []

                if looponce:
                    break
    else:
        def gen_nbb():
            bylabel = {}
            for user, uposts in allposts.items():
                label = np.argmax(labels[user])
                bylabel.setdefault(label, []).append(uposts)
            print([(k, len(v)) for k, v in bylabel.items()])

            X, y = [], []
            neglabel = np.array([1, 0], dtype=np.float32)
            poslabel = np.array([0, 1], dtype=np.float32)
            poscount = len(bylabel[1])
            while True:
                idxs = ([(1, i) for i in np.random.permutation(poscount)]
                        + [(0, i) for i in np.random.permutation(len(bylabel[0]))[:poscount]])
                idxs = [idxs[i] for i in np.random.permutation(len(idxs))]

                for label, idx in idxs:
                    X.append(vecify(bylabel[label][idx]))
                    if label == 0:
                        y.append(neglabel)
                    elif label == 1:
                        y.append(poslabel)
                    else:
                        raise RuntimeError("invalid label: %s" % label)

                    if len(X) == batch_size:
                        X, y = np.array(X), np.array(y)
                        yield (X, y)
                        X, y = [], []
        def gen_bal():
            bylabel = {}
            for user, uposts in allposts.items():
                label = np.argmax(labels[user])
                bylabel.setdefault(label, []).append(uposts)
            print([(k, len(v)) for k, v in bylabel.items()])

            assert batch_size % len(bylabel) == 0
            idxs = {}
            for label in bylabel:
                idxs[label] = list(range(len(bylabel[label])))

            X, y = [], []
            neglabel = np.array([1, 0], dtype=np.float32)
            poslabel = np.array([0, 1], dtype=np.float32)
            while True:
                for label in bylabel:
                    random.shuffle(idxs[label])

                for posidx, negidx in zip(idxs[1], idxs[0]):
                    X.append(vecify(bylabel[1][posidx]))
                    y.append(poslabel)

                    X.append(vecify(bylabel[0][negidx]))
                    y.append(neglabel)

                    if len(X) == batch_size:
                        X, y = np.array(X), np.array(y)
                        yield (X, y)
                        X, y = [], []

        if balbatch:
            gen = gen_bal
        else:
            gen = gen_nbb

    if returntok:
        return nb_words, gen, tok
    else:
        return nb_words, gen
