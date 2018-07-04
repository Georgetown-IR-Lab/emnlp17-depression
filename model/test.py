import os
import sys
import sklearn.metrics
import tensorflow as tf
from keras.utils.np_utils import to_categorical, categorical_probas_to_classes

from reddit import build_model
from redutil import datagen

if len(sys.argv) != 2:
    print("usage: <weights>")
    sys.exit(1)

weightsfn = sys.argv[1]    

def fn_model(fn, stype='testing'):
    # parse config options to use (except for LR, which conflicts with the . param separator)
    p = dict([x.split(":") for x in os.path.dirname(fn.replace("tmp/w/", "")).replace("lr:0.001.", "").split(".")])

    strk = ['af', 'ptype'] # force string conversion
    boolk = ['w2v', 'randposts', 'noempty', 'etrain', 'balbatch', 'cosine'] # force bool conversion
    for k in p:
        if k == 'prep':
            if p[k].lower() == 'none':
                p[k] = None
            else:
                p[k] = str(p[k])
        elif k in strk:
            p[k] = str(p[k])
        elif k in boolk:
            p[k] = (p[k] == 'True')
        else:
            p[k] = int(p[k])
    p["lr"] = 0.001

    tf.reset_default_graph()
    model, _ = build_model(p)
    model.load_weights(fn)
    
    _, genf = datagen(p['max_posts'], p['max_length'], stype=stype,
                      force_full=True, mintf=p['mintf'], mindf=p['mindf'],
                      noempty=p['noempty'], prep=p['prep'], batch_size=9999999999)
    for i, (X, y) in enumerate(genf()):
        assert i == 0, "test set should contain only one batch (and it should not be sampled)"
    val_X, val_y = X, categorical_probas_to_classes(y)
    
    y_pred = categorical_probas_to_classes(model.predict(val_X, batch_size=32))

    y_true = val_y
    posf1 = sklearn.metrics.f1_score(y_true, y_pred, pos_label=1, average='binary')
    posp = sklearn.metrics.precision_score(y_true, y_pred, pos_label=1, average='binary')
    posr = sklearn.metrics.recall_score(y_true, y_pred, pos_label=1, average='binary')

    return (posf1, posp, posr), y_pred, val_y

metrics, y_pred, val_y = fn_model(weightsfn)
print(metrics)
print(sklearn.metrics.classification_report(val_y, y_pred))
