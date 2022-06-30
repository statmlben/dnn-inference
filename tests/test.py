# test.py
# Test sig_test based on MNIST dataset
# Author: Ben Dai <bendai@cuhk.edu.hk>

import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras import backend as K
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, SGD
import argparse
import emoji 

def load_data():
    np.random.seed(0)
    num_classes = 2
    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X = np.vstack((x_train, x_test))
    y = np.hstack((y_train, y_test))
    ind = (y == 9) + (y == 7)
    X, y = X[ind], y[ind]
    X = X.astype('float32')
    X += .01*abs(np.random.randn(14251, 28, 28))
    y[y==7], y[y==9] = 0, 1

    if K.image_data_format() == 'channels_first':
        X = X.reshape(x.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X = X.reshape(X.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X /= 255.

    # convert class vectors to binary class matrices
    y = keras.utils.to_categorical(y, num_classes)

    ## define the learning models
    def cnn():
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(0.001), metrics=['accuracy'])
        return model

    model_null, model_alter = cnn(), cnn()
    return X, y, model_null, model_alter

## fitting param
if __name__=='__main__':
    X, y, model_null, model_alter = load_data()
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Test for ``sig_test.py``')
    parser.add_argument('-s', '--split', default='one-split', type=str)
    parser.add_argument('-r', '--inf_ratio', default=None, type=float)
    parser.add_argument('-pb', '--perturb', default=None, type=float)
    parser.add_argument('-cv', '--cv_num', default=2, type=int)
    parser.add_argument('-cp', '--cp', default='hommel', type=str)
    parser.add_argument('-tr', '--tune_ratio_method', default='fuse', type=str)
    args = parser.parse_args()

    from tensorflow.keras.callbacks import EarlyStopping
    from sig_test import split_test

    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=0, patience=10, restore_best_weights=True)

    fit_params = {'callbacks': [es],
                'epochs': 50,
                'batch_size': 32,
                'validation_split': .2,
                'verbose': 0}

    test_params = { 'split': args.split,
                    'inf_ratio': args.inf_ratio,
                    'perturb': args.perturb,
                    'cv_num': args.cv_num,
                    'cp': args.cp,
                    'verbose': 2}

    tune_params = { 'num_perm': 100,
                    'ratio_grid': [.2, .4, .6, .8],
                    'if_reverse': 0,
                    'perturb_range': 2.**np.arange(-3,3,.1),
                    'tune_ratio_method': args.tune_ratio_method,
                    'tune_pb_method': 'fuse',
                    'cv_num': 2,
                    'cp': 'hommel',
                    'verbose': 2}
    
    ## Inference based on dnn_inference
    inf_feats = [[np.arange(19,28), np.arange(13,20)], [np.arange(7,16), np.arange(9,16)]]
    cue = split_test(inf_feats=inf_feats, model_null=model_null, model_alter=model_alter, eva_metric='zero-one')
    ## test for `tuneHP`
    print(" TEST `tuneHP` ".center(70, '='))
    cue.tuneHP(k=0, X=X, y=y, test_params=test_params, fit_params=fit_params, tune_params=tune_params)
    print('\U00002705'+' results for tuned `test_params`:')
    print(cue.test_params)

    ## test for `test_base`
    print(" TEST `test_base` ".center(70, '='))
    p_value_cp, P_value_cv = cue.test_base(0, X, y, fit_params, cue.test_params, verbose=2)
    print('\U00002705'+' results for `test_base`:')
    print('p_value:\n')
    print(p_value_cp)
    print('p_value_cv:\n')
    print(P_value_cv)

    ## test for `testing`
    ## use the orginial `test_params`
    print(" TEST `testing` ".center(70, '='))
    cue.test_params = test_params
    P_value = cue.testing(X, y, fit_params, test_params, tune_params)
    print('\U00002705'+' results for `testing`:')
    print(P_value)


