'''Train a Bidirectional LSTM on the IMDB sentiment classification task.

Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from tensorflow.keras.datasets import imdb
from tensorflow.keras.losses import BinaryCrossentropy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from BBoxTest import ada_split_test
import random
import collections

random.seed(2)
max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 500

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

## pos, neg, neural word list
dict_imdb = set(imdb.get_word_index().keys())
pos_words, neg_words = pd.read_csv('positive_words.txt', header=None), pd.read_csv('negative_words.txt', header=None)
pos_words, neg_words = pos_words[0].values, neg_words[0].values
pos_words, neg_words = set(pos_words).intersection(dict_imdb), set(neg_words).intersection(dict_imdb)
neural_words = list(dict_imdb - pos_words - neg_words)
pos_words, neg_words = list(pos_words), list(neg_words)
pos_lst, neg_lst = [*map(imdb.get_word_index().get, pos_words)], [*map(imdb.get_word_index().get, neg_words)]
neural_lst = [*map(imdb.get_word_index().get, neural_words)]

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

X = np.vstack((x_train, x_test))
y = np.hstack((y_train, y_test))

# generate Z samples
## sample pos, neg, neural
rand_idx = np.random.choice(len(neural_lst), 350)
test_lst = np.array(neural_lst)[rand_idx]
test_wds = np.array(neural_words)[rand_idx]

# test_lst = np.array(pos_lst)
# test_wds = np.array(pos_words)
# freq_dict = dict(collections.Counter(X.flatten()))
# freq_test_wds = np.array([*map( freq_dict.get, test_lst )])
# freq_test_wds[freq_test_wds==None] = 0
# freq_test_wds = freq_test_wds.astype(int)
# test_idx = freq_test_wds.argsort()[-350:][::-1]
# test_wds = test_wds[test_idx]
# test_lst = test_lst[test_idx]

print('hypothesis testing for words: \n %s' %test_wds)

X_perm = np.random.permutation(X)
ind_X = np.zeros_like(X, bool)
for wd_tmp in test_lst:
    ind_X += (X==wd_tmp)
Z = np.where(ind_X, X_perm, X)

def imdb_model():
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(tf.keras.optimizers.Adam(learning_rate=.00005), 'binary_crossentropy', metrics=['accuracy'])
    return model

model_full, model_mask = imdb_model(), imdb_model()

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

scheduler = ReduceLROnPlateau(monitor='val_loss', mode='min',
                            factor=0.382, patience=3, verbose=True, min_lr=1e-8)

es = EarlyStopping(monitor='val_loss', mode='min',
                   verbose=1, patience=10,
                   restore_best_weights=True)

fit_params = {'callbacks': [scheduler, es],
              'epochs': 100,
              'batch_size': 32,
              'validation_split': .2,
              'verbose': 2}

split_params = {'split': 'two-split',
                'num_perm': 100,
                'ratio_grid': [.2, .4, .6, .8],
                'perturb_scale': 5,
                'min_inf': 100,
                'min_est': 100,
                'ratio_method': 'log',
                'verbose': 0}

shiing = ada_split_test(model=model_full, model_mask=model_mask,
                        change='mask', loss='BinaryCrossentropy')

p_value_tmp = shiing.testing(X, Z, y[:,np.newaxis], perturb=-5.,
                            cv_num=5, cp='hommel',
                            fit_params=fit_params,
                            split_params=split_params)

print('P-values: %s' %p_value_tmp)
