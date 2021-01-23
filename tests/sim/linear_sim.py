## linear_sim.py

import numpy as np
import sim_data
from functools import partial
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import keras.backend as K
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time
# from DnnT import DnnT
from dnn_inference import DnnT
import keras
from likelihood_ratio_test import likelihood_ratio_test
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression

array32 = partial(np.array, dtype=np.float32)
np.random.seed(0)

N, p = 1000, 100
print('the number of sample: %d; number of dim: %d' %(N, p))

verbose = 0
# specify model
P_value, SE_list, time_lst = [], [], []
P_value_lr = []

def log_loss(y_true, y_pred):
    label_true = np.argmax(y_true, 1)
    metric_tmp = -np.log(y_pred[range(len(y_pred)),label_true])
    return metric_tmp

for i in range(100):
    K.clear_session()

    ## Generate data
    beta_true = np.random.randn(p)
    beta_true[:10] = 0.
    X = np.random.randn(N,p)
    y = np.sign( np.dot(X, beta_true) + .1*np.random.randn(N) )
    y = (y + 1) / 2

    inf_cov = [range(0, 10), range(5, 15), range(10, 15)]
    ## For LRT
    # p_value_lr, time_lr = [], []
    # # model = LogisticRegression(penalty='none', max_iter=10000)
    # model = SGDClassifier(loss="log", penalty="l2", alpha=0.0,
    #                 max_iter=10000, learning_rate='constant', eta0=0.05)
    # for inf_cov_tmp in inf_cov:
    #     X_null_tmp = np.delete(X, inf_cov_tmp, 1)
    #     p_value_tmp = likelihood_ratio_test(X, y, model, X_null_tmp)
    #     p_value_lr.append(p_value_tmp)
    # P_value_lr.append(p_value_lr)

    ## For splits test
    def Reg_model(p, optimizer=Adam(lr=.05)):
        model = Sequential()
        model.add(Dense(2, use_bias=False, input_dim=p, activation='softmax'))
        # for l in range(L-2):
        #     model.add(Dense(d, use_bias=False, activation='relu'))
        # model.add(Dense(1, use_bias=False, activation='relu'))
        model.compile(loss='binary_crossentropy', optimizer=optimizer)
        return model

    y = keras.utils.to_categorical(y, 2)
    tic = time.perf_counter()
    ## Define the full model
    model = Reg_model(p=p)
    model_mask = Reg_model(p=p)

    ## define fitting params
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=50, restore_best_weights=True)

    fit_params = {'callbacks': [es],
                  'epochs': 1000,
                  'batch_size': 256,
                  'validation_split': .2,
                  'verbose': 0}

    split_params = {'split': 'two-split',
                    'if_reverse': 0,
                    'perturb_grid': [.01, .05, .1, .5, 1.],
                    'verbose': 1}

    shiing = DnnT(inf_cov=inf_cov, model=model, model_mask=model_mask, change='mask', eva_metric=log_loss)

    p_value_tmp = shiing.testing(X, y, cv_num=1, fit_params=fit_params, split_params=split_params)
    toc = time.perf_counter()
    P_value.append(p_value_tmp)
    time_lst.append(toc - tic)

P_value = np.array(P_value)
time_lst = np.array(time_lst)

# P_value_lr = np.array(P_value_lr)
# alpha = 0.05
# print('Performance for LRT')
# print('CASE 0: Type 1 error: %.3f' %(len(P_value_lr[:,0][P_value_lr[:,0] <= alpha])/len(P_value_lr)))
# for i in [1, 2]:
#     print('CASE %d: Power: %.3f' %(i, len(P_value_lr[:,i][P_value_lr[:,i] <= alpha])/len(P_value_lr)))

print('Performance for the proposed method')
# print('MSE: %.3f(%.3f)' %(SE_list.mean(), SE_list.std()))
print('Time: %.3f(%.3f)' %(time_lst.mean(), time_lst.std()))
print('CASE 0: Type 1 error: %.3f' %(len(P_value[:,0][P_value[:,0] <= shiing.alpha])/len(P_value)))
# print('CASE 1: Type 1 error: %.3f' %(len(P_value[:,1][P_value[:,1] <= .05])/len(P_value)))
for i in [1, 2]:
    print('CASE %d: Power: %.3f' %(i, len(P_value[:,i][P_value[:,i] <= shiing.alpha])/len(P_value)))
