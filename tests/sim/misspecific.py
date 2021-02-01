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

L0, d0 = 2, 32
N, p = 2000, 10
print('the number of sample: %d; number of dim: %d' %(N, p))

verbose = 0
# specify model
P_value, SE_list, time_lst = [], [], []
P_value_lr = []

for i in range(100):
    K.clear_session()

    ## Generate data
    X = np.random.randn(N,p)
    y = .1*X[:,5] + .2*X[:,6]**2 + .3*X[:,7]**3 + .4*X[:,8]*X[:,9] + .3*np.random.randn(N)
    y = y[:,np.newaxis]
    inf_cov = [range(0, 5), range(2, 7), range(5, 8)]
    # inf_cov = [range(0, 5)]

    def Reg_model(p, d, L=3, optimizer=Adam(lr=.001)):
        model = Sequential()
        model.add(Dense(d, input_dim=p, activation='relu'))
        for l in range(L-2):
            model.add(Dense(d, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model
    ## For splits test
    tic = time.perf_counter()
    ## Define the full model
    d, L = d0, L0
    model = Reg_model(p=p, d=d, L=L)
    model_mask = Reg_model(p=p, d=d, L=L)

    ## define fitting params
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0,
                        patience=50, restore_best_weights=True)

    fit_params = {'callbacks': [es],
                  'epochs': 1000,
                  'batch_size': 64,
                  'validation_split': .2,
                  'verbose': 0}

    split_params = {'split': 'two-split',
                    'perturb': None,
                    'if_reverse': 0,
                    'perturb_grid': [.01, .05, .1, .5, 1.],
                    'verbose': 1}

    shiing = DnnT(inf_cov=inf_cov, model=model, model_mask=model_mask,
                    change='mask', cp_path='./missp_checkpoints')

    p_value_tmp = shiing.testing(X, y, cv_num=5, fit_params=fit_params,
                                split_params=split_params, inf_ratio=None)
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
for i in [1, 2]:
    print('CASE %d: Power: %.3f' %(i, len(P_value[:,i][P_value[:,i] <= shiing.alpha])/len(P_value)))
