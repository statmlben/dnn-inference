## OS_ada.py

import numpy as np
import sim_data
from functools import partial
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time
from dnn_inference.DnnT import DnnT
from scipy.optimize import brentq

array32 = partial(np.array, dtype=np.float32)
np.random.seed(0)

p, L0, d0, K0 = 100, 3, 64, 5
tau, x_max, pho = 2., .2, 0.25
N = 6000
n_params = p*d0 + (L0-2)*d0**2 + d0
print('the number of sample: %d; number of parameters: %d' %(N, n_params))

def size_fun(x, N, min_N = 2000):
	return x + min_N * np.log(x) / 2 / np.log(min_N/2) - N

verbose = 0
# specify model
P_value, SE_list, time_lst = [], [], []

for i in range(300):
	K.clear_session()

	def Reg_model(p, d, L=3, optimizer=Adam(lr=.0005)):
		model = Sequential()
		model.add(Dense(d, use_bias=False, input_dim=p, activation='relu'))
		for l in range(L-2):
			model.add(Dense(d, use_bias=False, activation='relu'))
		model.add(Dense(1, use_bias=False, activation='relu'))
		model.compile(loss='mean_squared_error', optimizer=optimizer)
		return model

	## Generate data
	X = sim_data.gen_X(n=N, p=p, pho=pho, x_max=x_max, distribution='normal')
	X0 = X.copy()
	X0[:,:K0] = 0.
	y = sim_data.gen_Y(p=p, d=d0, L=L0, X=X0, tau=tau, K0=K0, noise=1., if_norm=False)
	y = y[:,np.newaxis]
	# print('mean y: %.3f' %np.mean(y))

	# import matplotlib.pyplot as plt
	# plt.hist(Y, bins=50)
	# plt.show()
	tic = time.perf_counter()
	## Define the full model
	# d, L = d0, L0
	L, d = 3, 64
	model = Reg_model(p=p, d=d, L=L)
	model_mask = Reg_model(p=p, d=d, L=L)

	## define fitting params
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, 
						patience=50, restore_best_weights=True)

	fit_params = {'callbacks': [es],
				  'epochs': 100,
				  'batch_size': 512,
				  'validation_split': .2,
				  'verbose': 0}

	split_params = {'split': 'one-split',
					'num_perm': 100,
					# 'ratio_grid': [.2, .4, .6],
					'ratio_grid': [.2, .3, .4, .5, .6, .7, .8, .9],
					# 'perturb': 0.01,
					'perturb_grid': [.01, .05, .1, .5, 1.],
					# 'perturb_grid': [.01, .1, 1.],
					'verbose': 0}

	inf_cov = [range(0, K0), range(int(K0/2), int(K0/2)+K0), 
				range(int(p/2), int(p/2)+K0), range(p-K0, p)]
	# inf_cov = [range(K0)]
	root, info = brentq(size_fun, 3, N, args=(N, 1000), full_output=True)
	inf_ratio = 1 - root / N
	shiing = DnnT(inf_feats=inf_cov, model=model, model_mask=model_mask, change='mask')

	p_value_tmp = shiing.testing(X, y, cv_num=5, fit_params=fit_params,
						split_params=split_params)
	toc = time.perf_counter()
	P_value.append(p_value_tmp)
	time_lst.append(toc - tic)

P_value = np.array(P_value)
time_lst = np.array(time_lst)

# print('MSE: %.3f(%.3f)' %(SE_list.mean(), SE_list.std()))
print('Time: %.1f(%.1f)' %(time_lst.mean(), time_lst.std()))
print('CASE 0: Type I error: %.3f' %(len(P_value[:,0][P_value[:,0] <= shiing.alpha])/len(P_value)))
# print('CASE 1: Type 1 error: %.3f' %(len(P_value[:,1][P_value[:,1] <= .05])/len(P_value)))

for i in [1, 2, 3]:
	print('CASE %d: Type II error: %.3f' %(i, 1 - len(P_value[:,i][P_value[:,i] <= shiing.alpha])/len(P_value)))
