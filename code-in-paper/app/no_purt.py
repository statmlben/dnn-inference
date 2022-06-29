## OS_ada.py

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
from DnnT import DnnT

array32 = partial(np.array, dtype=np.float32)
np.random.seed(0)

p0, L0, d0, K0 = 100, 2, 32, 3
tau, x_max, pho = 2., .1, .00
N = 10000
n_params = p0*d0 + (L0-2)*d0**2 + d0
print('the number of sample: %d; number of parameters: %d' %(N, n_params))

verbose = 0
# specify model
P_value, SE_list, time_lst = [], [], []

if_power = 0

if if_power == 1:
	num_sim = 100
else:
	num_sim = 1000

for i in range(num_sim):
	K.clear_session()

	def Reg_model(p, d, L=3, optimizer=Adam(lr=.0001)):
		model = Sequential()
		model.add(Dense(d, use_bias=False, input_dim=p, activation='relu'))
		for l in range(L-2):
			model.add(Dense(d, use_bias=False, activation='relu'))
		model.add(Dense(1, use_bias=False, activation='relu'))
		model.compile(loss='mean_squared_error', optimizer=optimizer)
		return model

	## Generate data
	X = sim_data.gen_X(n=N, p=p0, pho=pho, x_max=x_max, distribution='section6-2')
	X0 = X.copy()
	X0[:,:K0] = 0.

	y = sim_data.gen_Y(p=p0, d=d0, L=L0, X=X0, tau=tau, K0=K0, noise=1.)
	y = y[:,np.newaxis]
	# print('mean y: %.3f' %np.mean(y))
	
	# import matplotlib.pyplot as plt
	# plt.hist(Y, bins=50)
	# plt.show()
	p = int(p0*(1. - 1./np.log(N)))
	X = X[:,:p]
	
	tic = time.perf_counter()

	## Define the full model
	d, L = d0, L0
	model = Reg_model(p=p, d=d, L=L)
	model_mask = Reg_model(p=p, d=d, L=L)
	
	## define fitting params
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=50, restore_best_weights=True)
	
	fit_params = {'callbacks': [es],
				  'epochs': 100,
				  'batch_size': 512,
				  'validation_split': .2,
				  'verbose': 0}

	split_params = {'split': 'one-sample',
					'perturb': 1.0,
					'num_perm': 100,
					'ratio_grid': [.2],
					'perturb_grid': [.01, .05, .1, .5, 1.],
					'min_inf': 0,
					'min_est': 0,
					'ratio_method': 'fuse',
					'cv_num': 1,
					'cp': 'gmean',
					'verbose': 0}
	if if_power == 1:
		inf_cov = [range(0, K0), range(int(K0/2), int(K0/2)+K0), range(int(p/2), int(p/2)+K0), range(p-K0, p)]
	else:
		inf_cov = [range(0, K0)]
	shiing = DnnT(inf_cov=inf_cov, model=model, model_mask=model_mask, change='mask')
	
	p_value_tmp, fit_err, p_value_cv = shiing.testing(X, y, cv_num=1, cp='hommel', fit_params=fit_params, split_params=split_params, inf_ratio=.2)
	toc = time.perf_counter()
	# P_value.append(p_value_tmp)
	# time_lst.append(toc - tic)
	if fit_err == 0:
		P_value.append(p_value_tmp)
		time_lst.append(toc - tic)
		#P_value_cv.append(p_value_cv)

P_value = np.array(P_value)
time_lst = np.array(time_lst)

# print('MSE: %.3f(%.3f)' %(SE_list.mean(), SE_list.std()))
print('Time: %.3f(%.3f)' %(time_lst.mean(), time_lst.std()/np.sqrt(len(time_lst))))
print('CASE 0: Type 1 error: %.3f' %(len(P_value[:,0][P_value[:,0] <= shiing.alpha])/len(P_value)))

if if_power == 1:
	for i in [1, 2, 3]:
		print('CASE %d: Power: %.3f' %(i, len(P_value[:,i][P_value[:,i] <= shiing.alpha])/len(P_value)))
