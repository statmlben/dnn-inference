import numpy as np
import pandas as pd
from numpy import linalg as LA
import funs
from functools import partial
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.constraints import Constraint
import keras.backend as K
# from keras.constraints import x_max_norm
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam, SGD
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

array32 = partial(np.array, dtype=np.float32)
np.random.seed(0)

p, L0, d0, K0 = 100, 2, 64, 5
tau, x_max, pho = 2., .4, 0.25
verbose = 0
N = 6000
n_params = p*d0 + (L0-2)*d0**2 + d0
print('the number of sample: %d; number of parameters: %d' %(N, n_params))

# specify model

P_value, SE_list = [], []

for i in range(100):
	K.clear_session()

	def Reg_model(p, d, L=3, optimizer=Adam(lr=.0005)):
		model = Sequential()
		model.add(Dense(d, use_bias=False, input_dim=p, activation='relu'))
		for l in range(L-2):
			model.add(Dense(d, use_bias=False, activation='relu'))
			# model.add(Dense(d, use_bias=False, activation='relu', kernel_constraint=max_norm(1./tau)))
		model.add(Dense(1, use_bias=False, activation='relu'))
		model.compile(loss='mean_squared_error', optimizer=optimizer)
		return model

	## Generate data
	X = funs.gen_X(n=N, p=p, pho=pho, x_max=x_max, distribution='normal')
	X0 = X.copy()
	X0[:,:K0] = 0.
	# W = funs.gen_W(p=p, d=d0, L=L0, tau=tau, K0=5)
	y = funs.gen_Y(p=p, d=d0, L=L0, X=X0, tau=tau, K0=K0, noise=1.)
	# print('mean y: %.3f' %np.mean(y))

	## Define the full model
	d, L = d0, L0
	model = Reg_model(p=p, d=d, L=L)
	model_perm = Reg_model(p=p, d=d, L=L)
	
	## define fitting params
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=50, restore_best_weights=True)
	
	fit_params = {'callbacks': [es],
				  'epochs': 100,
				  'batch_size': 512,
				  'validation_split': .2,
				  'verbose': 0}

	# split_params = {'num_perm': 100,
	# 				'ratio_grid': [.1, .2, .3, .4],
	# 				'method_': 'perm_max',
	# 				'min_inf': 100,
	# 				'verbose': 1}

	shiing = funs.PermT(inf_cov=[range(0, K0), range(int(K0/2)+1, int(K0/2)+K0+1), range(int(p/2), int(p/2)+K0), range(p-K0, p)], 
						model=model, model_perm=model_perm)
	
	p_value_tmp, SE_tmp = shiing.testing(X, y, fit_params=fit_params)
	P_value.append(p_value_tmp)
	SE_list.append(SE_tmp)

P_value=np.array(P_value)
SE_list=np.array(SE_list)
print('MSE: %.3f(%.3f)' %(SE_list.mean(), SE_list.std()))
print('Type 1 error: %.3f' %(len(P_value[:,0][P_value[:,0] < .05])/len(P_value)))

for i in [1, 2, 3]:
	print('CASE %d: Power: %.3f' %(i, len(P_value[:,i][P_value[:,i] < .05])/len(P_value)))
