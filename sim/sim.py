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
from keras.constraints import max_norm
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam, SGD

array32 = partial(np.array, dtype=np.float32)

np.random.seed(0)

p, L0, d0 = 100, 3, 128
tau, x_max = 2., 1.
m = 500
# n = int(x_max*m*p*np.log(d0)*(L0**3)*((1/tau)**(2*L0)))
# n = int(n / 100)
n = 1000
n_params = p*d0 + (L0-2)*d0**2 + d0
print('the number of training sample: %d; number of parameters: %d' %(n, n_params))

# specify model

P_value, SE_lst, SE_mask_lst = [], [], []

for i in range(100):
	K.clear_session()
	def Reg_model(p, d, L=3):
		model = Sequential()
		model.add(Dense(d, use_bias=False, input_dim=p, activation='relu'))
		for l in range(L-2):
			model.add(Dense(d, use_bias=False, activation='relu'))
			# model.add(Dense(d, use_bias=False, activation='relu', kernel_constraint=max_norm(1./tau)))
		model.add(Dense(1, use_bias=False, activation='relu'))
		model.compile(loss='mean_squared_error', optimizer=Adam(lr=.001))
		return model
	## Generate data
	K0 = 5
	X = funs.gen_X(n=n+2*m, p=p, pho=.25, x_max=x_max, distribution='normal')
	X0 = X.copy()
	X0[:,:K0] = 0.
	# W = funs.gen_W(p=p, d=d0, L=L0, tau=tau, K0=5)
	Y = funs.gen_Y(p=p, d=d0, L=L0, X=X0, tau=tau, K0=K0, noise=1.)
	print('mean Y: %.3f' %np.mean(Y))

	# import matplotlib.pyplot as plt
	# plt.hist(Y, bins=50)
	# plt.show()

	## training
	X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=n, random_state=42)
	X_inf, X_inf_mask, y_inf, y_inf_mask = train_test_split(X_test, y_test, train_size=m, random_state=42)
	# generate hypothesis testing set
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, restore_best_weights=True)
	d, L = d0, L0
	model = Reg_model(p=p, d=d, L=L)
	history = model.fit(X_train, y_train, epochs=100, validation_split=.1, verbose=0, callbacks=[es])

	p_value = []
	for inf_cov in [range(0, 5), range(3, 8), range(45, 50), range(95, 100)]:
		Z_train = X_train.copy()
		Z_train[:,inf_cov]= 0.

		model_mask = Reg_model(p=p, d=d, L=L)

		# fit model
		history_mask = model_mask.fit(Z_train, y_train, epochs=100, validation_split=.2, verbose=0, callbacks=[es])

		## Inference
		from scipy.stats import norm
		Z_inf = X_inf_mask.copy()
		Z_inf[:,inf_cov] = 0.

		pred_y, pred_y_mask = model.predict(X_inf).flatten(), model_mask.predict(Z_inf).flatten()
		SE, SE_mask = (pred_y - y_inf)**2, (pred_y_mask - y_inf_mask)**2
		Lambda = np.sqrt(m) * ( SE.std()**2 + SE_mask.std()**2 )**(-1/2)*( SE.mean() - SE_mask.mean() )
		
		p_value_tmp = norm.cdf(Lambda)
		if p_value_tmp < .05:
			print('reject H0 with p_value: %.3f' %p_value_tmp)
		else:
			print('accept H0 with p_value: %.3f' %p_value_tmp)

		p_value.append(p_value_tmp)
		if inf_cov == range(0, 5):
			SE_lst.append(SE.mean())
			SE_mask_lst.append(SE_mask.mean())

	P_value.append(p_value)

P_value, SE_lst, SE_mask_lst = np.array(P_value), np.array(SE_lst), np.array(SE_mask_lst)

print('Type 1 error: %.3f' %(len(P_value[:,0][P_value[:,0] < .05])/len(P_value)))

for i in [1,2,3]:
	print('CASE %d: Power: %.3f' %(i, len(P_value[:,i][P_value[:,i] < .05])/len(P_value)))

print('training sample: %d; MSE: %.3f, MSE_mask: %.3f' %(n, SE_lst.mean(), SE_mask_lst.mean()))


## n = 500, m = 500
# Type 1 error: 0.037
# CASE 1: Power: 0.537
# CASE 2: Power: 0.907
# CASE 3: Power: 0.870

## Adaptive 
# Type 1 error: 0.050
# CASE 1: Power: 0.570
# CASE 2: Power: 0.890
# CASE 3: Power: 0.880

## n=1000, m = 500
# Type 1 error: 0.070
# CASE 1: Power: 0.920
# CASE 2: Power: 1.000
# CASE 3: Power: 1.000
# training sample: 1000; MSE: 1.857, MSE_mask: 2.113
