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
N = 2000
n_params = p*d0 + (L0-2)*d0**2 + d0
print('the number of sample: %d; number of parameters: %d' %(N, n_params))

for ratio in [.25]:
	P_value, SE_lst, SE_mask_lst, p_value_null = [], [], [], []
	m = int(ratio*N)
	n = N - 2*m

	for i in range(100):
		K.clear_session()

		def Reg_model(p, d, L=3, optimizer=Adam(lr=.001)):
			model = Sequential()
			model.add(Dense(d, use_bias=False, input_dim=p, activation='relu'))
			for l in range(L-2):
				model.add(Dense(d, use_bias=False, activation='relu'))
				# model.add(Dense(d, use_bias=False, activation='relu', kernel_constraint=max_norm(1./tau)))
			model.add(Dense(1, use_bias=False, activation='relu'))
			model.compile(loss='mean_squared_error', optimizer=optimizer)
			return model
			
		## Generate data
		K0 = 5
		X = funs.gen_X(n=N, p=p, pho=.25, x_max=x_max, distribution='normal')
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
		# fit model
		d, L = d0, L0
		model = Reg_model(p=p, d=d, L=L)
		es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20, restore_best_weights=True)
		history = model.fit(X_train, y_train, epochs=100, validation_split=.1, verbose=0, callbacks=[es])

		# generate hypothesis testing set
		p_value = []
		for inf_cov in [range(0, 5), range(3, 8), range(45, 50), range(95, 100)]:
			Z_train = X_train.copy()
			Z_train[:,inf_cov]= 0.

			model_mask = Reg_model(p=p, d=d, L=L)
			history_mask = model_mask.fit(Z_train, y_train, epochs=100, validation_split=.1, verbose=0, callbacks=[es])

			## Inference
			from scipy.stats import norm
			Z_inf = X_inf_mask.copy()
			Z_inf[:,inf_cov] = 0.

			pred_y, pred_y_mask = model.predict(X_inf).flatten(), model_mask.predict(Z_inf).flatten()
			SE, SE_mask = (pred_y - y_inf)**2, (pred_y_mask - y_inf_mask)**2
			Lambda = np.sqrt(m) * ( SE.std()**2 + SE_mask.std()**2 )**(-1/2)*( SE.mean() - SE_mask.mean() )

			print('Ratio is %.3f; SE: %.3f(%.3f); SE_mask: %.3f(%.3f)' %(ratio, SE.mean(), SE.std(), SE_mask.mean(), SE_mask.std()))
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

	print('Ratio: %.3f; P_value: %.3f; Type 1 error: %.3f' %(ratio, P_value[:,0].mean(), len(P_value[:,0][P_value[:,0] < .05])/len(P_value)))

	for i in [1,2,3]:
		print('Ratio: %.3f; CASE %d: Power: %.3f' %(ratio, i, len(P_value[:,i][P_value[:,i] < .05])/len(P_value)))

	print('Ratio: %.3f; training sample: %d; MSE: %.3f, MSE_mask: %.3f' %(ratio, n, SE_lst.mean(), SE_mask_lst.mean()))


# Ratio: 0.100; P_value: 0.776; Type 1 error: 0.040
# Ratio: 0.100; CASE 1: Power: 0.480
# Ratio: 0.100; CASE 2: Power: 0.980
# Ratio: 0.100; CASE 3: Power: 0.940
# Ratio: 0.100; training sample: 1600; MSE: 1.592, MSE_mask: 2.365

# Ratio: 0.200; P_value: 0.916; Type 1 error: 0.000
# Ratio: 0.200; CASE 1: Power: 0.510
# Ratio: 0.200; CASE 2: Power: 0.960
# Ratio: 0.200; CASE 3: Power: 0.970
# Ratio: 0.200; training sample: 1200; MSE: 2.304, MSE_mask: 1.434

# Ratio: 0.250; P_value: 0.916; Type 1 error: 0.010
# Ratio: 0.250; CASE 1: Power: 0.550
# Ratio: 0.250; CASE 2: Power: 0.990
# Ratio: 0.250; CASE 3: Power: 0.980
# Ratio: 0.250; training sample: 1000; MSE: 1.851, MSE_mask: 1.749

# Ratio: 0.300; Type 1 error: 0.020
# Ratio: 0.300; CASE 1: Power: 0.370
# Ratio: 0.300; CASE 2: Power: 0.940
# Ratio: 0.300; CASE 3: Power: 0.950
# Ratio: 0.300; training sample: 800; MSE: 2.371, MSE_mask: 2.197

# Ratio: 0.400; Type 1 error: 0.020
# Ratio: 0.400; CASE 1: Power: 0.090
# Ratio: 0.400; CASE 2: Power: 0.520
# Ratio: 0.400; CASE 3: Power: 0.460
# Ratio: 0.400; training sample: 400; MSE: 3.578, MSE_mask: 2.576
