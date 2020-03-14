import numpy as np
import pandas as pd
from numpy import linalg as LA
from scipy.stats import norm
from functools import partial
import numpy as np
from keras.constraints import Constraint
from sklearn.linear_model import LinearRegression
from scipy.optimize import brentq
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import warnings
import keras.backend as K
from keras.initializers import glorot_uniform
import tensorflow as tf


array32 = partial(np.array, dtype=np.float32)

def gen_X(n, p, pho, x_max=1., distribution='uniform'):
	if distribution == 'uniform':
		cov = np.zeros((p,p), dtype='float32')
		for i in range(p):
			for j in range(p):
				cov[i,j] = pho**(abs(i-j))
		M = np.linalg.cholesky(cov)
		W = np.random.multivariate_normal(np.zeros(p), cov, n).T
		W = np.array(W, 'float32')
		Z = np.dot(M, W)
		# X = x_max*(2*norm.cdf(Z)-1).T
		X = x_max*(2*norm.cdf(Z)-1).T
		X = array32(X)
	if distribution == 'normal':
		cov = np.zeros((p,p), dtype='float32')
		for i in range(p):
			for j in range(p):
				cov[i,j] = pho**(abs(i-j))
		X = x_max*np.random.multivariate_normal(np.zeros(p), cov, n)
	return X

def gen_W(p, d, L, tau, K0=5):
	W_tmp = array32(np.random.random(size=(d,p)))
	# W_tmp[:,:K0] = 0.
	col_sums = np.sqrt(np.sum(W_tmp**2, axis=0))
	# col_sums[col_sums==0] = 1.
	W_tmp = W_tmp/col_sums[np.newaxis, :]
	W_tmp = W_tmp / np.sqrt(p)*tau
	W = [W_tmp]
	for l in range(L-2):
		W_tmp = array32(np.random.random(size=(d,d)))
		col_sums = np.sqrt(np.sum(W_tmp**2, axis=0))
		W_tmp = W_tmp/col_sums[np.newaxis, :]
		W_tmp = W_tmp/np.sqrt(d)*tau
		W.append(W_tmp)
	W_tmp = array32(np.ones((1,d)))
	W_tmp = W_tmp / np.sqrt(d)*tau
	W.append(W_tmp)
	return W

def gen_Y(p, d, L, X, tau, K0=5, noise=.1):
	n = len(X)
	W = gen_W(p, d, L, tau, K0)
	Y_tmp = X.T
	for l in range(L):
		Y_tmp = relu(np.dot(W[l], Y_tmp))
	Y_tmp = Y_tmp + noise*np.random.randn(n)
	return Y_tmp.flatten()

def relu(x):
	return x * (x > 0)

def group_norm(W, p=2, q=1):
	return np.sum((np.sum(np.abs(W)**p, axis=1))**(q/p))**(1/q)

class DeepT(object):
	def __init__(self, inf_cov, model, model_mask, alpha=.05, verbose=0):
		self.inf_cov = inf_cov
		self.model = model
		self.model_mask = model_mask
		self.alpha = alpha

	# def reset_model(self):
	# 	initial_weights = self.model.get_weights()
	# 	backend_name = K.backend()
	# 	if backend_name == 'tensorflow': 
	# 		k_eval = lambda placeholder: placeholder.eval(session=K.get_session())
	# 	elif backend_name == 'theano': 
	# 		k_eval = lambda placeholder: placeholder.eval()
	# 	else: 
	# 		raise ValueError("Unsupported backend")
	# 	new_weights = [k_eval(glorot_uniform()(w.shape)) for w in initial_weights]
	# 	self.model.set_weights(new_weights)

	def reset_model(self):
		if int(tf.__version__[0]) == 2:
			# for layer in self.model.layers: 
			# 	if isinstance(layer, tf.keras.Model):
			# 		reset_weights(layer)
			# 		continue
			# 	for k, initializer in layer.__dict__.items():
			# 		if "initializer" not in k:
			# 			continue
			# 			# find the corresponding variable
			# 		var = getattr(layer, k.replace("_initializer", ""))
			# 		var.assign(initializer(var.shape, var.dtype))

			for layer in self.model.layers:
				if isinstance(layer, tf.keras.Model): #if you're using a model as a layer
					reset_weights(layer) #apply function recursively
					continue

				#where are the initializers?
				if hasattr(layer, 'cell'):
					init_container = layer.cell
				else:
					init_container = layer

				for key, initializer in init_container.__dict__.items():
					if "initializer" not in key: #is this item an initializer?
					  continue #if no, skip it

					# find the corresponding variable, like the kernel or the bias
					if key == 'recurrent_initializer': #special case check
						var = getattr(init_container, 'recurrent_kernel')
					else:
						var = getattr(init_container, key.replace("_initializer", ""))
					
					if var is None:
						continue
					else:
						var.assign(initializer(var.shape, var.dtype))
		
		if int(tf.__version__[0]) == 1:
			session = K.get_session()
			for layer in self.model.layers:
				if hasattr(layer, 'kernel_initializer'):
					layer.kernel.initializer.run(session=session)
			for layer in self.model_mask.layers:
				if hasattr(layer, 'kernel_initializer'):
					layer.kernel.initializer.run(session=session)


	## can be extent to @abstractmethod
	def mask_cov(self, X, k=0, type_='vector'):
		if type_ == 'vector':
			Z = X.copy()
			Z[:,self.inf_cov[k]]= 0.
		return Z

	def perm_cov(self, X, k=0, type_='vector'):
		if type_ == 'vector':
			Z = X.copy()
			Z[:,self.inf_cov[k]]= np.random.permutation(Z[:,self.inf_cov[k]])
		return Z

	def noise_cov(self, X, k=0, type_='vector'):
		if type_ == 'vector':
			Z = X.copy()
			Z[:,self.inf_cov[k]] = np.random.randn(len(X), len(self.inf_cov[k]))
		return Z

	def adaRatio(self, X, y, k=0, fit_params={}, num_perm=100, ratio_grid=[.1, .2, .3, .4], min_inf=50, method_='perm_max', verbose=0):
		if method_=='noise':
			for ratio_tmp in reversed(ratio_grid):
				found = 0
				self.reset_model()
				m_tmp = int(len(X)*ratio_tmp)
				n_tmp = len(X) - 2*m_tmp
				# split data
				X_perm = self.noise_cov(X, k)
				X_train, X_test, y_train, y_test = train_test_split(X_perm, y, train_size=n_tmp, random_state=42)
				# training for full model
				history = self.model.fit(x=X_train, y=y_train, **fit_params)
				# training for mask model
				Z_train = self.mask_cov(X_train, k)
				history_mask = self.model_mask.fit(x=Z_train, y=y_train, **fit_params)
				## evaluate the performance
				P_value = []
				for t in range(num_perm):
					# permutate testing sample
					X_test_perm = self.noise_cov(X_test, k)
					# split two sample
					X_inf, X_inf_mask, y_inf, y_inf_mask = train_test_split(X_test_perm, y_test, train_size=m_tmp, random_state=42)
					Z_inf = self.mask_cov(X_inf_mask, k)
					# evaluation
					pred_y = self.model.predict(X_inf).flatten()
					pred_y_mask = self.model_mask.predict(Z_inf).flatten()
					SE_tmp = (pred_y - y_inf)**2
					SE_mask_tmp = (pred_y_mask - y_inf_mask)**2
					Lambda_tmp = np.sqrt(m_tmp) * ( SE_tmp.std()**2 + SE_mask_tmp.std()**2 )**(-1/2)*( SE_tmp.mean() - SE_mask_tmp.mean())
					p_value_tmp = norm.cdf(Lambda_tmp)
					P_value.append(p_value_tmp)
				P_value = np.array(P_value)
				## compute the type 1 error
				Err1 = len(P_value[P_value<self.alpha])/len(P_value)
				if verbose==1:
					print('Type 1 error: %.3f; inference sample ratio: %.3f' %(Err1, ratio_tmp))
				if Err1 < self.alpha:
					found = 1
					break
			if found==0:
				warnings.warn("No ratio can control the Type 1 error, pls increase the sample size, and inference sample ratio is set as the min of ratio_grid.")
			if m_tmp < min_inf:
				warnings.warn("The estimated inference sample is too small, pls increase the sample size, and inference sample is set as 100")
				m_tmp = min_inf
				n_tmp = len(X) - 2*m_tmp
			return n_tmp, m_tmp
		
		if method_ == 'perm_max':
			candidate, Err1_lst, ratio_lst = [], [], []
			found = 0
			for ratio_tmp in reversed(ratio_grid):
				self.reset_model()
				m_tmp = int(len(X)*ratio_tmp)
				n_tmp = len(X) - 2*m_tmp
				# split data
				X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_tmp, random_state=42)
				# permutate training sample
				y_train_perm = np.random.permutation(y_train)
				# training for full model
				history = self.model.fit(x=X_train, y=y_train_perm, **fit_params)
				# training for mask model
				Z_train = self.mask_cov(X_train, k)
				history_mask = self.model_mask.fit(x=Z_train, y=y_train_perm, **fit_params)
				
				## evaluate the performance
				P_value = []
				Z_test = self.mask_cov(X_test, k)
				pred_y = self.model.predict(X_test).flatten()
				pred_y_mask = self.model_mask.predict(Z_test).flatten()
				# print(len(pred_y), len(pred_y_mask))
				
				for t in range(num_perm):
					# permutate testing sample
					y_test_perm = np.random.permutation(y_test)
					# split two sample
					ind_inf, ind_inf_mask = train_test_split(range(len(y_test_perm)), train_size=m_tmp, random_state=42)
					# X_inf, X_inf_mask, y_inf, y_inf_mask = train_test_split(X_test, y_test_perm, train_size=m_tmp, random_state=42)
					# Z_inf = self.mask_cov(X_inf_mask, k)
					# evaluation
					SE_tmp = (y_test_perm[ind_inf] - pred_y[ind_inf])**2
					SE_mask_tmp = (y_test_perm[ind_inf_mask] - pred_y_mask[ind_inf_mask])**2
					# pred_y = self.model.predict(X_inf).flatten()
					# pred_y_mask = self.model_mask.predict(Z_inf).flatten()
					# SE_tmp = (pred_y - y_inf)**2
					# SE_mask_tmp = (pred_y_mask - y_inf_mask)**2
					Lambda_tmp = np.sqrt(m_tmp) * ( SE_tmp.std()**2 + SE_mask_tmp.std()**2 )**(-1/2)*( SE_tmp.mean() - SE_mask_tmp.mean() )
					p_value_tmp = norm.cdf(Lambda_tmp)
					# p_value_neg = norm.cdf(-Lambda_tmp)
					P_value.append(p_value_tmp)
				
				P_value = np.array(P_value)
				## compute the type 1 error
				Err1 = len(P_value[P_value<self.alpha])/len(P_value)
				Err1_lst.append(Err1)
				ratio_lst.append(ratio_tmp)
				
				if verbose==1:
					print('Type 1 error: %.3f; p_value: %.3f, inference sample ratio: %.3f' %(Err1, P_value.mean(), ratio_tmp))
				
				if Err1 <= self.alpha:
					found = 1
					m_opt = m_tmp
					n_opt = len(X) - 2*m_opt
					break

			if found==0:
				warnings.warn("No ratio can control the Type 1 error, pls increase the sample size, and inference sample ratio is set as the min of ratio_grid.")
				Err1_lst, ratio_lst = np.array(Err1_lst), np.array(ratio_lst)
				m_opt = int(ratio_lst[np.argmin(Err1_lst)] * len(X))
				n_opt = len(X) - 2*m_opt
				
			if m_opt < min_inf:
				warnings.warn("The estimated inference sample is too small, pls increase the sample size, and inference sample is set as 100")
				m_opt = min_inf
				n_opt = len(X) - 2*m_opt
			return n_opt, m_opt


		if method_ == 'perm_abs':
			candidate = []
			found = 0
			for ratio_tmp in reversed(ratio_grid):
				self.reset_model()
				m_tmp = int(len(X)*ratio_tmp)
				n_tmp = len(X) - 2*m_tmp
				# split data
				X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_tmp, random_state=42)
				# permutate training sample
				y_train_perm = np.random.permutation(y_train)
				# training for full model
				history = self.model.fit(x=X_train, y=y_train_perm, **fit_params)
				# training for mask model
				Z_train = self.mask_cov(X_train, k)
				history_mask = self.model_mask.fit(x=Z_train, y=y_train_perm, **fit_params)
				## evaluate the performance
				P_value = []

				Z_test = self.mask_cov(X_test, k)
				pred_y = self.model.predict(X_test).flatten()
				pred_y_mask = self.model_mask.predict(Z_test).flatten()
				print(len(pred_y), len(pred_y_mask))
				for t in range(num_perm):
					# permutate testing sample
					y_test_perm = np.random.permutation(y_test)
					# split two sample
					ind_inf, ind_inf_mask = train_test_split(range(len(y_test_perm)), train_size=m_tmp, random_state=42)
					# X_inf, X_inf_mask, y_inf, y_inf_mask = train_test_split(X_test, y_test_perm, train_size=m_tmp, random_state=42)
					# Z_inf = self.mask_cov(X_inf_mask, k)
					# evaluation
					SE_tmp = (y_test_perm[ind_inf] - pred_y[ind_inf])**2
					SE_mask_tmp = (y_test_perm[ind_inf_mask] - pred_y_mask[ind_inf_mask])**2
					# pred_y = self.model.predict(X_inf).flatten()
					# pred_y_mask = self.model_mask.predict(Z_inf).flatten()
					# SE_tmp = (pred_y - y_inf)**2
					# SE_mask_tmp = (pred_y_mask - y_inf_mask)**2
					Lambda_tmp = np.sqrt(m_tmp) * ( SE_tmp.std()**2 + SE_mask_tmp.std()**2 )**(-1/2)*( SE_tmp.mean() - SE_mask_tmp.mean() )
					p_value_tmp = norm.cdf(Lambda_tmp)
					# p_value_neg = norm.cdf(-Lambda_tmp)
					P_value.append(p_value_tmp)
				P_value = np.array(P_value)
				## compute the type 1 error
				Err1 = len(P_value[P_value<self.alpha])/len(P_value)

				if verbose==1:
					print('Type 1 error: %.3f; p_value: %.3f, inference sample ratio: %.3f' %(Err1, P_value.mean(), ratio_tmp))
				if Err1 <= self.alpha:
					found = 1
					candidate.append([m_tmp, np.abs(P_value.mean()-.5)])
			
			if found == 1:
				candidate = np.array(candidate)
				m_opt = int(candidate[np.argmin(candidate[:,1]),0])
				n_opt = len(X) - 2*m_opt
			else:
				warnings.warn("No ratio can control the Type 1 error, pls increase the sample size, and inference sample ratio is set as the min of ratio_grid.")
				m_opt=min_inf
				n_opt = len(X) - 2*m_opt

			if m_opt < min_inf:
				warnings.warn("The estimated inference sample is too small, pls increase the sample size, and inference sample is set as 100")
				m_opt = min_inf
				n_opt = len(X) - 2*m_opt
			return n_opt, m_opt


	# def adaRatio(self, X, y, k=0, fit_params={}, num_perm=100, ratio_grid=[.1, .2, .3, .4], min_inf=50, verbose=0):
	# 	found = 0
	# 	for ratio_tmp in reversed(ratio_grid):
	# 		self.reset_model()
	# 		m_tmp = int(len(X)*ratio_tmp)
	# 		n_tmp = len(X) - 2*m_tmp
	# 		# split data
	# 		X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_tmp, random_state=42)
	# 		# permutate training sample
	# 		y_train_perm = np.random.permutation(y_train)
	# 		# training for full model
	# 		history = self.model.fit(x=X_train, y=y_train_perm, **fit_params)
	# 		# training for mask model
	# 		Z_train = self.mask_cov(X_train, k)
	# 		history_mask = self.model_mask.fit(x=Z_train, y=y_train_perm, **fit_params)
	# 		## evaluate the performance
	# 		P_value = []
	# 		for t in range(num_perm):
	# 			# permutate testing sample
	# 			y_test_perm = np.random.permutation(y_test)
	# 			# split two sample
	# 			X_inf, X_inf_mask, y_inf, y_inf_mask = train_test_split(X_test, y_test_perm, train_size=m_tmp, random_state=42)
	# 			Z_inf = self.mask_cov(X_inf_mask, k)
	# 			# evaluation
	# 			pred_y = self.model.predict(X_inf).flatten()
	# 			pred_y_mask = self.model_mask.predict(Z_inf).flatten()
	# 			SE_tmp = (pred_y - y_inf)**2
	# 			SE_mask_tmp = (pred_y_mask - y_inf_mask)**2
	# 			Lambda_tmp = np.sqrt(m_tmp) * ( SE_tmp.std()**2 + SE_mask_tmp.std()**2 )**(-1/2)*( SE_tmp.mean() - SE_mask_tmp.mean() )
	# 			p_value_tmp = norm.cdf(Lambda_tmp)
	# 			P_value.append(p_value_tmp)
	# 		P_value = np.array(P_value)
	# 		## compute the type 1 error
	# 		Err1 = len(P_value[P_value<self.alpha])/len(P_value)
	# 		if verbose==1:
	# 			print('Type 1 error: %.3f; inference sample ratio: %.3f' %(Err1, ratio_tmp))
	# 		if Err1 < self.alpha:
	# 			found = 1
	# 			break
	# 	if found==0:
	# 		warnings.warn("No ratio can control the Type 1 error, pls increase the sample size, and inference sample ratio is set as the min of ratio_grid.")
	# 	if m_tmp < min_inf:
	# 		warnings.warn("The estimated inference sample is too small, pls increase the sample size, and inference sample is set as 100")
	# 		m_tmp = min_inf
	# 		n_tmp = len(X) - 2*m_tmp
	# 	return n_tmp, m_tmp


	def testing(self, X, y, pred_size=None, inf_size=None, fit_params={}, split_params={}):
		P_value = []
		for k in range(len(self.inf_cov)):
			if (pred_size == None) or (inf_size == None):
				n, m = self.adaRatio(X, y, k, fit_params=fit_params, **split_params)
				print('%d-th inference; Adaptive data splitting: n: %d; m: %d' %(k, n, m))
			else:
				n, m = pred_size, inf_size
			self.reset_model()
			X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n, random_state=42)
			X_inf, X_inf_mask, y_inf, y_inf_mask = train_test_split(X_test, y_test, train_size=m, random_state=42)
			## prediction and inference in full model
			history = self.model.fit(X_train, y_train, **fit_params)
			pred_y = self.model.predict(X_inf).flatten()
			SE = (pred_y - y_inf)**2
			# prediction and inference in mask model
			Z_train = self.mask_cov(X_train, k)
			history_mask = self.model_mask.fit(Z_train, y_train, **fit_params)
			
			Z_inf = self.mask_cov(X_inf_mask, k)
			pred_y_mask = self.model_mask.predict(Z_inf).flatten()
			SE_mask = (pred_y_mask - y_inf_mask)**2
			## compute p-value
			Lambda = np.sqrt(m) * ( SE.std()**2 + SE_mask.std()**2 )**(-1/2)*( SE.mean() - SE_mask.mean())
			print('SE: %.3f(%.3f); SE_mask: %.3f(%.3f)' %(SE.mean(), SE.std(), SE_mask.mean(), SE_mask.std()))
			p_value_tmp = norm.cdf(Lambda)

			if p_value_tmp < self.alpha:
				print('reject H0 with p_value: %.3f' %p_value_tmp)
			else:
				print('accept H0 with p_value: %.3f' %p_value_tmp)

			P_value.append(p_value_tmp)
		return P_value, SE.mean()


