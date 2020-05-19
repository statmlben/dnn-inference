import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import warnings
import keras.backend as K
from keras.initializers import glorot_uniform
import tensorflow as tf
from sklearn.model_selection import KFold

class DeepT(object):
	def __init__(self, inf_cov, model, model_mask, change='mask', alpha=.05, verbose=0, eva_metric='mse'):
		self.inf_cov = inf_cov
		self.model = model
		self.model_mask = model_mask
		self.alpha = alpha
		self.change = change
		self.eva_metric = eva_metric

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

			for layer in self.model_mask.layers:
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
				if hasattr(layer, 'bias_initializer'):
					layer.bias.initializer.run(session=session)     
			for layer in self.model_mask.layers:
				if hasattr(layer, 'kernel_initializer'):
					layer.kernel.initializer.run(session=session)
				if hasattr(layer, 'bias_initializer'):
					layer.bias.initializer.run(session=session)  


	## can be extent to @abstractmethod
	def mask_cov(self, X, k=0):
		Z = X.copy()
		if type(self.inf_cov[k]) is list:
			Z[:, self.inf_cov[k][0][:,None], self.inf_cov[k][1], 0] = 0.
		else:
			Z[:,self.inf_cov[k]]= 0.
		return Z

	def perm_cov(self, X, k=0):
		Z = X.copy()
		if type(self.inf_cov[k]) is list:
			Z[:,self.inf_cov[k][0][:,None], self.inf_cov[k][1], 0]= np.random.permutation(Z[:,self.inf_cov[k][0][:,None], self.inf_cov[k][1], 0])
		else:
			Z[:,self.inf_cov[k]]= np.random.permutation(Z[:,self.inf_cov[k]])
		return Z

	def noise_cov(self, X, k=0):
		Z = X.copy()
		Z[:,self.inf_cov[k]] = np.random.randn(len(X), len(self.inf_cov[k]))
		return Z

	def adaRatio(self, X, y, k=0, fit_params={}, perturb=.001, split='one-sample', num_perm=100, perturb_grid=[.01, .1, 1.], ratio_grid=[.1, .2, .3, .4], min_inf=0, min_est=0, metric='fuse', verbose=0):
		candidate, Err1_lst, ratio_lst = [], [], []
		found = 0
		if split == 'two-sample':
			for ratio_tmp in reversed(ratio_grid):
				self.reset_model()
				m_tmp = int(len(X)*ratio_tmp)
				if m_tmp < min_inf:
					continue
				n_tmp = len(X) - 2*m_tmp
				if n_tmp < min_est:
					continue
				# split data
				X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_tmp, random_state=42)
				# permutate training sample
				y_train_perm = np.random.permutation(y_train)
				# training for full model
				history = self.model.fit(x=X_train, y=y_train_perm, **fit_params)
				
				# training for mask model
				if self.change == 'mask':
					Z_train = self.mask_cov(X_train, k)
				if self.change == 'perm':
					Z_train = self.perm_cov(X_train, k)
				history_mask = self.model_mask.fit(x=Z_train, y=y_train_perm, **fit_params)
				
				## evaluate the performance
				P_value = []
				if self.change == 'mask':
					Z_test = self.mask_cov(X_test, k)
				if self.change == 'perm':
					Z_test = self.perm_cov(X_test, k)
				
				if self.eva_metric == 'mse':
					pred_y = self.model.predict(X_test).flatten()
					pred_y_mask = self.model_mask.predict(Z_test).flatten()
				
				if self.eva_metric == 'zero-one':
					pred_y = self.model.predict(X_test)
					pred_y_mask = self.model_mask.predict(Z_test)
					pred_label = np.argmax(pred_y, 1)
					pred_label_mask = np.argmax(pred_y_mask, 1)
					label_test = np.argmax(y_test, 1)
				
				for j in range(num_perm):
					# permutate testing sample
					if self.eva_metric == 'mse':
						y_test_perm = np.random.permutation(y_test)
					if self.eva_metric == 'zero-one':
						label_test_perm = np.random.permutation(label_test)
					# split two sample
					ind_inf, ind_inf_mask = train_test_split(range(len(y_test)), train_size=m_tmp, random_state=42)
					# evaluation
					if self.eva_metric == 'mse':
						metric_tmp = (y_test_perm[ind_inf] - pred_y[ind_inf])**2
						metric_mask_tmp = (y_test_perm[ind_inf_mask] - pred_y_mask[ind_inf_mask])**2
					if self.eva_metric == 'zero-one':
						metric_tmp = 1. - 1.*(label_test_perm[ind_inf] == pred_label[ind_inf])
						metric_mask_tmp = 1. - 1.*(label_test_perm[ind_inf_mask] == pred_label[ind_inf_mask])
					diff_tmp = metric_tmp - metric_mask_tmp
					Lambda_tmp = np.sqrt(len(diff_tmp)) * ( diff_tmp.std() )**(-1)*( diff_tmp.mean() )
					p_value_tmp = norm.cdf(Lambda_tmp)
					P_value.append(p_value_tmp)
				
				P_value = np.array(P_value)
				## compute the type 1 error
				Err1 = len(P_value[P_value < self.alpha]) / len(P_value)
				Err1_lst.append(Err1)
				ratio_lst.append(ratio_tmp)
				
				if verbose==1:
					print('Type 1 error: %.3f; p_value: %.3f, inference sample ratio: %.3f' %(Err1, P_value.mean(), ratio_tmp))
				
				if Err1 <= self.alpha:
					found = 1
					if metric == 'fuse':
						m_opt = m_tmp
						n_opt = len(X) - 2*m_opt
						break

			if found == 1:
				if metric == 'close':
					Err1_lst, ratio_lst = np.array(Err1_lst), np.array(ratio_lst)
					Err1_lst, ratio_lst = Err1_lst[Err1_lst <= self.alpha], ratio_lst[Err1_lst <= self.alpha]
					m_opt = int(ratio_lst[np.argmin(Err1_lst)] * len(X))
					n_opt = len(X) - 2*m_opt

			if found==0:
				warnings.warn("No ratio can control the Type 1 error, pls increase the sample size, and inference sample ratio is set as the min of ratio_grid.")
				Err1_lst, ratio_lst = np.array(Err1_lst), np.array(ratio_lst)
				m_opt = int(ratio_lst[np.argmin(Err1_lst)] * len(X))
				n_opt = len(X) - 2*m_opt
			
			return n_opt, m_opt

		if split == 'one-sample':
			if perturb != None:
				perturb_grid = [perturb]
			for perturb_tmp in perturb_grid:
				## stop if current perturb is enough to control the type 1 error
				if found == 1:
					break
				Err1_lst, ratio_lst, perturb_lst = [], [], []
				for ratio_tmp in reversed(ratio_grid):
					self.reset_model()
					m_tmp = int(len(X)*ratio_tmp)
					if m_tmp < min_inf:
						continue
					n_tmp = len(X) - m_tmp
					if n_tmp < min_est:
						continue
					# split data
					X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_tmp, random_state=42)
					# permutate training sample
					y_train_perm = np.random.permutation(y_train)
					# training for full model
					history = self.model.fit(x=X_train, y=y_train_perm, **fit_params)
					# training for mask model
					if self.change == 'mask':
						Z_train = self.mask_cov(X_train, k)
					if self.change == 'perm':
						Z_train = self.perm_cov(X_train, k)
					history_mask = self.model_mask.fit(x=Z_train, y=y_train_perm, **fit_params)
					
					## evaluate the performance
					if self.change == 'mask':
						Z_test = self.mask_cov(X_test, k)
					if self.change == 'perm':
						Z_test = self.perm_cov(X_test, k)

					if self.eva_metric == 'mse':
						pred_y = self.model.predict(X_test).flatten()
						pred_y_mask = self.model_mask.predict(Z_test).flatten()
					if self.eva_metric == 'zero-one':
						pred_y = self.model.predict(X_test)
						pred_y_mask = self.model.predict(Z_test)
						label_test = np.argmax(y_test, 1)
						pred_label = np.argmax(pred_y, 1)
						pred_label_mask = np.argmax(pred_y_mask, 1)

					P_value = []
					for j in range(num_perm):
						# permutate testing sample
						if self.eva_metric == 'mse':
							y_test_perm = np.random.permutation(y_test)
						if self.eva_metric == 'zero-one':
							label_test_perm = np.random.permutation(label_test)
						# evaluation
						if self.eva_metric == 'zero-one':
							metric_tmp = 1. - 1.*(label_test_perm == pred_label)
							metric_mask_tmp = 1. - 1.*(label_test_perm == pred_label_mask)
							
						if perturb_tmp == 'auto':
							diff_tmp = metric_tmp - metric_mask_tmp + metric_tmp.std()*np.random.randn(len(metric_tmp))
						else:
							diff_tmp = metric_tmp - metric_mask_tmp + perturb_tmp*np.random.randn(len(metric_tmp))
						
						Lambda_tmp = np.sqrt(len(diff_tmp)) * ( diff_tmp.std() )**(-1)*( diff_tmp.mean() )
						p_value_tmp = norm.cdf(Lambda_tmp)
						P_value.append(p_value_tmp)
					
					# print('diff: %.3f(%.3f); SE: %.3f(%.3f); SE_mask: %.3f(%.3f)' %(diff_tmp.mean(), diff_tmp.std(), SE_tmp.mean(), SE_tmp.std(), SE_mask_tmp.mean(), SE_mask_tmp.std()))
					P_value = np.array(P_value)
					## compute the type 1 error
					Err1 = len(P_value[P_value<self.alpha])/len(P_value)
					Err1_lst.append(Err1)
					ratio_lst.append(ratio_tmp)
					perturb_lst.append(perturb_tmp)
				
					if verbose==1:
						print('Type 1 error: %.3f; p_value: %.3f, inference sample ratio: %.3f, perturb: %s' %(Err1, P_value.mean(), ratio_tmp, perturb_tmp))
					
					if Err1 <= self.alpha:
						found = 1
						if metric == 'fuse':
							m_opt = m_tmp
							n_opt = len(X) - m_opt
							perturb_opt = perturb_tmp
							break
				
				if found == 1:
					if metric == 'min':
						Err1_lst, ratio_lst = np.array(Err1_lst), np.array(ratio_lst)
						m_opt = int(ratio_lst[np.argmin(Err1_lst)] * len(X))
						n_opt = len(X) - m_opt
						perturb_opt = perturb_tmp
						
					if metric == 'close':
						Err1_lst, ratio_lst, perturb_lst = np.array(Err1_lst), np.array(ratio_lst), np.array(perturb_lst)
						Err1_lst, ratio_lst, perturb_lst = Err1_lst[Err1_lst <= self.alpha], ratio_lst[Err1_lst <= self.alpha], perturb_lst[Err1_lst <= self.alpha]
						Err1_lst = self.alpha - Err1_lst
						m_opt = int(ratio_lst[np.argmin(Err1_lst)] * len(X))
						n_opt = len(X) - m_opt
						perturb_opt = perturb_lst[np.argmin(Err1_lst)]

			if found==0:
				warnings.warn("No ratio and perturb_level can control the Type 1 error, pls increase the perturb_level and sample size, and inference sample ratio is set as the min of ratio_grid.")
				Err1_lst, ratio_lst = np.array(Err1_lst), np.array(ratio_lst)
				m_opt = int(ratio_lst[np.argmin(Err1_lst)] * len(X))
				n_opt = len(X) - m_opt
				perturb_opt = perturb_lst[np.argmin(Err1_lst)]
		
			return n_opt, m_opt, perturb_opt

	def testing(self, X, y, fit_params, split_params, pred_size=None, inf_size=None):
		P_value = []
		for k in range(len(self.inf_cov)):
			self.reset_model()
			if split_params['split'] == 'one-sample':
				if (pred_size == None) or (inf_size == None):
					n, m, perturb_level = self.adaRatio(X, y, k, fit_params=fit_params, **split_params)
					print('%d-th inference; Adaptive data splitting: n: %d; m: %d; perturb: %s' %(k, n, m, perturb_level))
				else:
					n, m, perturb_level = pred_size, inf_size, split_params['perturb']
			
			if split_params['split'] == 'two-sample':
				if (pred_size == None) or (inf_size == None):
					n, m = self.adaRatio(X, y, k, fit_params=fit_params, **split_params)
					print('%d-th inference; Adaptive data splitting: n: %d; m: %d' %(k, n, m))
				else:
					n, m = pred_size, inf_size

			X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n, random_state=42)
			if split_params['split'] == 'two-sample':
				X_inf, X_inf_mask, y_inf, y_inf_mask = train_test_split(X_test, y_test, train_size=m, random_state=42)
			if split_params['split'] == 'one-sample':
				X_inf, X_inf_mask, y_inf, y_inf_mask = X_test.copy(), X_test.copy(), y_test.copy(), y_test.copy()
			## prediction and inference in full model
			self.reset_model()
			history = self.model.fit(X_train, y_train, **fit_params)
			if self.eva_metric == 'mse':
				pred_y = self.model.predict(X_inf).flatten()
				metric_full = (pred_y - y_inf)**2
			if self.eva_metric == 'zero-one':
				inf_label = np.argmax(y_inf, 1)
				pred_y = self.model.predict(X_inf)
				pred_label = np.argmax(pred_y, 1)
				metric_full = 1. - 1.*(pred_label == inf_label)

			# prediction and inference in mask model
			if self.change == 'mask':
				Z_train = self.mask_cov(X_train, k)
			if self.change == 'perm':
				Z_train = self.perm_cov(X_train, k)
			
			self.reset_model()
			history_mask = self.model_mask.fit(Z_train, y_train, **fit_params)
			
			if self.change == 'mask':
				Z_inf = self.mask_cov(X_inf_mask, k)
			if self.change == 'perm':
				Z_inf = self.perm_cov(X_inf_mask, k)
			
			if self.eva_metric == 'mse':
				pred_y_mask = self.model_mask.predict(Z_inf).flatten()
				eva_metric_mask = (pred_y_mask - y_inf_mask)**2

			if self.eva_metric == 'zero-one':
				inf_label_mask = np.argmax(y_inf_mask, 1)
				pred_y_mask = self.model_mask.predict(Z_inf)
				pred_y_label = np.argmax(pred_y_mask, 1)
				metric_mask = 1. - 1.*(pred_y_label == inf_label_mask)

			## compute p-value
			if split_params['split'] == 'one-sample':
				if perturb_level == 'auto':
					diff_tmp = metric_full - metric_mask + metric_full.std() * np.random.randn(len(metric_full))
				else:
					diff_tmp = metric_full - metric_mask + perturb_level * np.random.randn(len(metric_full))
			
			if split_params['split'] == 'two-sample':
				diff_tmp = metric_full - metric_mask
			
			Lambda = np.sqrt(len(diff_tmp)) * ( diff_tmp.std() )**(-1)*( diff_tmp.mean() )
			print('diff: %.3f(%.3f); metric: %.3f(%.3f); metric_mask: %.3f(%.3f)' %(diff_tmp.mean(), diff_tmp.std(), metric_full.mean(), metric_full.std(), metric_mask.mean(), metric_mask.std()))
			p_value_tmp = norm.cdf(Lambda)

			if p_value_tmp < self.alpha:
				print('reject H0 with p_value: %.3f' %p_value_tmp)
			else:
				print('accept H0 with p_value: %.3f' %p_value_tmp)

			P_value.append(p_value_tmp)
		return P_value, metric_full.mean()


