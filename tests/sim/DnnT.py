import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import warnings
import keras.backend as K
from keras.initializers import glorot_uniform
import tensorflow as tf
from sklearn.model_selection import KFold

class DnnT(object):
	def __init__(self, inf_cov, model, model_mask, change='mask', alpha=.05, verbose=0, eva_metric='mse'):
		self.inf_cov = inf_cov
		self.model = model
		self.model_mask = model_mask
		self.alpha = alpha
		self.change = change
		self.eva_metric = eva_metric

	def metric(self, y_true, y_pred):
		if self.eva_metric == 'mse':
			metric_tmp = ((y_true - y_pred)**2).flatten()
		elif self.eva_metric == 'mae':
			metric_tmp = abs(y_true - y_pred).flatten()
		elif self.eva_metric == 'zero-one':
			label_pred = np.argmax(y_pred, 1)
			label_true = np.argmax(y_true, 1)
			metric_tmp = 1. - 1.*(label_true == label_pred)
		elif self.eva_metric == 'cross-entropy':
			label_true = np.argmax(y_true, 1)
			metric_tmp = np.log(y_pred[range(len(y_pred)),label_true])
		else:
			metric_tmp = self.eva_metric(y_true, y_pred)
		return metric_tmp

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
				if ((hasattr(layer, 'kernel_initializer')) and (layer.kernel != None)):
					layer.kernel.initializer.run(session=session)
				if ((hasattr(layer, 'bias_initializer')) and (layer.bias != None)):
					layer.bias.initializer.run(session=session)     
			for layer in self.model_mask.layers:
				if ((hasattr(layer, 'kernel_initializer')) and (layer.kernel != None)):
					layer.kernel.initializer.run(session=session)
				if ((hasattr(layer, 'bias_initializer')) and (layer.bias != None)):
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

	def adaRatio(self, X, y, k=0, fit_params={}, perturb=.001, split='two-sample', 
				num_perm=100, perturb_grid=[.01, .05, .1, .5, 1.], ratio_grid=[.2, .3, .4], 
				min_inf=0, min_est=0, ratio_method='fuse', cv_num=1, verbose=0):
		candidate, Err1_lst, ratio_lst = [], [], []
		found = 0
		if split == 'two-sample':
			for ratio_tmp in reversed(ratio_grid):
				m_tmp = int(len(X)*ratio_tmp)
				if m_tmp < min_inf:
					continue
				n_tmp = len(X) - 2*m_tmp
				if n_tmp < min_est:
					continue
				# split data
				P_value = []
				for h in range(cv_num):
					self.reset_model()
					P_value_cv = []
					X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_tmp, random_state=1)
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
					
					pred_y = self.model.predict(X_test)
					pred_y_mask = self.model_mask.predict(Z_test)
					for j in range(num_perm):
						# permutate testing sample
						y_test_perm = np.random.permutation(y_test)
						# split two sample
						ind_inf, ind_inf_mask = train_test_split(range(len(y_test)), train_size=m_tmp, random_state=42)
						# evaluation
						metric_tmp = self.metric(y_test_perm[ind_inf], pred_y[ind_inf])
						metric_mask_tmp = self.metric(y_test_perm[ind_inf_mask], pred_y_mask[ind_inf_mask])
						diff_tmp = metric_tmp - metric_mask_tmp
						Lambda_tmp = np.sqrt(len(diff_tmp)) * ( diff_tmp.std() )**(-1)*( diff_tmp.mean() )
						p_value_tmp = norm.cdf(Lambda_tmp)
						P_value_cv.append(p_value_tmp)
					P_value.append(P_value_cv)
				
				P_value = np.array(P_value)
				P_value = np.mean(P_value, 0)
				## compute the type 1 error
				Err1 = len(P_value[P_value < self.alpha]) / len(P_value)
				Err1_lst.append(Err1)
				ratio_lst.append(ratio_tmp)
				
				if verbose==1:
					print('Type 1 error: %.3f; p_value: %.3f, inference sample ratio: %.3f' %(Err1, P_value.mean(), ratio_tmp))
				
				if Err1 <= self.alpha:
					found = 1
					if ratio_method == 'fuse':
						m_opt = m_tmp
						n_opt = len(X) - 2*m_opt
						break

			if found == 1:
				if ratio_method == 'close':
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
					P_value = []
					for h in range(cv_num):
						P_value_cv = []
						X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_tmp, random_state=h)
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

						pred_y = self.model.predict(X_test)
						pred_y_mask = self.model_mask.predict(Z_test)

						for j in range(num_perm):
							# permutate testing sample
							y_test_perm = np.random.permutation(y_test)
							# evaluation
							metric_tmp = self.metric(y_test_perm, pred_y)
							metric_mask_tmp = self.metric(y_test_perm, pred_y_mask)

							if perturb_tmp == 'auto':
								diff_tmp = metric_tmp - metric_mask_tmp + metric_tmp.std()*np.random.randn(len(metric_tmp))
							else:
								diff_tmp = metric_tmp - metric_mask_tmp + perturb_tmp*np.random.randn(len(metric_tmp))
							
							Lambda_tmp = np.sqrt(len(diff_tmp)) * ( diff_tmp.std() )**(-1)*( diff_tmp.mean() )
							p_value_tmp = norm.cdf(Lambda_tmp)
							P_value_cv.append(p_value_tmp)
						
						if verbose==1:
							print('cv: %d; p_value: %.3f, inference sample ratio: %.3f, perturb: %s' %(h, np.mean(P_value_cv), ratio_tmp, perturb_tmp))

						P_value.append(P_value_cv)
				
					P_value = np.array(P_value)
					P_value = np.mean(P_value, 0)
					## compute the type 1 error
					Err1 = len(P_value[P_value<self.alpha])/len(P_value)
					if verbose==1:
						print('Type 1 error: %.3f; p_value: %.3f, inference sample ratio: %.3f, perturb: %s' %(Err1, P_value.mean(), ratio_tmp, perturb_tmp))

					Err1_lst.append(Err1)
					ratio_lst.append(ratio_tmp)
					perturb_lst.append(perturb_tmp)
				

					if Err1 <= self.alpha:
						found = 1
						if ratio_method == 'fuse':
							m_opt = m_tmp
							n_opt = len(X) - m_opt
							perturb_opt = perturb_tmp
							break
				
				if found == 1:
					if ratio_method == 'min':
						Err1_lst, ratio_lst = np.array(Err1_lst), np.array(ratio_lst)
						m_opt = int(ratio_lst[np.argmin(Err1_lst)] * len(X))
						n_opt = len(X) - m_opt
						perturb_opt = perturb_tmp
						
					if ratio_method == 'close':
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

	def testing(self, X, y, fit_params, split_params, cv_num=1, est_size=None, inf_size=None):
		P_value = []
		for k in range(len(self.inf_cov)):
			self.reset_model()
			if split_params['split'] == 'one-sample':
				if (est_size == None) or (inf_size == None):
					n, m, perturb_level = self.adaRatio(X, y, k, fit_params=fit_params, **split_params)
					print('%d-th inference; Adaptive data splitting: n: %d; m: %d; perturb: %s' %(k, n, m, perturb_level))
				else:
					n, m, perturb_level = est_size, inf_size, split_params['perturb']
			
			if split_params['split'] == 'two-sample':
				if (est_size == None) or (inf_size == None):
					n, m = self.adaRatio(X, y, k, fit_params=fit_params, **split_params)
					print('%d-th inference; Adaptive data splitting: n: %d; m: %d' %(k, n, m))
				else:
					n, m = est_size, inf_size
			
			P_value_cv = []
			for h in range(cv_num):
				X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n, random_state=h)
				if split_params['split'] == 'two-sample':
					X_inf, X_inf_mask, y_inf, y_inf_mask = train_test_split(X_test, y_test, train_size=m, random_state=42)
				if split_params['split'] == 'one-sample':
					X_inf, X_inf_mask, y_inf, y_inf_mask = X_test.copy(), X_test.copy(), y_test.copy(), y_test.copy()
				## prediction and inference in full model
				self.reset_model()
				history = self.model.fit(X_train, y_train, **fit_params)

				pred_y = self.model.predict(X_inf)
				metric_full = self.metric(y_inf, pred_y)

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
				
				pred_y_mask = self.model_mask.predict(Z_inf)
				metric_mask = self.metric(y_inf_mask, pred_y_mask)

				## compute p-value
				if split_params['split'] == 'one-sample':
					if perturb_level == 'auto':
						diff_tmp = metric_full - metric_mask + metric_full.std() * np.random.randn(len(metric_full))
					else:
						diff_tmp = metric_full - metric_mask + perturb_level * np.random.randn(len(metric_full))
				
				if split_params['split'] == 'two-sample':
					diff_tmp = metric_full - metric_mask
				
				Lambda = np.sqrt(len(diff_tmp)) * ( diff_tmp.std() )**(-1)*( diff_tmp.mean() )
				p_value_tmp = norm.cdf(Lambda)
				print('cv: %d; p_value: %.3f; diff: %.3f(%.3f); metric: %.3f(%.3f); metric_mask: %.3f(%.3f)' %(h, p_value_tmp, diff_tmp.mean(), diff_tmp.std(), metric_full.mean(), metric_full.std(), metric_mask.mean(), metric_mask.std()))

				P_value_cv.append(p_value_tmp)

			p_value_mean = np.mean(P_value_cv)
			if p_value_mean < self.alpha:
				print('reject H0 with p_value: %.3f' %p_value_mean)
			else:
				print('accept H0 with p_value: %.3f' %p_value_mean)

			P_value.append(p_value_mean)
		return P_value