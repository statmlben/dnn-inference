import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import glorot_uniform
import tensorflow as tf
from sklearn.model_selection import KFold
import os

class HPermT(object):
	"""Class for holdout permutation test (HPT) based on deep neural networks.

	Parameters
	----------

	inf_feats : list-like of shape (num of tests, dim of features)
	 List of covariates/Features under hypothesis testings, one element corresponding to a hypothesis testing.

	model : {keras-defined neural network}
	 A neural network for original full dataset

	alpha: float (0,1), default=0.05
	 The nominal level of the hypothesis testing

	verbose: {0, 1}, default=0
	 If print the testing results, 1 indicates YES, 0 indicates NO.

	eva_metric: {'mse', 'zero-one', 'cross-entropy', or custom metric function}
	 The evaluation metric, ``'mse'`` is the l2-loss for regression, ``'zero-one'`` is the zero-one loss for classification, ``'cross-entropy'`` is log-loss for classification. It can also be custom metric function as ``eva_metric(y_true, y_pred)``.

	cp_path: {string}, default='./HPT_checkpoints'
	 The checkpoints path to save the models
	"""

	def __init__(self, inf_feats, model, alpha=.05, num_perm=1000, verbose=0, eva_metric='mse', cp_path = './HPT_checkpoints'):
		self.inf_feats = inf_feats
		self.model = model
		self.alpha = alpha
		self.num_perm = num_perm
		self.eva_metric = eva_metric
		self.cp_path = cp_path

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

	def save_init(self):
		"""
		Save the initialization for the network model under class HPT
		"""
		self.model.save_weights(self.cp_path+'/model_init.h5')
		# self.model_mask.save_weights(self.cp_path+'/model_mask_init.h5')

	def reset_model(self):
		"""
		Reset the full and mask network models under class HPT
		"""
		self.model.load_weights(self.cp_path+'/model_init.h5')
		# self.model_mask.load_weights(self.cp_path+'/model_mask_init.h5')

	# def reset_model(self):
	# 	if int(tf.__version__[0]) == 2:
	# 		# for layer in self.model.layers:
	# 		# 	if isinstance(layer, tf.keras.Model):
	# 		# 		reset_weights(layer)
	# 		# 		continue
	# 		# 	for k, initializer in layer.__dict__.items():
	# 		# 		if "initializer" not in k:
	# 		# 			continue
	# 		# 			# find the corresponding variable
	# 		# 		var = getattr(layer, k.replace("_initializer", ""))
	# 		# 		var.assign(initializer(var.shape, var.dtype))
	#
	# 		for layer in self.model.layers:
	# 			if isinstance(layer, tf.keras.Model): #if you're using a model as a layer
	# 				reset_weights(layer) #apply function recursively
	# 				continue
	#
	# 			#where are the initializers?
	# 			if hasattr(layer, 'cell'):
	# 				init_container = layer.cell
	# 			else:
	# 				init_container = layer
	#
	# 			for key, initializer in init_container.__dict__.items():
	# 				if "initializer" not in key: #is this item an initializer?
	# 				  continue #if no, skip it
	#
	# 				# find the corresponding variable, like the kernel or the bias
	# 				if key == 'recurrent_initializer': #special case check
	# 					var = getattr(init_container, 'recurrent_kernel')
	# 				else:
	# 					var = getattr(init_container, key.replace("_initializer", ""))
	#
	# 				if var is None:
	# 					continue
	# 				else:
	# 					var.assign(initializer(var.shape, var.dtype))
	#
	# 	if int(tf.__version__[0]) == 1:
	# 		session = K.get_session()
	# 		for layer in self.model.layers:
	# 			if hasattr(layer, 'kernel_initializer'):
	# 				layer.kernel.initializer.run(session=session)
	# 			if hasattr(layer, 'bias_initializer'):
	# 				layer.bias.initializer.run(session=session)
	# 		for layer in self.model_perm.layers:
	# 			if hasattr(layer, 'kernel_initializer'):
	# 				layer.kernel.initializer.run(session=session)
	# 			if hasattr(layer, 'bias_initializer'):
	# 				layer.bias.initializer.run(session=session)

	## can be extent to @abstractmethod
	# def mask_cov(self, X, k=0):
	# 	"""
	# 	Return instances with masked k-th hypothesized features.
	#
	# 	Parameters
	# 	----------
	# 	X : array-like
	# 	 Target instances.
	#
	# 	k : integer, default = 0
	# 	 k-th hypothesized features in inf_feats
	# 	"""
	# 	Z = X.copy()
	# 	if type(self.inf_feats[k]) is list:
	# 		Z[:,self.inf_feats[k][0][:,None], self.inf_feats[k][1], 0] = 0.
	# 	else:
	# 		Z[:,self.inf_feats[k]]= 0.
	# 	return Z

	def perm_cov(self, X, k=0):
		"""
		Return instances with permuted k-th hypothesized features.

		Parameters
		----------
		X : array-like
		 Target instances.

		k : integer, default = 0
		 k-th hypothesized features in inf_feats
		"""
		Z = X.copy()
		if type(self.inf_feats[k]) is list:
			## for channels_last image data: shape should be (#samples, img_rows, img_cols, channel)
			Z[:,self.inf_feats[k][0][:,None], self.inf_feats[k][1], :] = np.random.permutation(Z[:,self.inf_feats[k][0][:,None], self.inf_feats[k][1], :])
		else:
			Z[:,self.inf_feats[k]]= np.random.permutation(Z[:,self.inf_feats[k]])
		return Z

	def noise_cov(self, X, k=0):
		Z = X.copy()
		Z[:,self.inf_feats[k]] = np.random.randn(len(X), len(self.inf_feats[k]))
		return Z

	def testing(self, X, y, fit_params, cv_num=5, cp='hommel', inf_ratio=.2):
		"""
		Return p-values for hypothesis testing for inf_feats in class HpermT.

		Parameters
		----------

		X : array-like | shape = (n_samples, dim_features)
	 		Instances matrix/tensor, where n_samples in the number of samples and dim_features is the dimension of the features.
			 If X is vectorized feature, ``shape`` should be ``(#Samples, dim of feaures)``
			 If X is image/matrix data, ``shape`` should be ``(#samples, img_rows, img_cols, channel)``, that is, **X must channel_last image data**.	- **y: {array-like} of shape (n_samples,)**
			 Output vector/matrix relative to X.

		fit_params: {dict of fitting parameters}
	 		See keras ``fit``: (https://keras.rstudio.com/reference/fit.html), including ``batch_size``, ``epoch``, ``callbacks``, ``validation_split``, ``validation_data``, and so on.

		cv_num: int, default=5
			The number of cross-validation to shuffle the estimation/inference samples in testing.

		cp: {'gmean', 'min', 'hmean', 'Q1', 'hommel', 'cauchy'}, default ='hommel'
			A method to combine p-values obtained from cross-validation.

		inf_ratio: float, default=None
			A pre-specific inference sample ratio, if ``est_size=None``, then it is determined by adaptive splitting method ``metric``.

		Return
		------

		P_value: array of float [0, 1]
			The p_values for target hypothesis testings.

		"""

		## create checkpoints path
		if not os.path.exists(self.cp_path):
			os.mkdir(self.cp_path)
		## save initial weights
		self.save_init()

		P_value = []
		for k in range(len(self.inf_feats)):
			self.reset_model()
			m, n = int(inf_ratio * len(X)), len(X) - int(inf_ratio * len(X))
			P_value_cv = []
			score_cv, score_perm_cv = [], []
			for h in range(cv_num):
				X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n, random_state=h)
				## prediction and inference in full model
				self.reset_model()
				history = self.model.fit(X_train, y_train, **fit_params)
				## save and load model
				path_tmp = self.cp_path+'/model'+'_inf'+str(k)+'_cv'+str(h)+'.h5'
				self.model.save_weights(path_tmp)
				self.model.load_weights(path_tmp)
				## prediction for original dataset
				pred_y = self.model.predict(X_test)
				metric_full = self.metric(y_test, pred_y)
				score = metric_full.mean()
				score_cv.append(score)

				score_perm = []
				for l in range(self.num_perm):
					Z_test = self.perm_cov(X_test, k)
					pred_y_perm = self.model.predict(Z_test)
					# pred_y_perm = self.model(Z_tmp, training=False)
					metric_perm = self.metric(y_test, pred_y_perm)
					score_perm.append(metric_perm.mean())
				score_perm_cv.append(score_perm)

			score_cv = np.array(score_cv)
			score_perm_cv = np.array(score_perm_cv)
			cv_ave_score, cv_ave_score_perm = score_cv.mean(), np.mean(score_perm_cv, axis=0)
			## compute p-value
			print("%d th inf-feats perf score: %.3f, perf permutation score: %.3f(%.3f); num_perm: %d" %(k, cv_ave_score, cv_ave_score_perm.mean(), cv_ave_score_perm.std(), len(cv_ave_score_perm) ))
			p_value_tmp = (np.sum(cv_ave_score_perm <= cv_ave_score) + 1.0) / (self.num_perm + 1.0)
			if p_value_tmp < self.alpha:
				print('reject %d th H0 with p_value: %.3f' %(k, p_value_tmp))
			else:
				print('accept %d th H0 with p_value: %.3f' %(k, p_value_tmp))
			P_value.append(p_value_tmp)
		# return P_value
		self.p_values = P_value
		return P_value
