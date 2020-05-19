import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import warnings
import keras.backend as K
from keras.initializers import glorot_uniform
import tensorflow as tf
from sklearn.model_selection import KFold

class PermT(object):
	def __init__(self, inf_cov, model, model_perm, alpha=.05, num_folds=5, num_perm=100, verbose=0, eva_metric='mse'):
		self.inf_cov = inf_cov
		self.model = model
		self.model_perm = model_perm
		self.alpha = alpha
		self.num_perm = num_perm
		self.num_folds = num_folds
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
				if hasattr(layer, 'bias_initializer'):
					layer.bias.initializer.run(session=session)     
			for layer in self.model_perm.layers:
				if hasattr(layer, 'kernel_initializer'):
					layer.kernel.initializer.run(session=session)
				if hasattr(layer, 'bias_initializer'):
					layer.bias.initializer.run(session=session)  

	## can be extent to @abstractmethod
	def mask_cov(self, X, k=0):
		Z = X.copy()
		if type(self.inf_cov[k]) is list:
			Z[:,self.inf_cov[k][0][:,None], self.inf_cov[k][1], 0] = 0.
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

	def testing(self, X, y, fit_params={}):
		P_value = []
		for k in range(len(self.inf_cov)):
			kfold = KFold(n_splits=self.num_folds, shuffle=True)
			self.reset_model()
			print('%d-th permutation inference' %k)
			## prediction and inference in full model
			score_cv = []
			for train, test in kfold.split(X, y):
				self.reset_model()
				history = self.model.fit(X[train], y[train], **fit_params)
				pred_y = self.model.predict(X[test])
				metric_full = self.metric(y[test], pred_y)
				score_cv.append(metric_full.mean())
			score = np.mean(score_cv)
			# prediction and inference in mask model
			score_perm = []
			for l in range(self.num_perm):
				score_perm_cv = []
				Z = self.perm_cov(X, k)
				for train_perm, test_perm in kfold.split(Z, y):
					self.reset_model()
					history_perm = self.model_perm.fit(Z[train_perm], y[train_perm], **fit_params)
					pred_y_perm = self.model_perm.predict(Z[test_perm])
					metric_perm = self.metric(y[test_perm], pred_y_perm)
					score_perm_cv.append(metric_perm.mean())
				score_perm.append(np.mean(score_perm_cv))
			score_perm = np.array(score_perm)
			## compute p-value
			print("perf score: %.3f, perf permutation score: %.3f(%.3f)" %(score, score_perm.mean(), score_perm.std()))
			p_value_tmp = (np.sum(score_perm <= score) + 1.0) / (self.num_perm + 1)

			if p_value_tmp < self.alpha:
				print('reject H0 with p_value: %.3f' %p_value_tmp)
			else:
				print('accept H0 with p_value: %.3f' %p_value_tmp)

			P_value.append(p_value_tmp)
		return P_value, metric_full.mean()