## funs.py

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t
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
from sklearn.model_selection import KFold
import itertools 

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
		np.fill_diagonal(cov, 1.)
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