.. -*- mode: rst -*-

|PyPi|_ |Keras|_ |MIT| |Python3| |tensorflow|_ |downloads|_ |downloads_month|_

.. |PyPi| image:: https://badge.fury.io/py/dnn-inference.svg
.. _PyPi: https://badge.fury.io/py/dnn-inference

.. |Keras| image:: https://img.shields.io/badge/keras-tf.keras-red.svg
.. _Keras: https://keras.io/

.. |MIT| image:: https://img.shields.io/pypi/l/varsvm.svg

.. |Python3| image:: https://img.shields.io/badge/python-3-green.svg

.. |tensorflow| image:: https://img.shields.io/badge/keras-tensorflow-blue.svg
.. _tensorflow: https://www.tensorflow.org/

.. |downloads| image:: https://pepy.tech/badge/dnn-inference
.. _downloads: https://pepy.tech/project/dnn-inference
.. |downloads_month| image:: https://pepy.tech/badge/dnn-inference/month
.. _downloads_month: https://pepy.tech/project/dnn-inference

Dnn-Inference
=============

.. image:: ./logo/logo_header.png
   :align: center
   :width: 800

Dnn-Inference is a Python module for hypothesis testing based on deep neural networks.

Website: https://dnn-inference.readthedocs.io

This project was created by `Ben Dai <https://www.bendai.org/>`_. If there is any problem and suggestion please contact me via <bdai@umn.edu>.

.. image:: ./logo/demo_result.png
   :align: center
   :width: 800

Reference
---------
**If you use this code please star the repository and cite the following paper:**

.. code:: bib

 @misc{dai2021significance,
       title={Significance tests of feature relevance for a blackbox learner},
       author={Ben Dai and Xiaotong Shen and Wei Pan},
       year={2021},
       eprint={2103.04985},
       archivePrefix={arXiv},
       primaryClass={stat.ML}
 }

Installation
------------

Dependencies
~~~~~~~~~~~~

Deep-Inference requires:

- Python
- Numpy
- Keras
- Tensorflow>=1.15
- sklearn
- SciPy

User installation
~~~~~~~~~~~~~~~~~

Install Deep-Inference using ``pip`` ::

	pip install dnn-inference

or ::

	pip install git+https://github.com/statmlben/dnn-inference.git

Source code
~~~~~~~~~~~

You can check the latest sources with the command::

    git clone https://github.com/statmlben/dnn-inference.git


Documentation
-------------

DnnT
~~~~
Class for one-split/two-split test based on deep neural networks.

.. code:: python

	class dnn_inference.DnnT(inf_cov, model, model_mask, change='mask', alpha=.05, verbose=0, eva_metric='mse')

- Parameters:

	- **inf_cov: {list-like of shape (num of tests, dim of features)}**

	 List of covariates/Features under hypothesis testings, one element corresponding to a hypothesis testing.

	- **model: {keras-defined neural network}**

	 A neural network for original full dataset

	- **model_mask: {keras-defined neural network}**

	 A neural network for masked dataset by masking/changing the features under hypothesis testing

	- **change: {'mask', 'perm'}, default='mask'**

	 The way to change the testing features, ``'mask'`` replaces testing features as zeros, while ``'perm'`` permutes features via instances.

	- **alpha: float (0,1), default=0.05**

	 The nominal level of the hypothesis testing

	- **verbose: {0, 1}, default=0**

	 If print the testing results, 1 indicates YES, 0 indicates NO.

	- **eva_metric: {'mse', 'zero-one', 'cross-entropy', or custom metric function}**

	 The evaluation metric, ``'mse'`` is the l2-loss for regression, ``'zero-one'`` is the zero-one loss for classification, ``'cross-entropy'`` is log-loss for classification. It can also be custom metric function as ``eva_metric(y_true, y_pred)``.

- **Method**:

.. code:: python

	def testing(self, X, y, cv_num=5, cp='hommel', fit_params, split_params, inf_ratio=None)

Method under class ``DnnT``, conduct the hypothesis testings according to the given data.

- Parameters:

	- **X: {array-like} of shape (n_samples, dim_features)**

	 Instances matrix/tensor, where n_samples in the number of samples and dim_features is the dimension of the features.
	 If X is vectorized feature, ``shape`` should be ``(#Samples, dim of feaures)``
	 If X is image/matrix data, ``shape`` should be ``(#samples, img_rows, img_cols, channel)``, that is, **X must channel_last image data**.	- **y: {array-like} of shape (n_samples,)**
	 Output vector/matrix relative to X.

	- **fit_params: {dict of fitting parameters}**

	 See keras ``fit``: (https://keras.rstudio.com/reference/fit.html), including ``batch_size``, ``epoch``, ``callbacks``, ``validation_split``, ``validation_data``, and so on.

	- **split_params: {dict of splitting parameters}**

		- **split: {'one-split', 'two-split'}, default='one-split'**

		 one-split or two-split test statistic.

		- **perturb: float, default=None**

		 Perturb level for the one-split test, if ``perturb = None``, then the perturb level is determined by adaptive tunning.

		- **num_perm: int, default=100**

		 Number of permutation for determine the splitting ratio.

		- **ratio_grid: list of float (0,1), default=[.2, .4, .6, .8]**

		 A list of estimation/inference ratios under searching.

		- **if_reverse: {0,1}, default=0**

		 ``if_reverse = 0`` indicates the loop of ``ratio_grid`` starts from smallest one to largest one; ``if_reverse = 1`` indicates the loop of ``ratio_grid`` starts from largest one to smallest one.

		- **perturb_grid: list of float, default=[.01, .05, .1, .5, 1.]**

		 A list of perturb levels under searching.

		- **min_inf: int, default=0**

		 The minimal size for inference sample.

		- **min_est: int, default=0**

		 The minimal size for estimation sample.

		- **ratio_method: {'fuse', 'close'}, default='fuse'**

		 The adaptive splitting method to determine the optimal estimation/inference ratios.

		- **cv_num: int, default=1**

		 The number of cross-validation to shuffle the estimation/inference samples in adaptive ratio splitting.

		- **cp: {'gmean', 'min', 'hmean', 'Q1', 'hommel', 'cauchy'}, default ='hommel'**

		 A method to combine p-values obtained from cross-validation. see (https://arxiv.org/pdf/1212.4966.pdf) for more detail.

		- **verbose: {0,1}, default=1**

	- **cv_num: int, default=1**

	 The number of cross-validation to shuffle the estimation/inference samples in testing.

	- **cp: {'gmean', 'min', 'hmean', 'Q1', 'hommel', 'cauchy'}, default ='hommel'**

	 A method to combine p-values obtained from cross-validation.

	- **inf_ratio: float, default=None**

	 A pre-specific inference sample ratio, if ``est_size=None``, then it is determined by adaptive splitting method ``metric``.

- Return:

	- **P_value: array of float [0, 1]**

	 The p_values for target hypothesis testings.


PermT
~~~~~
Class for permutation testing based on deep neural networks.

**Remark:** *permutation testing break the dependence of the features, which may lead to incorrect p-values*.

.. code:: python

	class dnn_inference.PermT(inf_cov, model, model_mask, alpha=.05, num_folds=5, num_perm=100, eva_metric='mse', verbose=0)

- Parameters:

	- **inf_cov: {list-like of shape (num of tests, dim of features)}**

	 List of covariates/Features under hypothesis testings, one element corresponding to a hypothesis testing.

	- **model: {keras-defined neural network}**

	 A neural network for original full dataset

	- **model_mask: {keras-defined neural network}**

	 A neural network for masked dataset by masking/changing the features under hypothesis testing

	- **alpha: float (0,1), default=0.05**

	 The nominal level of the hypothesis testing

	- **num_folds: int, default=5**

	 Number of CV-folds to compute the score.

	- **verbose: {0, 1}, default=0**

	 If print the testing results, 1 indicates YES, 0 indicates NO.

	- **eva_metric: {'mse', 'zero-one', 'cross-entropy', or custom metric function}**

	 The evaluation metric, ``'mse'`` is the l2-loss for regression, ``'zero-one'`` is the zero-one loss for classification, ``'cross-entropy'`` is log-loss for classification. It can also be custom metric function as ``eva_metric(y_true, y_pred)``.

- **Method**:

.. code:: python

	def testing(self, X, y, fit_params)

Method under class ``DnnT``, conduct the hypothesis testings according to the given data.

- Parameters:

	- **X: {array-like}**

	 Instances matrix/tensor, where n_samples in the number of samples and dim_features is the dimension of the features.
	 If X is vectorized feature, ``shape`` should be ``(#Samples, dim of feaures)``
	 If X is image/matrix data, ``shape`` should be ``(#samples, img_rows, img_cols, channel)``, that is, **X must channel_last image data**.

	- **y: {array-like} of shape (n_samples,)**

	 Output vector/matrix relative to X.

	- **fit_params: {dict of fitting parameters}**

	 See keras ``fit``: (https://keras.rstudio.com/reference/fit.html), including ``batch_size``, ``epoch``, ``callbacks``, ``validation_split``, ``validation_data``, and so on.

- Return:

	- **P_value: array of float [0, 1]**

	 The p_values for target hypothesis testings.

Example
~~~~~~~
.. code:: python

	import numpy as np
	import keras
	from keras.datasets import mnist
	from keras.models import Sequential
	from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
	from tensorflow.python.keras import backend as K
	import time
	from sklearn.model_selection import train_test_split
	from keras.optimizers import Adam, SGD
	from dnn_inference import DnnT

	num_classes = 2

	# input image dimensions
	img_rows, img_cols = 28, 28

	# the data, split between train and test sets
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	X = np.vstack((x_train, x_test))
	y = np.hstack((y_train, y_test))
	ind = (y == 9) + (y == 7)
	X, y = X[ind], y[ind]
	X = X.astype('float32')
	X += .01*abs(np.random.randn(14251, 28, 28))
	y[y==7], y[y==9] = 0, 1

	if K.image_data_format() == 'channels_first':
		X = X.reshape(x.shape[0], 1, img_rows, img_cols)
		input_shape = (1, img_rows, img_cols)
	else:
		X = X.reshape(X.shape[0], img_rows, img_cols, 1)
		input_shape = (img_rows, img_cols, 1)

	X /= 255.

	# convert class vectors to binary class matrices
	y = keras.utils.to_categorical(y, num_classes)

	K.clear_session()

	def cnn():
		model = Sequential()
		model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(num_classes, activation='softmax'))
		model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(0.005), metrics=['accuracy'])
		return model

	tic = time.perf_counter()
	model, model_mask = cnn(), cnn()

	from keras.callbacks import EarlyStopping
	es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10, restore_best_weights=True)

	fit_params = {'callbacks': [es],
				  'epochs': 5,
				  'batch_size': 32,
				  'validation_split': .2,
				  'verbose': 1}

	inf_cov = [[np.arange(19,28), np.arange(13,20)], [np.arange(21,28), np.arange(4, 13)],
			   [np.arange(7,16), np.arange(9,16)]]

	shiing = DnnT(inf_cov=inf_cov, model=model, model_mask=model_mask, change='mask', eva_metric='zero-one')

	p_value_tmp = shiing.testing(X, y, fit_params=fit_params)
	toc = time.perf_counter()
	print('testing time: %.3f' %(toc-tic))
	shiing.visual(X, y)
