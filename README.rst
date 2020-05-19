.. -*- mode: rst -*-

|Keras|_ |Python3| |tensorflow|_

.. |Keras| image:: https://img.shields.io/badge/keras-tf.keras-red.svg
.. _Keras: https://keras.io/

.. |Python3| image:: https://img.shields.io/badge/python-3-green.svg

.. |tensorflow| image:: https://img.shields.io/badge/keras-tensorflow-blue.svg
.. _tensorflow: https://www.tensorflow.org/

Dnn-Inference
============

Dnn-Inference is a Python module for hypothesis testing based on deep neural networks. 

This project was created by `Ben Dai <http://users.stat.umn.edu/~bdai/>`_. If there is any problem and suggestion please contact me via <bdai@umn.edu>.

Installation
------------

Dependencies
~~~~~~~~~~~~

Deep-Inference requires:

- Python
- keras
- tensorflow
- sklearn
- sciPy

User installation
~~~~~~~~~~~~~~~~~

Install Deep-Inference using ``pip`` ::

	pip install dnn-inference
	pip install git+https://github.com/statmlben/dnn-inference.git

Source code
~~~~~~~~~~~

You can check the latest sources with the command::

    git clone https://github.com/statmlben/dnn-inference.git


Documentation
------------

DnnT
~~~~~~~~~~~~
- class dnn_inference.DnnT(inf_cov, model, model_mask, change='mask', alpha=.05, verbose=0, eva_metric='mse')

	- Parameters:
		- **inf_cov: {List-like of shape (num of tests, dim of features)}** 
			List of covariates/Features under hypothesis testings, one element corresponding to a hypothesis testing.
		- **model**: a neural network for original full dataset
		- **model_mask**: a neural network for masked dataset by masking/changing the features under hypothesis testing
		- **change**: a way to changing the testing features, 


		- **beta**: Primal variable, or coefficients of the support vector in the decision function.
		- **C**: Penalty parameter C of the error term.
		- **max_iter**: Hard limit on iterations for coordinate descent.
		- **eps**: Tolerance for stopping criterion based on the relative l1 norm for difference of beta and beta_old.
		- **print_step**: If print the interations for coordinate descent, 1 indicates YES, 0 indicates NO.
	- Methods:
		- **decision_function(X)**: Evaluates the decision function for the samples in X.
			- X : array-like, shape (n_samples, n_features)
		- **fit(X, y, sample_weight=1.)**: Fit the SVM model.
			- X : {array-like, sparse matrix}, shape (n_samples, n_features)
			- y : array-like, shape (n_samples,) **NOTE: y must be +1 or -1!**
			- sample_weight : array-like, shape (n_samples,), weight for each sample.



PermT
~~~~~~~~~~~~



Example
~~~~~~~~~~~~~~~~~
