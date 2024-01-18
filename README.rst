.. dnn-inference documentation master file

🔬 dnn-inf: significance tests of feature relevance for a black-box model 
==========================================================================

.. raw:: html

    <embed>
        <a href="https://github.com/statmlben/dnn-inference"><img src="logo/logo_c.png" align="right" height="138" /></a>
    </embed>

.. -*- mode: rst -*-

|PyPi|_ |Keras|_ |MIT|_ |Python3|_ |tensorflow|_ |downloads|_ |downloads_month|_

.. |PyPi| image:: https://badge.fury.io/py/dnn-inference.svg
.. _PyPi: https://pypi.org/project/dnn-inference/

.. |Keras| image:: https://img.shields.io/badge/keras-tf.keras-red.svg
.. _Keras: https://keras.io/

.. |MIT| image:: https://img.shields.io/pypi/l/dnn-inference.svg
.. _MIT: https://opensource.org/licenses/MIT

.. |Python3| image:: https://img.shields.io/badge/python-3-green.svg
.. _Python3: www.python.org

.. |tensorflow| image:: https://img.shields.io/badge/keras-tensorflow-blue.svg
.. _tensorflow: https://www.tensorflow.org/

.. |downloads| image:: https://pepy.tech/badge/dnn-inference
.. _downloads: https://pepy.tech/project/dnn-inference

.. |downloads_month| image:: https://pepy.tech/badge/dnn-inference/month
.. _downloads_month: https://pepy.tech/project/dnn-inference

.. image:: ./logo/logo_header.png
   :width: 900

**dnn-inf** is a Python module for hypothesis testing based on black-box models, including **deep neural networks**. 

- GitHub repo: `https://github.com/statmlben/dnn-inference <https://github.com/statmlben/dnn-inference>`_
- Documentation: `https://dnn-inference.readthedocs.io <https://dnn-inference.readthedocs.io/en/latest/>`_
- PyPi: `https://pypi.org/project/dnn-inference <https://pypi.org/project/nonlinear-causal>`_
- Open Source: `MIT license <https://opensource.org/licenses/MIT>`_
- Paper: `arXiv:2103.04985 <https://arxiv.org/abs/2103.04985>`_

🎯 What We Can Do
-----------------

**dnn-inference** is able to provide an asymptotically valid `p-value` to examine if $S$ is discriminative features to predict $Y$.
Specifically, the proposed testing is:
$$H_0: R(f^*) - R_{S}(g^*) = 0,   H_a: R(f^*) - R_{S}(g^*) < 0,$$
where $X_S$ is a collection of hypothesized features, $R$ and $R_S$ are risk functions with/without the hypothesized features $X_S$, 
and $f^*$ and $g^*$ are population minimizers on $R$ and $R_S$ respectively. 
The proposed test just considers the difference between the best predictive scores with/without hypothesized features. 
Please check more details in our paper `arXiv:2103.04985 <https://arxiv.org/abs/2103.04985>`_.

- When `log-likelihood` is used as a loss function, then the test is equivalent to a conditional independence test: `$Y indep X_{S} | X_{S^c}$`. 
- Only `a small number of fitting` on neural networks is required, and the number can be as small as 1.
- Asymptotically Type I error control and power consistency.


Installation
============

Dependencies
------------

``dnn-inference`` requires: **Python>=3.8** + `requirements.txt <./requirements.txt>`_

.. code:: bash

  pip install -r requirements.txt

User installation
-----------------

Install ``dnn-inference`` using ``pip``

.. code:: bash

	pip install dnn_inference
	pip install git+https://github.com/statmlben/dnn-inference.git

Reference
---------
**If you use this code please star the repository and cite the following paper:**

.. code:: bib

   @article{dai2022significance,
      title={Significance Tests of Feature Relevance for a Black-Box Learner},
      author={Dai, Ben and Shen, Xiaotong and Pan, Wei},
      journal={IEEE Transactions on Neural Networks and Learning Systems},
      year={2022},
      publisher={IEEE}
   }


Notebook
========

- **MNIST dataset**: `Notebook1 <https://dnn-inference.readthedocs.io/en/latest/nb/MNIST_demo.html>`_

- **Boston house prices dataset**: `Notebook2 <https://dnn-inference.readthedocs.io/en/latest/nb/Boston_house_prices.html>`_