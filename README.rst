.. -*- mode: rst -*-

|PyPi|_ |Keras|_ |MIT| |Python3| |tensorflow|_ |downloads|_ |downloads_month|_

.. |dAI| image:: https://img.shields.io/badge/Powered%20by-cuhk%40dAI-purple.svg
.. _dAI: https://www.bendai.org

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

.. image:: ./logo/demo_result.png
   :align: center
   :width: 800

Three-lines-of-code
-------------------
.. figure:: ./logo/dnn_inf.gif


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
- Tensorflow>=2.0
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

