# base.py
# Author: Ben Dai <bendai@cuhk.edu.hk>
# MIT License

import numpy as np
from scipy.stats import norm, mode
from sklearn.model_selection import train_test_split
import warnings
import tensorflow as tf
from scipy.stats import hmean, gmean
import scipy.optimize
import os
from scipy.optimize import brentq

def size_fun(x, N, min_N=2000):
    return x + min_N * np.log(x) / 2 / np.log(min_N/2) - N

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def comb_p_value(P_value, cp='hommel'):
    """
    combining p-values
    
    Parameters
    ----------
    P_value: list-like
    
    cp: {'gmean', 'min', 'hmean', 'Q1', 'hommel', 'cauchy'}, default ='hommel'
     A method to combine p-values obtained from cross-validation. see (https://arxiv.org/pdf/1212.4966.pdf) for more detail. 
    """
    P_value = np.array(P_value)
    P_value
    cv_num = len(P_value)
    # print(P_value)
    if cv_num > 1:
        P_value = np.array(P_value)
        if cp == 'gmean':
            P_value_cp = np.e*gmean(P_value, 0)
        elif cp == 'median':
            P_value_cp = 2*np.median(P_value, 0)
        elif cp == 'Q1':
            P_value_cp = cv_num/2.*np.partition(P_value, 1)[1]
        elif cp == 'min':
            P_value_cp = cv_num*np.min(P_value, 0)
        elif cp == 'hmean':
            P_value_cp = np.e * np.log(cv_num) * hmean(P_value, 0)
        elif cp == 'hommel':
            const = np.sum(1. / (np.arange(cv_num) + 1.))
            order_const = const*(cv_num/(np.arange(cv_num) + 1.))
            P_value_cp = np.sort(P_value)*order_const
            P_value_cp = np.min(P_value_cp)
        elif cp == 'cauchy':
            t0 = np.mean(np.tan((.5 - P_value)*np.pi))
            P_value_cp = .5 - np.arctan(t0)/np.pi
        else:
            warnings.warn("cp should be {geometric, min, median, Q1, hmean, hommel, cauchy}")
    else:
        P_value_cp = np.mean(P_value, 0)
    P_value_cp = np.minimum(P_value_cp, 1.)
    return P_value_cp
