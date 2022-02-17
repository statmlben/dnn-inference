"""
Significance testing based on black-box models
"""

# Author: Ben Dai <bendai@cuhk.edu.hk>

from tensorflow.keras.models import load_model
import numpy as np
from scipy.stats import norm, mode
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import glorot_uniform
import tensorflow as tf
from sklearn.model_selection import KFold
from scipy.stats import hmean, gmean
import scipy.optimize
import matplotlib.pyplot as plt
import os
from scipy.optimize import brentq
from copy import deepcopy
import time


def comb_p_value(P_value, cp):
    """
    combining p-values
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


def size_fun(x, N, min_N=2000):
    return x + min_N * np.log(x) / 2 / np.log(min_N/2) - N

class split_test(object):
    """
    Class for one-split/two-split test based on deep neural networks.

    Parameters
    ----------

    inf_feats: list-like | shape = (num of tests, dim of features)
     List of covariates/Features under hypothesis testings, one element corresponding to a hypothesis testing.

    model: {keras-defined neural network}
     A neural network for original full dataset

    model_mask: {keras-defined neural network}
     A neural network for masked dataset by masking/changing the features under hypothesis testing

    change: {'mask', 'perm'}, default='mask'
     The way to change the testing features, ``'mask'`` replaces testing features as zeros, while ``'perm'`` permutes features via instances.

    alpha: float (0,1), default=0.05
     The nominal level of the hypothesis testing

    verbose: {0, 1}, default=0
     If print the testing results, 1 indicates YES, 0 indicates NO.

    eva_metric: {'mse', 'zero-one', 'cross-entropy', or custom metric function}
     The evaluation metric, ``'mse'`` is the l2-loss for regression, ``'zero-one'`` is the zero-one loss for classification, ``'cross-entropy'`` is log-loss for classification. It can also be custom metric function as ``eva_metric(y_true, y_pred)``.

    cp_path: string, default='./splitTest_checkpoints'
     The checkpoints path to save the models

    Methods
    -------
    adaRatio
    dual_feat
    testing
    visual
    """

    def __init__(self, inf_feats, model, model_mask, change='mask', alpha=.05, verbose=0, eva_metric='mse', cp_path='./splitTest_checkpoints'):
        self.inf_feats = inf_feats
        self.model = model
        self.model_mask = model_mask
        self.alpha = alpha
        self.change = change
        self.eva_metric = eva_metric
        self.p_values = []
        self.p_values_comb = []
        self.cp_path = cp_path

    def metric(self, y_true, y_pred):
        """
        Return the loss for `self.eva_metric`

        Parameters
        ----------
        y_true: the ground truth

        y_pred: the predicted label

        """

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
            metric_tmp = -np.log(y_pred[range(len(y_pred)),label_true])
        else:
            metric_tmp = self.eva_metric(y_true, y_pred)
        return metric_tmp

    def save_init(self):
        """
        Save the initialization for full and mask network models under class Dnn
        """
        self.model.built = True
        self.model_mask.built = True

        # self.model.save_weights(self.cp_path+'/model_init.h5')
        # self.model_mask.save_weights(self.cp_path+'/model_mask_init.h5')
        self.model.save(self.cp_path+'/model_init')
        self.model_mask.save(self.cp_path+'/model_mask_init')

    def reset_model(self):
        """
        Reset the full and mask network models under class Dnn
        """
        self.model.built = True
        self.model_mask.built = True

        # self.model.load_weights(self.cp_path+'/model_init.h5')
        # self.model_mask.load_weights(self.cp_path+'/model_mask_init.h5')
        self.model = load_model(self.cp_path+'/model_init')
        self.model_mask = load_model(self.cp_path+'/model_mask_init')

    def reload_model(self, path_full, path_mask):
        """
        reload the pre-saved model.
        """
        # path_tmp = self.cp_path+'/model'+'_inf'+str(k)+'_cv'+str(h)+'.h5'
        # mask_path_tmp = self.cp_path+'/model_mask'+'_inf'+str(k)+'_cv'+str(h)+'.h5'
        self.model.save_weights(path_full)
        self.model_mask.save_weights(path_mask)
        self.model.load_weights(path_full)
        self.model_mask.load_weights(path_mask)

    ## can be extent to @abstractmethod
    def dual_feat(self, X, cat_feats=[], k=0):
        """
        Return instances with masked/perm k-th hypothesized features (dual feats).

        Parameters
        ----------
        X: array-like
         Target instances.

        cat_feats: list-like, default = []
         The col-index for categorical features

        k: integer, default = 0
         k-th hypothesized features in inf_feats
        """

        if self.change == 'mask':
            Z = self.mask_cov(X, cat_feats=cat_feats, k=k)
        elif self.change == 'perm':
            Z = self.perm_cov(X, k=k)
        return Z


    def mask_cov(self, X, cat_feats=[], k=0):
        """
        Return instances with masked k-th hypothesized features.

        Parameters
        ----------
        X: array-like
         Target instances.

        cat_feats: list-like, default = []
         The col-index for categorical features; now it's only work for tabular data

        k: integer, default = 0
         k-th hypothesized features in inf_feats
        """
        Z = X.copy()
        n_sample = len(Z)

        if len(self.inf_feats[k].shape) > 1:
            ## for channels_last image data: shape should be (#samples, img_rows, img_cols, channel)
            if len(cat_feats):
                warnings.warn("cat_feats is ignored. cat_feats only works for tabular data, whereas a image dataset is given.")
            Z[:, self.inf_feats[k][0][:,None], self.inf_feats[k][1], :] = 0.
        else:
            ## this for tabular data
            cat_inf_feats = set(self.inf_feats[k]).intersection(set(cat_feats))
            cont_inf_feats = set(self.inf_feats[k]) - set(cat_inf_feats)
            cat_inf_feats, cont_inf_feats = list(cat_inf_feats), list(cont_inf_feats)
            
            Z[:,cont_inf_feats] = np.array([np.mean(Z[:,cont_inf_feats], axis=0)]*n_sample)
            if len(cat_inf_feats) > 0:
                Z[:,cat_inf_feats] = np.array([mode(Z[:,cat_inf_feats], axis=0)[0][0]]*n_sample)
        return Z

    def perm_cov(self, X, k=0):
        """
        Return instances with permuted k-th hypothesized features.

        Parameters
        ----------
        X: array-like
         Target instances.

        k: integer, default = 0
         k-th hypothesized features in inf_feats
        """
        Z = X.copy()
        if len(self.inf_feats[k].shape) > 1:
            ## for channels_last image data: shape should be (#samples, img_rows, img_cols, channel)
            Z[:,self.inf_feats[k][0][:,None], self.inf_feats[k][1], :]= np.random.permutation(Z[:,self.inf_feats[k][0][:,None], self.inf_feats[k][1], :])
        else:
            Z[:,self.inf_feats[k]]= np.random.permutation(Z[:,self.inf_feats[k]])
        return Z

    def noise_cov(self, X, k=0):
        """
        Return instances with niosed k-th hypothesized features.

        Parameters
        ----------
        X: array-like
         Target instances.

        k: integer, default = 0
         k-th hypothesized features in inf_feats
        """
        Z = X.copy()
        Z[:,self.inf_feats[k]] = np.random.randn(len(X), len(self.inf_feats[k]))
        return Z

    def adaRatio(self, X, y, k=0, cat_feats=[], fit_params={}, split_params={}):
        """
        Return a data-adaptive splitting ratio and perturbation level.

        Parameters
        ----------
        X: array-like | shape=(n_samples, dim1, dim2, ...)
            Features.

        y: array-like | shape=(n_samples, dim)
            Outcomes.

        k: integer, default = 0
            k-th hypothesized features in inf_feats

        fit_params: dict | shape = dict of fitting parameters
            See keras ``fit``: (https://keras.rstudio.com/reference/fit.html), including ``batch_size``, ``epoch``, ``callbacks``, ``validation_split``, ``validation_data``.

        split_params: {dict of splitting parameters}

            split: {'one-split', 'two-split'}, default='one-split'
                one-split or two-split tests.

            perturb: float, default=None
                Perturb level for the one-split test, if ``perturb = None``, then the perturb level is determined by adaptive tunning.

            num_perm: int, default=100
                Number of permutation for determine the splitting ratio.

            ratio_grid: list of float (0,1), default=[.2, .4, .6, .8]**
                A list of estimation/inference ratios under searching.

            if_reverse: {0,1}, default=0
                ``if_reverse = 0`` indicates the loop of ``ratio_grid`` starts from smallest one to largest one; ``if_reverse = 1`` indicates the loop of ``ratio_grid`` starts from largest one to smallest one.

            perturb_scale: integer, default=5
                The scale of perturb, and the perturbation grid is generated based on 2**range(-perturb_scale, perturb_scale)*var(losses by full model)

            min_inf: int, default=0
                The minimal size for inference sample.

            min_est: int, default=0
                The minimal size for estimation sample.

            ratio_method: {'fuse', 'close'}, default='fuse'
                The adaptive splitting method to determine the optimal estimation/inference ratios.

            cv_num: int, default=1
                The number of cross-validation to shuffle the estimation/inference samples in adaptive ratio splitting. We recommend to set **cv_num** in **adaRatio** as same as **cv_num** in **testing**.

            cp: {'gmean', 'min', 'hmean', 'Q1', 'hommel', 'cauchy'}, default ='hommel'
                A method to combine p-values obtained from cross-validation. see (https://arxiv.org/pdf/1212.4966.pdf) for more detail.

            verbose: {0,1}, default=1
                if print the results the adaptive tuning procedure.

        cp: {'gmean', 'min', 'hmean', 'Q1', 'hommel', 'cauchy'} | default = 'hommel'
            A method to combine p-values obtained from cross-validation. see (https://arxiv.org/pdf/1212.4966.pdf) for more detail.

        verbose: {0,1} | default=1
            If print the adaptive splitting process.

        Returns
        -------
        n_opt: integer
            A reasonable estimation sample size.

        m_opt: integer
            A reasonable inference sample size.

        perturb_opt: float
            A reasonable perturbation level.

        """

        split_params_default = {'split': 'one-split',
                        'perturb': None,
                        'num_perm': 100,
                        'ratio_grid': [.2, .4, .6, .8],
                        'if_reverse': 0,
                        'perturb_scale': 5,
                        'min_inf': 0,
                        'min_est': 0,
                        'ratio_method': 'fuse',
                        'cv_num': 1,
                        'cp': 'hommel',
                        'verbose': 1}
        
        split_params_default.update(split_params)
        split_params = split_params_default

        perturb=split_params['perturb']
        split=split_params['split']
        perturb_scale=split_params['perturb_scale']
        ratio_grid=split_params['ratio_grid']
        if_reverse=split_params['if_reverse']
        min_inf=split_params['min_inf']
        min_est=split_params['min_est']
        ratio_method=split_params['ratio_method']
        num_perm=split_params['num_perm']
        cv_num= split_params['cv_num']
        cp = split_params['cp']
        verbose= split_params['verbose']

        ratio_grid.sort()
        if if_reverse == 1:
            ratio_grid = list(reversed(ratio_grid))

        candidate, Err1_lst, ratio_lst, P_value_lst = [], [], [], []
        found = 0
        if split == 'two-split':
            for ratio_tmp in ratio_grid:
                ratio_tmp = ratio_tmp/2
                m_tmp = int(len(X)*ratio_tmp)
                n_tmp = len(X) - 2*m_tmp
                if (m_tmp < min_inf) or (n_tmp < min_est):
                    continue
                
                # split data
                P_value = []
                for h in range(cv_num):
                    self.reset_model()
                    P_value_cv = []
                    ## generate permutated samples
                    X_perm = X.copy()
                    X_perm = self.perm_cov(X_perm, k)
                    ## split sample
                    X_train, X_test, y_train, y_test = train_test_split(X_perm, y, train_size=n_tmp, random_state=1)
                    # training for full model
                    history = self.model.fit(x=X_train, y=y_train, **fit_params)

                    # training for mask model
                    Z_train = self.dual_feat(X_train, cat_feats, k)
                    history_mask = self.model_mask.fit(x=Z_train, y=y_train, **fit_params)
                    Z_test = self.dual_feat(X_test, cat_feats, k)

                    # evaluation for mask model
                    pred_y_mask = self.model_mask.predict(Z_test)
                    for j in range(num_perm):
                        X_test_perm = X_test.copy()
                        X_test_perm = self.perm_cov(X_test_perm, k)
                        pred_y = self.model.predict(X_test_perm)
                        ind_inf, ind_inf_mask = train_test_split(range(len(pred_y)), train_size=m_tmp, random_state=42)
                        metric_tmp = self.metric(y_test[ind_inf], pred_y[ind_inf])
                        metric_mask_tmp = self.metric(y_test[ind_inf_mask], pred_y_mask[ind_inf_mask])
                        p_value_tmp = self.diff_p_value(metric_tmp, metric_mask_tmp)
                        P_value_cv.append(p_value_tmp)
                    P_value.append(P_value_cv)
                P_value = np.array(P_value)
                P_value_cp = np.array([comb_p_value(P_value[:,i], cp=cp) for i in range(num_perm)])
                ## compute the type 1 error
                Err1 = len(P_value_cp[P_value_cp < self.alpha]) / len(P_value_cp)
                Err1_lst.append(Err1)
                # P_value_lst.append(P_value)
                ratio_lst.append(ratio_tmp)

                if verbose==1:
                    print('(AdaRatio) Est. Type 1 error: %.3f; p_value_mean: %.3f, inf sample ratio: %.3f' 
                        %(Err1, P_value_cp.mean(), ratio_tmp))
                    # print('(AdaRatio) p_value: %.3f, inference sample ratio: %.3f' %(P_value.mean(), ratio_tmp))

                if Err1 < self.alpha:
                    found = 1
                    if ratio_method == 'fuse':
                        m_opt = m_tmp
                        n_opt = len(X) - 2*m_opt
                        break

            if found==0:
                warnings.warn("No ratio can control the Type 1 error, pls increase the sample size, and the inference sample ratio is set as the min of ratio_grid.")
                Err1_lst, ratio_lst = np.array(Err1_lst), np.array(ratio_lst)
                m_opt = int(ratio_lst[np.argmin(Err1_lst)] * len(X))
                n_opt = len(X) - 2*m_opt

            return n_opt, m_opt

        if split == 'one-split':
            if perturb != None:
                perturb_grid = [perturb]

            for perturb_idx_tmp in range(-perturb_scale, perturb_scale):
                perturb_level_tmp = 2**(perturb_idx_tmp)
                ## stop if current perturb is enough to control the type 1 error
                if found == 1:
                    break
                Err1_lst, ratio_lst, perturb_lst, P_value_lst = [], [], [], []
                for ratio_tmp in ratio_grid:
                    m_tmp = int(len(X)*ratio_tmp)
                    n_tmp = len(X) - m_tmp

                    if (m_tmp < min_inf) or (n_tmp < min_est):
                        continue
                    # split data
                    P_value = []
                    for h in range(cv_num):
                        self.reset_model()
                        P_value_cv = []
                        ## generate permutated samples
                        X_perm = X.copy()
                        X_perm = self.perm_cov(X_perm, k)
                        # split samples
                        X_train, X_test, y_train, y_test = train_test_split(X_perm, y, train_size=n_tmp, random_state=h)
                        # training for full model
                        history = self.model.fit(x=X_train, y=y_train, **fit_params)
                        
                        # training for mask model
                        Z_train = self.dual_feat(X_train, cat_feats, k)
                        history_mask = self.model_mask.fit(x=Z_train, y=y_train, **fit_params)
                        
                        ## evaluation for the mask model
                        Z_test = self.dual_feat(X_test, cat_feats, k)
                        pred_y_mask = self.model_mask.predict(Z_test)
                        metric_mask_tmp = self.metric(y_test, pred_y_mask)

                        # evaluation
                        for j in range(num_perm):
                            X_test_perm = X_test.copy()
                            X_test_perm = self.perm_cov(X_test_perm, k)
                            ## compute the metric based on full model
                            pred_y = self.model.predict(X_test_perm)
                            metric_tmp = self.metric(y_test, pred_y)
                            ## compute the p-value based on the diff metrics
                            p_value_tmp = self.diff_p_value(metric_tmp, metric_mask_tmp, 
                                                    perturb_level=perturb_level_tmp)
                            P_value_cv.append(p_value_tmp)
                        P_value.append(P_value_cv)

                    P_value = np.array(P_value)
                    
                    P_value_cp = np.array([comb_p_value(P_value[:,i], cp=cp) for i in range(num_perm)])
                    Err1 = len(P_value_cp[P_value_cp<=self.alpha])/len(P_value_cp)
                    Err1_lst.append(Err1)

                    if verbose==1:
                        print('(AdaRatio) Est. Type 1 err: %.3f; p_value_mean: %.3f, inf sample ratio: %.3f, perturb_level: %s' 
                            %(Err1, P_value_cp.mean(), ratio_tmp, perturb_level_tmp))

                    P_value_lst.append(P_value_cp)
                    ratio_lst.append(ratio_tmp)
                    perturb_lst.append(perturb_idx_tmp)

                    if Err1 < self.alpha:
                        found = 1
                        m_opt = m_tmp
                        n_opt = len(X) - m_opt
                        perturb_idx_opt = perturb_idx_tmp
                        break
            if found==0:
                warnings.warn("No ratio and perturb_level can control the Type 1 error," \
                    "pls increase the perturb_level and sample size, and inference sample ratio is set as the one minimize the permutation Type 1 Error.")
                Err1_lst, ratio_lst = np.array(Err1_lst), np.array(ratio_lst)
                m_opt = int(ratio_lst[np.argmin(Err1_lst)] * len(X))
                n_opt = len(X) - m_opt
                perturb_idx_opt = perturb_lst[np.argmin(Err1_lst)]

            return n_opt, m_opt, perturb_idx_opt


    def diff_p_value(self, metric_full, metric_mask, perturb_level=0.):
        """
        Return p-value for diff btw `metric_full` and `metric_mask` for the one-/two-split test.

        Parameters
        ----------
        metric_full: {array-like} of shape (n_samples)
            metric for samples based on full model

        metric_mask: {array-like} of shape (n_samples)
            metric for samples based on mask model

        Returns
        -------

        P_value: array of float [0, 1]
            The p_values for target one-/two-split test.
        """
        perturb_base = metric_full.std()
        diff_tmp = metric_full - metric_mask + perturb_level * perturb_base * np.random.randn(len(metric_full))
        Lambda_tmp = np.sqrt(len(diff_tmp)) * ( diff_tmp.std() )**(-1)*( diff_tmp.mean() )
        p_value_tmp = norm.cdf(Lambda_tmp)
        return p_value_tmp

    def ada_testing(self, X, Z, y, ratio, fit_params, split_params={}, cat_feats=[], cv_num=5, cp='hommel'):
        """
        Return p-values for hypothesis testing for inf_feats in class split_test.
        Parameters
        ----------
        X: {array-like} of shape (n_samples, dim_features)
            Instances matrix/tensor, where n_samples in the number of samples and dim_features is the dimension of the features.
             If X is vectorized feature, ``shape`` should be ``(#Samples, dim of feaures)``
             If X is image/matrix data, ``shape`` should be ``(#samples, img_rows, img_cols, channel)``, that is, *X must channel_last image data*.   

        Z: {array-like} of shape (n_samples, dim_features)
            X with masked/perm k-th hypothesized features (dual feats).

        y: {array-like} of shape (n_samples, )
             Output vector/matrix relative to X.

        ratio: float, default = None**
            A pre-specific inference sample ratio, if ``est_size = None``, then it is determined by adaptive splitting method ``metric``.

        fit_params: {dict of fitting parameters}
            See keras ``fit``: (https://keras.rstudio.com/reference/fit.html), including ``batch_size``, ``epoch``, ``callbacks``, ``validation_split``, ``validation_data``, and so on.

        split_params: {dict of splitting parameters}

            split: {'one-split', 'two-split'}, default = 'one-split'
                one-split or two-split test statistic.

            perturb: float, default = None
                Perturb level for the one-split test, if ``perturb = None``, then the perturb level is determined by adaptive tunning.

            num_perm: int, default = 100
                Number of permutation for determine the splitting ratio.

            ratio_grid: list of float (0,1), default = [.2, .4, .6, .8]
                A list of estimation/inference ratios under searching.

            if_reverse: {0,1}, default = 0
                ``if_reverse = 0`` indicates the loop of ``ratio_grid`` starts from smallest one to largest one; ``if_reverse = 1`` indicates the loop of ``ratio_grid`` starts from largest one to smallest one.

            perturb_scale: integer, default = 5
                The scale of perturb, and the perturbation grid is generated based on 2**range(-perturb_scale, perturb_scale)*var(losses by full model)

            min_inf: int, default = 0
                The minimal size for inference sample.

            min_est: int, default = 0
                The minimal size for estimation sample.

            ratio_method: {'fuse', 'close'}, default = 'fuse'
                The adaptive splitting method to determine the optimal estimation/inference ratios.

            cv_num: int, default = *cv_num*
                The number of cross-validation to shuffle the estimation/inference samples in adaptive ratio splitting.

            cp: {'gmean', 'min', 'hmean', 'Q1', 'hommel', 'cauchy'}, default = *cp*
                A method to combine p-values obtained from cross-validation. see (https://arxiv.org/pdf/1212.4966.pdf) for more detail.
                
            verbose: {0,1}, default = 1
                if print the results the adaptive tuning procedure.

        cat_feats: list, default = []
            The col-index for categorical features; **now it's only work for tabular data**

        cv_num: int, default = 5
            The number of cross-validation to shuffle the estimation/inference samples in testing.

        cp: {'gmean', 'min', 'hmean', 'Q1', 'hommel', 'cauchy'}, default ='hommel'
            A method to combine p-values obtained from cross-validation.
        
        Returns
        -------
        P_value: array of float [0, 1]
            The p_values for target hypothesis testings.
        """
        split_params_default = {'split': 'one-split',                      
                                'num_perm': 100,
                                'perturb_scale': 5,
                                'ratio_grid': [.2, .4, .6, .8],
                                'if_reverse': 0,
                                'min_inf': 0,
                                'min_est': 0,
                                'ratio_method': 'fuse',
                                'cv_num': cv_num,
                                'cp': cp,
                                'verbose': 1}

        if not os.path.exists(self.cp_path):
            os.mkdir(self.cp_path)
  
        self.save_init()
        init_lr_full = deepcopy(self.model.optimizer.lr.numpy())
        init_lr_mask = deepcopy(self.model.optimizer.lr.numpy())

        P_value = []

        for k in range(len(self.inf_feats)):
            self.reset_model()
            self.model.optimizer.lr.assign(init_lr_full)
            self.model_mask.optimizer.lr.assign(init_lr_mask)

            # split inference sample and estimation sample
            m, n = int(ratio * len(X)), len(X) - int(ratio * len(X))
            perturb_idx = split_params['perturb']
            perturb_level = 2**perturb_idx
            print('%d-th inference; log ratio data splitting: n: %d; m: %d; perturb: %s' %(k, n, m, perturb_level))

            P_value_cv = []
            
            # split training and testing sets
            for h in range(cv_num):
                X_train, X_test, Z_train, Z_test, y_train, y_test = train_test_split(X, Z, y, train_size=n, random_state=h)
                if split_params['split'] == 'two-split':
                    X_inf, X_inf_mask, Z_inf, Z_inf_mask, y_inf, y_inf_mask = train_test_split(X_test, Z_test, y_test, train_size=.5, random_state=42)
                if split_params['split'] == 'one-split':
                    X_inf, X_inf_mask, Z_inf, Z_inf_mask, y_inf, y_inf_mask = X_test.copy(), X_test.copy(), Z_test.copy(), Z_test.copy(), y_test.copy(), y_test.copy()
            
            # reset the model
            self.reset_model()
            self.model.optimizer.lr.assign(init_lr_full)
            self.model_mask.optimizer.lr.assign(init_lr_mask)
            
            # test for X
            history = self.model.fit(X_train, y_train, **fit_params)
            pred_y = self.model.predict(X_inf)
            metric_full = self.metric(y_inf, pred_y)
            
            # test for Z
            history_mask = self.model_mask.fit(Z_train, y_train, **fit_params)
            pred_y_mask = self.model_mask.predict(Z_inf)
            metric_mask = self.metric(y_inf_mask, pred_y_mask)
            
            #calculate the p_value
            p_value_tmp = self.diff_p_value(metric_full, metric_mask, perturb_level)
            print('cv: %d; p_value: %.5f; metric: %.5f(%.5f); metric_mask: %.5f(%.5f)' %(h, p_value_tmp, metric_full.mean(), metric_full.std(), metric_mask.mean(), metric_mask.std()))

            P_value_cv.append(p_value_tmp)
        
        # calculate the mean of the p_value
        self.p_values_comb.append(P_value_cv) 
        P_value_cv = np.array(P_value_cv)
        p_value_mean = comb_p_value(P_value_cv, cp=cp)

        print('#'*50)

        if p_value_mean < self.alpha:
            print('%d-th inf: reject H0 with p_value: %.3f' %(k, p_value_mean))
        else:
            print('%d-th inf: accept H0 with p_value: %.3f' %(k, p_value_mean))

        print('#'*50)

        P_value.append(p_value_mean)
   
        self.p_values = P_value
        return P_value   

    def visual(self, X, y, plt_params={'cmap': 'RdBu', 'alpha':0.6}, plt_mask_params={'cmap': 'RdBu', 'alpha':0.6}):
        """
        Visualization for the inference results based on one illustrative example

        Parameters
        ----------
        X: array-like
         demo instances.

        y: array-like
         demo labels

        plt_params: dict
         dictory of parameters for the imshow for original image see: (https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.imshow.html)

        plt_mask_params: dict
         dictory of parameters for the imshow for mask see: (https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.imshow.html)

        """
        if len(X.shape) == 2:
            print('sorry, visual function only work for image data.')
        else:
            num_class = y.shape[1]
            demo_ind = np.array([np.where(y[:,k]==1)[0][0] for k in range(num_class)])
            X_demo = X[demo_ind]

            cols, rows = len(self.inf_feats), num_class
            fig = plt.figure(constrained_layout=False)
            spec = fig.add_gridspec(ncols=cols, nrows=rows)
            for row in range(rows):
                for col in range(cols):
                    X_mask_tmp = np.nan*np.ones(X_demo.shape)
                    X_mask_tmp = self.mask_cov(X_mask_tmp, k=col)[0]
                    ax = fig.add_subplot(spec[row, col])
                    im1 = ax.imshow(X_demo[row], vmin=0, vmax=1, **plt_params)
                    ax.axis('off')
                    im2 = ax.imshow(X_mask_tmp, vmin=0, vmax=1, **plt_mask_params)
                    ax.axis('off')
                    if row == 0:
                        ax.set_title('p_values: %.3f' %self.p_values[col])
            plt.subplots_adjust(top = 0.95, bottom=0.05, hspace=0.03, wspace=0.03, right=0.95)
            plt.show()
