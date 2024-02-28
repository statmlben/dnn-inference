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
import time, datetime
from .base import comb_p_value, size_fun
import emoji 

class split_test(object):
    """
    Class for one-split/two-split test based on deep neural networks.

    Parameters
    ----------

    inf_feats: list-like | shape = (num of tests, dim of hypothesized features)
     List of covariates/features under hypothesis testings, `inf_feats[k]` corresponding to the k-th hypothesis testing. 
     `len(inf_feats[k]) == 2` indicates hypothesis for image data; where `inf_feats[k][0]` is the hypothesized rows;
     and `inf_feats[k][1]` is the hypothesized columns.
     `len(inf_feats[k]) == 1` indicates hypothesis for tabular data; where `inf_feats[k]` is the hypothesized features.

    model: {keras-defined neural network}
     A neural network for original full dataset

    model_alter: {keras-defined neural network}
     A neural network for masked dataset by masking/changing the features under hypothesis testing

    change: {'mask', 'perm'}, default='perm'
     The way to change the testing features, ``'mask'`` replaces testing features as zeros, while ``'perm'`` permutes features via instances.

    alpha: float (0,1), default=0.05
     The nominal level of the hypothesis testing

    verbose: {0, 1}, default=0
     If print the testing results, 1 indicates YES, 0 indicates NO.

    eva_metric: {'zero-one' or custom loss function in `tf.keras.losses`}
     The evaluation metric, ``'zero-one'`` is the zero-one loss for classification, 
     It can also be custom loss function as ``eva_metric(y_true, y_pred)`` see `tf.keras.losses`.
     Please specify `from_logits=False`, `reduction='none'`

    cp_path: string, default='./saved'
     The checkpoints path to save the models

    Methods
    -------
    tuneHP
    alter_feat
    testing
    visual
    """

    def __init__(self, inf_feats, model_null, model_alter, alpha=.05, verbose=0, eva_metric='zero-one', cp_path='./saved'):
        self.name = 'split_test'
        self.inf_feats = inf_feats
        self.model_null = model_null
        self.model_alter = model_alter
        self.alpha = alpha
        self.change = 'perm'
        self.eva_metric = eva_metric
        self.p_values = []
        self.p_values_comb = []
        self.cp_path = cp_path
        self.test_params = {}
        self.tune_params = {}

    def update_cp_dir(self):
        """
        Update checkpoint dir by using datetime
        """
        start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
        self.cp_path = os.path.join(self.cp_path, self.name, start_time)

    def update_test_params(self, test_params):
        """
        Update test params
        """
        test_params_default = { 'split': 'one-split',
                                'inf_ratio': None,
                                'perturb': None,
                                'cv_num': 5,
                                'cp': 'hommel',
                                # 'min_inf': 0,
                                # 'min_est': 0,
                                'verbose': 1}
        ## update testing params
        test_params_default.update(test_params)
        test_params = test_params_default

        if test_params['split'] == "two-split":
            test_params['perturb'] = 0.

        ## check everything
        # assert isinstance(test_params['min_inf'], int), "'min_inf' must be int."
        # assert isinstance(test_params['min_est'], int), "'min_est' must be int."
        assert test_params['split'] in ['one-split', 'two-split'], "'split' must be 'one-split' or 'two-split'!"
        if test_params['inf_ratio'] is not None:
            assert (test_params['inf_ratio'] > 0) and ((test_params['inf_ratio'] < 1.)), "'inf_ratio' must be in (0,1)"
        if test_params['perturb'] is not None:
            assert test_params['perturb'] >= 0., "'perturb' must be nonnegative."
        assert isinstance(test_params['cv_num'], int), "'cv_num' must be int."
        assert test_params['cp'] in ['gmean', 'min', 'hmean', 'Q1', 'hommel', 'cauchy'], "combing method `cp` must in ['gmean', 'min', 'hmean', 'Q1', 'hommel', 'cauchy']."

        self.test_params = test_params
        return test_params

    def update_tune_params(self, tune_params):
        """
        Update tune params
        """
        tune_params_default = { 'num_perm': 100,
                                'ratio_grid': [.2, .4, .6, .8],
                                'if_reverse': 0,
                                'perturb_range': 2.**np.arange(-3,3,.1),
                                'tune_ratio_method': 'fuse',
                                'tune_pb_method': 'fuse',
                                'cv_num': 1,
                                'cp': 'hommel',
                                'verbose': 1}
        ## update tune params
        tune_params_default.update(tune_params)
        tune_params = tune_params_default

        tune_params['ratio_grid'].sort()
        tune_params['perturb_range'].sort()
        if tune_params['if_reverse'] == 1:
            tune_params['ratio_grid'] = list(reversed(tune_params['ratio_grid']))
        
        ## check everything
        assert isinstance(tune_params['num_perm'], int), "'num_perm' must be int."
        assert (min(tune_params['ratio_grid']) > 0.0) and (max(tune_params['ratio_grid']) < 1.0), "'ratio_grid' in `tune_params` must in (0,1)."
        assert tune_params['if_reverse'] in [0, 1], "`if_reverse` in `tune_params` must be 0 or 1."
        assert (min(tune_params['perturb_range']) > 0.0), "'perturb_range' in `tune_params` must in nonnegative."
        assert tune_params['tune_ratio_method'] in ['log-ratio', 'fuse'], "`tune_ratio_method` in `tune_params` must be `log-ratio` or `fuse`."
        assert tune_params['tune_pb_method'] in ['fuse'], "`tune_pb_method` in `tune_params` must be `fuse`."
        assert tune_params['cp'] in ['gmean', 'min', 'hmean', 'Q1', 'hommel', 'cauchy'], "combing method `cp` in `tune_params` must in ['gmean', 'min', 'hmean', 'Q1', 'hommel', 'cauchy']."
        assert isinstance(tune_params['cv_num'], int), "'cv_num' must be int."
        assert isinstance(tune_params['verbose'], int), "'verbose' must be int."

        self.tune_params = tune_params
        return tune_params

    def metric(self, y_true, y_pred):
        """
        Return the loss for `self.eva_metric`.
        Use a `loss` class in ``https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss``
        Please specify `from_logits=False`, `reduction='none'`

        Parameters
        ----------
        y_true: the ground truth

        y_pred: the predicted label

        """
        if self.eva_metric == 'mse':
            metric_tmp = np.array((y_true - y_pred)**2).flatten()
        elif self.eva_metric == 'mae':
            metric_tmp = np.array(abs(y_true - y_pred)).flatten()
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
        if os.path.exists(self.cp_path+'/model_null_init'):
            pass
        else:
            self.model_null.built = True
            self.model_alter.built = True
            # self.model.save_weights(self.cp_path+'/model_null_init.h5')
            # self.model_alter.save_weights(self.cp_path+'/model_alter_init.h5')
            self.model_null.save(self.cp_path+'/model_null_init')
            self.model_alter.save(self.cp_path+'/model_alter_init')

    def reset_model(self):
        """
        Reset the full and mask network models under class Dnn
        """
        if not os.path.exists(self.cp_path+'/model_null_init'):
            self.update_cp_dir()
            self.save_init()
        self.model_null.built = True
        self.model_alter.built = True
        # self.model.load_weights(self.cp_path+'/model_null_init.h5')
        # self.model_alter.load_weights(self.cp_path+'/model_alter_init.h5')
        self.model_null = load_model(self.cp_path+'/model_null_init')
        self.model_alter = load_model(self.cp_path+'/model_alter_init')

    def reload_model(self, path_null, path_alter):
        """
        reload the pre-saved model.
        """
        # path_tmp = self.cp_path+'/model'+'_inf'+str(k)+'_cv'+str(h)+'.h5'
        # mask_path_tmp = self.cp_path+'/model_alter'+'_inf'+str(k)+'_cv'+str(h)+'.h5'
        self.model_null.save_weights(path_null)
        self.model_alter.save_weights(path_alter)
        self.model_null.load_weights(path_null)
        self.model_alter.load_weights(path_alter)

    def alter_feat(self, X, k=0, cat_feats=[]):
        """
        Return instances with masked/perm k-th hypothesized features (alter feats).

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

    def mask_cov(self, X, k=0, cat_feats=[]):
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

        if hasattr(self.inf_feats[k][0], "__len__"):
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
        if hasattr(self.inf_feats[k][0], "__len__"):
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


    def pb_ttest(self, metric_null, metric_alter, perturb_level=0.):
        """
        Perturb T-test: return p-value for diff btw `metric_null` and `metric_alter` for the one-/two-split test.

        Parameters
        ----------
        metric_null: {array-like} of shape (n_samples)
            metric for samples based on null model

        metric_alter: {array-like} of shape (n_samples)
            metric for samples based on alter model

        perturb_level: float
            The level of perturbation; the final perturbation is `perturb_level * std(metric_null)`

        Returns
        -------

        P_value: array of float [0, 1]
            The p_values for target one-/two-split test.
        """
        inf_sample = len(metric_null)
        perturb_base = metric_null.std() + 1e-5
        diff_tmp = metric_null - metric_alter + perturb_level * perturb_base * np.random.randn(inf_sample)
        Lambda_tmp = np.sqrt(inf_sample) * ( diff_tmp.std() )**(-1)*( diff_tmp.mean() )
        p_value_tmp = norm.cdf(Lambda_tmp)
        return p_value_tmp

    def perm_p_value(self, k, X_inf, X_inf_alter, y_inf, y_inf_alter, perturb_level=0., num_perm=100, return_metric=False,):
        P_value_perm, metric_null_perm, metric_alter_perm = [], [], []
        for j in range(num_perm):
            ## generate perm datasets for the null/alter models
            # X_inf_perm = X_inf.copy()
            X_inf_perm = self.perm_cov(X_inf, k)
            # X_inf_alter_perm = X_inf_alter.copy()
            X_inf_alter_perm = self.perm_cov(X_inf_alter, k)
            ## predict the outcome by null/alter models
            # pred_y = self.model_null.predict(X_inf_perm)
            # pred_y_alter = self.model_alter.predict(X_inf_alter_perm)
            ## https://github.com/tensorflow/tensorflow/issues/40261
            pred_y = self.model_null(X_inf_perm, training=False)
            pred_y_alter = self.model_alter(X_inf_alter_perm, training=False)
            ## compute the metrics
            metric_null_tmp = self.metric(y_inf, pred_y)
            metric_alter_tmp = self.metric(y_inf_alter, pred_y_alter)
            ## compute p-value
            p_value_tmp = self.pb_ttest(metric_null_tmp, metric_alter_tmp, perturb_level=perturb_level)
            P_value_perm.append(p_value_tmp)
            if return_metric:
                metric_null_perm.append(metric_null_tmp)
                metric_alter_perm.append(metric_alter_tmp)
        if return_metric:
            return P_value_perm, np.array(metric_null_perm), np.array(metric_alter_perm)
        else:
            return P_value_perm
        
    def tuneHP(self, k, X, y, fit_params, test_params, tune_params):
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

        tune_params: {dict of splitting parameters}

            perturb: float, default=None
                Perturb level for the one-split test, if ``perturb = None``, then the perturb level is determined by adaptive tunning.

            num_perm: int, default=100
                Number of permutation to estimate the Type I error.

            ratio_grid: list of float (0,1), default=[.2, .4, .6, .8]**
                A list of estimation/inference ratios under searching.

            if_reverse: {0,1}, default=0
                ``if_reverse = 0`` indicates the loop of ``ratio_grid`` starts from smallest one to largest one; ``if_reverse = 1`` indicates the loop of ``ratio_grid`` starts from largest one to smallest one.

            perturb_scale: integer, default=5
                The scale of perturb, and the perturbation grid is generated based on 2**range(-perturb_scale, perturb_scale)*var(losses by full model)

            tune_ratio_method: {'fuse', 'log-ratio'}, default='fuse'
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
        self.save_init()
        ## default testing params
        tune_params = self.update_tune_params(tune_params)
        test_params = self.update_test_params(test_params)
        find_ratio, find_pb = 0, 0

        if tune_params['tune_ratio_method'] == 'log-ratio':
            root, info = brentq(size_fun, 3., len(X), args=(len(X), 2000.), full_output=True)
            test_params['inf_ratio'] = 1. - root / len(X)

        if test_params['inf_ratio'] is None:
            err_lst, ratio_lst, p_value_lst = [], [], []
            for ratio_tmp in tune_params['ratio_grid']:
                test_params['inf_ratio'] = ratio_tmp
                ## permutate data to estimate the Type I error
                X_perm = X.copy()
                X_perm = self.perm_cov(X_perm, k)
                # split data
                P_value_cv = []
                for h in range(tune_params['cv_num']):
                    ## train models via CV
                    self.reset_model()
                    _, _, X_inf, X_inf_alter, y_inf, y_inf_alter, _, _ = self.get_metrics(k, X_perm, y, fit_params, test_params, return_data=True)
                    if test_params['split'] == "two-split":
                        P_value_perm = self.perm_p_value(k, X_inf, X_inf_alter, y_inf, y_inf_alter, test_params['perturb'], tune_params['num_perm'])
                    else:
                        P_value_perm = self.perm_p_value(k, X_inf, X_inf_alter, y_inf, y_inf_alter, 1., tune_params['num_perm'])
                    P_value_cv.append(P_value_perm)
                P_value_cv = np.array(P_value_cv)
                P_value_cp = np.array([comb_p_value(P_value_cv[:,i], cp=tune_params['cp']) for i in range(tune_params['num_perm'])])
                ## compte Type 1 error
                err_tmp = len(P_value_cp[P_value_cp < self.alpha]) / len(P_value_cp)
                err_lst.append(err_tmp)
                ratio_lst.append(ratio_tmp)
                ## print the estimated Type 1 error
                if tune_params['verbose']>=2:
                    print('(tuneHP: ratio) Est. Type 1 error: %.3f; inf sample ratio: %.3f' %(err_tmp, ratio_tmp))
                ## if the Type I error is under control, early stop
                if err_tmp < self.alpha:
                    find_ratio = 1
                    if tune_params['tune_ratio_method'] == 'fuse':
                        if tune_params['verbose']>=1:
                            print('\U00002705 (tuneHP: ratio) Done with inf sample ratio: %.3f' %(ratio_tmp))
                        break
            
            ## if we can not control the estimated Type I error
            if find_ratio==0:
                warnings.warn("No ratio can control the Type 1 error, pls increase the sample size, and the inference sample ratio is set as the min of ratio_grid.")
                test_params['inf_ratio'] = min(tune_params['ratio_grid'])

        if test_params['perturb'] is None:
            ## check if the test is "one-split" test
            assert test_params['split'] == 'one-split', "Only 'one-split' test need to tune `perturb`."

            X_perm = X.copy()
            X_perm = self.perm_cov(X_perm, k)
            P_value_cv = []
            for h in range(tune_params['cv_num']):
                self.reset_model()
                _, _, X_inf, X_inf_alter, y_inf, y_inf_alter, _, _ = self.get_metrics(k, X_perm, y, fit_params, test_params, return_data=True)
                _, metric_null_perm, metric_alter_perm = self.perm_p_value(k, X_inf, X_inf_alter, y_inf, y_inf_alter, 1., tune_params['num_perm'], True)
                P_value_pb = []
                for perturb_tmp in tune_params['perturb_range']:
                    P_value_perm = [self.pb_ttest(metric_null_perm[i], metric_alter_perm[i], perturb_level=perturb_tmp) for i in range(tune_params['num_perm'])]
                    P_value_pb.append(P_value_perm)
                P_value_cv.append(P_value_pb)
            P_value_cv = np.array(P_value_cv)
            for u in range(len(tune_params['perturb_range'])):
                pb_tmp = tune_params['perturb_range'][u]
                test_params['perturb'] = pb_tmp
                P_value_cp = np.array([comb_p_value(P_value_cv[:,u,i], cp=tune_params['cp']) for i in range(tune_params['num_perm'])])
                err_tmp = len(P_value_cp[P_value_cp < self.alpha]) / len(P_value_cp)
                if tune_params['verbose']>=2:
                    print('(tuneHP: pb) Est. Type 1 error: %.3f; perturbation level: %.3f' %(err_tmp, pb_tmp))
                if err_tmp < self.alpha:
                    find_pb = 1
                    if tune_params['tune_pb_method'] == 'fuse':
                        if tune_params['verbose']>=1:
                            print('\U00002705 (tuneHP: pb) Done with inf pb level: %.3f' %(pb_tmp))
                        break
            assert find_pb, "No pb level can control the Type 1 error, pls increase pb_range in `tune_params`."
        self.test_params = test_params
        return test_params

    def get_metrics(self, k, X, y, fit_params, test_params, return_data=False):
        """
        Return metrics for null/alter models.

        Parameters
        ----------

        X: {array-like} of shape (n_samples, dim_features)
            Instances matrix/tensor, where n_samples in the number of samples and dim_features is the dimension of the features.
             If X is vectorized feature, ``shape`` should be ``(#Samples, dim of feaures)``
             If X is image/matrix data, ``shape`` should be ``(#samples, img_rows, img_cols, channel)``, that is, *X must channel_last image data*.   

        y: {array-like} of shape (n_samples,)
             Output vector/matrix relative to X.

        test_params: {dict of testing parameters}

            split: {'one-split', 'two-split'}, default='one-split'
                one-split or two-split test statistic.

            inf_ratio: float, default=0.5
                A pre-specific inference sample ratio, if ``est_size=None``, then it is determined by adaptive splitting method.

            perturb: float, default=1.
                Perturb level for the one-split test, if ``perturb = None``, then the perturb level is determined by adaptive tunning.

            cv_num: int, default=5
                The number of cross-validation to shuffle the estimation/inference samples in testing.

            cp: {'gmean', 'min', 'hmean', 'Q1', 'hommel', 'cauchy'}, default ='hommel'
                A method to combine p-values obtained from cross-validation.
    
        fit_params: {dict of fitting parameters}
            See keras ``fit``: (https://keras.rstudio.com/reference/fit.html), including ``batch_size``, ``epoch``, ``callbacks``, ``validation_split``, ``validation_data``, and so on.

        Returns
        -------
        P_value: array of float [0, 1]
            The p_values for target hypothesis testings.
        """
        ## default testing params
        self.update_test_params(test_params)
        if test_params['split'] == 'two-split':
            n = len(X) - 2*int(len(X)*test_params['inf_ratio']/2)
        else:
            n = len(X) - int(len(X)*test_params['inf_ratio'])
        ## reset learning rate
        init_lr_null = deepcopy(self.model_null.optimizer.lr.numpy())
        init_lr_alter = deepcopy(self.model_alter.optimizer.lr.numpy())

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n, random_state=42)
        if test_params['split'] == 'two-split':
            X_inf, X_inf_alter, y_inf, y_inf_alter = train_test_split(X_test, y_test, train_size=.5, random_state=42)
        if test_params['split'] == 'one-split':
            X_inf, X_inf_alter, y_inf, y_inf_alter = X_test.copy(), X_test.copy(), y_test.copy(), y_test.copy()
        
        ## initialize the models and learning rates
        self.reset_model()
        self.model_null.optimizer.lr.assign(init_lr_null)
        self.model_alter.optimizer.lr.assign(init_lr_alter)
        
        ## fit, predict, and inference in the null model
        history = self.model_null.fit(X_train, y_train, **fit_params)
        pred_y = self.model_null(X_inf, training=False)
        metric_null = self.metric(y_inf, pred_y)
        
        # fit, predict, and inference in the alternative model
        Z_train = self.alter_feat(X_train, k)
        history_alter = self.model_alter.fit(Z_train, y_train, **fit_params)
        Z_inf = self.alter_feat(X_inf_alter, k)
        pred_y_alter = self.model_alter(Z_inf, training=False)
        metric_alter = self.metric(y_inf_alter, pred_y_alter)

        if return_data:
            return X_train, y_train, X_inf, X_inf_alter, y_inf, y_inf_alter, metric_null, metric_alter
        else:
            return metric_null, metric_alter


    def test_base(self, k, X, y, fit_params, test_params, verbose=0):
        """
        Return p-values for hypothesis testing for inf_feats in class split_test when `inf_ratio` and `perturb` are given.

        Parameters
        ----------

        X: {array-like} of shape (n_samples, dim_features)
            Instances matrix/tensor, where n_samples in the number of samples and dim_features is the dimension of the features.
             If X is vectorized feature, ``shape`` should be ``(#Samples, dim of feaures)``
             If X is image/matrix data, ``shape`` should be ``(#samples, img_rows, img_cols, channel)``, that is, *X must channel_last image data*.   

        y: {array-like} of shape (n_samples,)
             Output vector/matrix relative to X.

        test_params: {dict of testing parameters}

            split: {'one-split', 'two-split'}, default='one-split'
                one-split or two-split test statistic.

            inf_ratio: float, default=0.5
                A pre-specific inference sample ratio, if ``est_size=None``, then it is determined by adaptive splitting method.

            min_inf: int, default=0
                The minimal size for inference sample.

            min_est: int, default=0
                The minimal size for estimation sample.

            perturb: float, default=1.
                Perturb level for the one-split test, if ``perturb = None``, then the perturb level is determined by adaptive tunning.

            cv_num: int, default=5
                The number of cross-validation to shuffle the estimation/inference samples in testing.

            cp: {'gmean', 'min', 'hmean', 'Q1', 'hommel', 'cauchy'}, default ='hommel'
                A method to combine p-values obtained from cross-validation.
    
        fit_params: {dict of fitting parameters}
            See keras ``fit``: (https://keras.rstudio.com/reference/fit.html), including ``batch_size``, ``epoch``, ``callbacks``, ``validation_split``, ``validation_data``, and so on.

        Returns
        -------

        P_value: array of float [0, 1]
            The p_values for target hypothesis testings.
        """

        ## update test params
        test_params = self.update_test_params(test_params)

        ## reset learning rate
        init_lr_null = deepcopy(self.model_null.optimizer.lr.numpy())
        init_lr_alter = deepcopy(self.model_alter.optimizer.lr.numpy())

        ## (two-split) determine the splitting ratio for est and inf samples
        ## testing
        P_value_cv = []
        for h in range(test_params['cv_num']):
            metric_null, metric_alter = self.get_metrics(k, X, y, fit_params, test_params)
            ## compute p-value
            p_value_tmp = self.pb_ttest(metric_null, metric_alter, test_params['perturb'])
            if verbose >= 2:
                print('cv: %d; p_value: %.5f; loss_null: %.5f(%.5f); loss_alter: %.5f(%.5f)' 
                    %(h, p_value_tmp, 
                        metric_null.mean(), metric_null.std(), 
                        metric_alter.mean(), metric_alter.std()))
            P_value_cv.append(p_value_tmp)
        
        self.p_values_comb.append(P_value_cv)
        
        P_value_cv = np.array(P_value_cv)
        p_value_cp = comb_p_value(P_value_cv, cp=test_params['cp'])
        
        if verbose >= 1:
            if p_value_cp < self.alpha:
                print(' \U0001F9EA'+' %d-th Hypothesis: reject H0 with p_value: %.3f' %(k, p_value_cp))
            else:
                print(' \U0001F9EA'+' %d-th Hypothesis: accept H0 with p_value: %.3f' %(k, p_value_cp))
        return p_value_cp, P_value_cv

    def testing(self, X, y, fit_params, test_params={}, tune_params={}):
        """
        Return p-values for hypothesis testing for inf_feats in class split_test.

        Parameters
        ----------

        X: {array-like} of shape (n_samples, dim_features)
            Instances matrix/tensor, where n_samples in the number of samples and dim_features is the dimension of the features.
             If X is vectorized feature, ``shape`` should be ``(#Samples, dim of feaures)``
             If X is image/matrix data, ``shape`` should be ``(#samples, img_rows, img_cols, channel)``, that is, *X must channel_last image data*.   

        y: {array-like} of shape (n_samples,)
             Output vector/matrix relative to X.
    
        fit_params: {dict of fitting parameters}
            See keras ``fit``: (https://keras.rstudio.com/reference/fit.html), including ``batch_size``, ``epoch``, ``callbacks``, ``validation_split``, ``validation_data``, and so on.

        tune_params: {dict of splitting parameters} when ``inf_ratio=None``

            num_perm: int, default=100
                Number of permutation for determine the splitting ratio.

            ratio_grid: list of float (0,1), default=[.2, .4, .6, .8]
                A list of estimation/inference ratios under searching.

            if_reverse: {0,1}, default=0
                ``if_reverse = 0`` indicates the loop of ``ratio_grid`` starts from smallest one to largest one; ``if_reverse = 1`` indicates the loop of ``ratio_grid`` starts from largest one to smallest one.

            perturb_scale: integer, default=5
                The scale of perturb, and the perturbation grid is generated based on ``2**range(-perturb_scale, perturb_scale)*var(losses by full model)``

            tune_ratio_method: {'fuse', 'log-ratio'}, default=`fuse`
                The adaptive splitting method to determine the optimal estimation/inference ratios. 

            cv_num: int, default=`cv_num`
                The number of cross-validation to shuffle the estimation/inference samples in adaptive ratio splitting.

            cp: {'gmean', 'min', 'hmean', 'Q1', 'hommel', 'cauchy'}, default = `cp`
                A method to combine p-values obtained from cross-validation. see (https://arxiv.org/pdf/1212.4966.pdf) for more detail.

            verbose: {0,1}, default=1
                if print the results the adaptive tuning procedure.

        cv_num: int, default=5
            The number of cross-validation to shuffle the estimation/inference samples in testing.

        cp: {'gmean', 'min', 'hmean', 'Q1', 'hommel', 'cauchy'}, default ='hommel'
            A method to combine p-values obtained from cross-validation.

        inf_ratio: float, default=`None`
            A pre-specific inference sample ratio, if ``est_size=None``, then it is determined by adaptive splitting method.

        Returns
        -------

        P_value: array of float [0, 1]
            The p_values for target hypothesis testings.

        """
        self.update_cp_dir()
        ## create checkpoints path
        if not os.path.exists(self.cp_path):
            os.makedirs(self.cp_path)

        ## default splitting params
        test_params=self.update_test_params(test_params)
        tune_params=self.update_tune_params(tune_params)

        test_params_org = test_params.copy()
        ## save initial weights
        self.save_init()
        ## reset learning rate
        init_lr_full = deepcopy(self.model_null.optimizer.lr.numpy())
        init_lr_alter = deepcopy(self.model_alter.optimizer.lr.numpy())

        P_value, P_value_cv = [], []
        
        for k in range(len(self.inf_feats)):
            print(" %s test for %d-th Hypothesis ".center(70, '=') %(test_params['split'], k))
            ## initialize the models and learning rates
            self.reset_model()
            test_params = test_params_org
            test_params = self.tuneHP(k, X, y, fit_params, test_params, tune_params)
            p_value_cp_tmp, p_value_cv_tmp = self.test_base(k, X, y, fit_params, test_params, verbose=test_params['verbose'])
            P_value.append(p_value_cp_tmp)
            P_value_cv.append(list(p_value_cv_tmp))
        # return P_value, fit_err, P_value_cv
        self.p_values = np.array(P_value)
        self.p_values_comb = np.array(P_value_cv)
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
                    #if X.shape[-1]==3:
                    #    X_mask_tmp = X_mask_tmp[:,:,0] 
                    ax = fig.add_subplot(spec[row, col])
                    im1 = ax.imshow(X_demo[row], vmin=0, vmax=1, **plt_params)
                    ax.axis('off')
                    im2 = ax.imshow(X_mask_tmp, vmin=0, vmax=1, **plt_mask_params)
                    ax.axis('off')
                    if row == 0:
                        ax.set_title('p_values: %.3f' %self.p_values[col])
            plt.subplots_adjust(top = 0.95, bottom=0.05, hspace=0.03, wspace=0.03, right=0.95)
            plt.show()


class perm_test(object):
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

    Methods
    -------
    testing
    """

    def __init__(self, inf_feats, model, model_perm, alpha=.05, num_folds=5, num_perm=100, verbose=0, eva_metric='mse'):
        self.inf_feats = inf_feats
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
            #   if isinstance(layer, tf.keras.Model):
            #       reset_weights(layer)
            #       continue
            #   for k, initializer in layer.__dict__.items():
            #       if "initializer" not in k:
            #           continue
            #           # find the corresponding variable
            #       var = getattr(layer, k.replace("_initializer", ""))
            #       var.assign(initializer(var.shape, var.dtype))

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

    def mask_cov(self, X, k=0):
        Z = X.copy()
        if type(self.inf_feats[k]) is list:
            Z[:,self.inf_feats[k][0][:,None], self.inf_feats[k][1], 0] = 0.
        else:
            Z[:,self.inf_feats[k]]= 0.
        return Z

    def perm_cov(self, X, k=0):
        Z = X.copy()
        if type(self.inf_feats[k]) is list:
            Z[:,self.inf_feats[k][0][:,None], self.inf_feats[k][1], 0]= np.random.permutation(Z[:,self.inf_feats[k][0][:,None], self.inf_feats[k][1], 0])
        else:
            Z[:,self.inf_feats[k]]= np.random.permutation(Z[:,self.inf_feats[k]])
        return Z

    def noise_cov(self, X, k=0):
        Z = X.copy()
        Z[:,self.inf_feats[k]] = np.random.randn(len(X), len(self.inf_feats[k]))
        return Z

    def testing(self, X, y, fit_params={}):
        """
        Return p-values for hypothesis testing for inf_feats in class perm_test.

        Parameters
        ----------

        X : array-like | shape = (n_samples, dim_features)
            Instances matrix/tensor, where n_samples in the number of samples and dim_features is the dimension of the features.
             If X is vectorized feature, ``shape`` should be ``(#Samples, dim of feaures)``
             If X is image/matrix data, ``shape`` should be ``(#samples, img_rows, img_cols, channel)``, that is, *X must channel_last image data*.   
        
        y: {array-like} of shape (n_samples,)
             Output vector/matrix relative to X.

        fit_params: {dict of fitting parameters}
            See keras ``fit``: (https://keras.rstudio.com/reference/fit.html), including ``batch_size``, ``epoch``, ``callbacks``, ``validation_split``, ``validation_data``, and so on.

        cv_num: int, default=5
            The number of cross-validation to shuffle the estimation/inference samples in testing.

        cp: {'gmean', 'min', 'hmean', 'Q1', 'hommel', 'cauchy'}, default ='hommel'
            A method to combine p-values obtained from cross-validation.

        inf_ratio: float, default=None
            A pre-specific inference sample ratio, if ``est_size=None``, then it is determined by adaptive splitting method ``metric``.

        Returns
        -------

        P_value: array of float [0, 1]
            The p_values for target hypothesis testings.

        """
        P_value = []
        for k in range(len(self.inf_feats)):
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



class Hperm_test(object):
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

    Methods
    -------
    testing
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
            metric_tmp = np.array((y_true - y_pred)**2).flatten()
        elif self.eva_metric == 'mae':
            metric_tmp = np.array(abs(y_true - y_pred)).flatten()
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
        Save the initialization for the network model under class HPT
        """
        self.model.save_weights(self.cp_path+'/model_init.h5')
        # self.model_alter.save_weights(self.cp_path+'/model_alter_init.h5')

    def reset_model(self):
        """
        Reset the full and mask network models under class HPT
        """
        self.model.load_weights(self.cp_path+'/model_init.h5')
        # self.model_alter.load_weights(self.cp_path+'/model_alter_init.h5')

    # def reset_model(self):
    #   if int(tf.__version__[0]) == 2:
    #       # for layer in self.model.layers:
    #       #   if isinstance(layer, tf.keras.Model):
    #       #       reset_weights(layer)
    #       #       continue
    #       #   for k, initializer in layer.__dict__.items():
    #       #       if "initializer" not in k:
    #       #           continue
    #       #           # find the corresponding variable
    #       #       var = getattr(layer, k.replace("_initializer", ""))
    #       #       var.assign(initializer(var.shape, var.dtype))
    #
    #       for layer in self.model.layers:
    #           if isinstance(layer, tf.keras.Model): #if you're using a model as a layer
    #               reset_weights(layer) #apply function recursively
    #               continue
    #
    #           #where are the initializers?
    #           if hasattr(layer, 'cell'):
    #               init_container = layer.cell
    #           else:
    #               init_container = layer
    #
    #           for key, initializer in init_container.__dict__.items():
    #               if "initializer" not in key: #is this item an initializer?
    #                 continue #if no, skip it
    #
    #               # find the corresponding variable, like the kernel or the bias
    #               if key == 'recurrent_initializer': #special case check
    #                   var = getattr(init_container, 'recurrent_kernel')
    #               else:
    #                   var = getattr(init_container, key.replace("_initializer", ""))
    #
    #               if var is None:
    #                   continue
    #               else:
    #                   var.assign(initializer(var.shape, var.dtype))
    #
    #   if int(tf.__version__[0]) == 1:
    #       session = K.get_session()
    #       for layer in self.model.layers:
    #           if hasattr(layer, 'kernel_initializer'):
    #               layer.kernel.initializer.run(session=session)
    #           if hasattr(layer, 'bias_initializer'):
    #               layer.bias.initializer.run(session=session)
    #       for layer in self.model_perm.layers:
    #           if hasattr(layer, 'kernel_initializer'):
    #               layer.kernel.initializer.run(session=session)
    #           if hasattr(layer, 'bias_initializer'):
    #               layer.bias.initializer.run(session=session)

    ## can be extent to @abstractmethod
    # def mask_cov(self, X, k=0):
    #   """
    #   Return instances with masked k-th hypothesized features.
    #
    #   Parameters
    #   ----------
    #   X : array-like
    #    Target instances.
    #
    #   k : integer, default = 0
    #    k-th hypothesized features in inf_feats
    #   """
    #   Z = X.copy()
    #   if type(self.inf_feats[k]) is list:
    #       Z[:,self.inf_feats[k][0][:,None], self.inf_feats[k][1], 0] = 0.
    #   else:
    #       Z[:,self.inf_feats[k]]= 0.
    #   return Z

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
             If X is image/matrix data, ``shape`` should be ``(#samples, img_rows, img_cols, channel)``, that is, *X must channel_last image data*.   

        y: {array-like} of shape (n_samples,)
             Output vector/matrix relative to X.

        fit_params: {dict of fitting parameters}
            See keras ``fit``: (https://keras.rstudio.com/reference/fit.html), including ``batch_size``, ``epoch``, ``callbacks``, ``validation_split``, ``validation_data``, and so on.

        cv_num: int, default=5
            The number of cross-validation to shuffle the estimation/inference samples in testing.

        cp: {'gmean', 'min', 'hmean', 'Q1', 'hommel', 'cauchy'}, default ='hommel'
            A method to combine p-values obtained from cross-validation.

        inf_ratio: float, default=None
            A pre-specific inference sample ratio, if ``est_size=None``, then it is determined by adaptive splitting method ``metric``.

        Returns
        -------

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
                # path_tmp = self.cp_path+'/model'+'_inf'+str(k)+'_cv'+str(h)+'.h5'
                # self.model.save_weights(path_tmp)
                # self.model.load_weights(path_tmp)
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
