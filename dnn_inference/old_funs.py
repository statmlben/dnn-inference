def adaRatio(self, X, y, k=0, cat_feats=[], fit_params={}, tune_params={}):
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

    tune_params_default = { 'split': 'one-split',
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
    
    tune_params_default.update(tune_params)
    tune_params = tune_params_default

    perturb=tune_params['perturb']
    split=tune_params['split']
    perturb_scale=tune_params['perturb_scale']
    ratio_grid=tune_params['ratio_grid']
    if_reverse=tune_params['if_reverse']
    min_inf=tune_params['min_inf']
    min_est=tune_params['min_est']
    ratio_method=tune_params['ratio_method']
    num_perm=tune_params['num_perm']
    cv_num= tune_params['cv_num']
    cp = tune_params['cp']
    verbose= tune_params['verbose']

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
                # training for null model
                history = self.model.fit(x=X_train, y=y_train, **fit_params)

                # training for mask model
                Z_train = self.alter_feat(X_train, k)
                history_mask = self.model_alter.fit(x=Z_train, y=y_train, **fit_params)
                Z_test = self.alter_feat(X_test, k)

                # evaluation for mask model
                pred_y_mask = self.model_alter.predict(Z_test)
                for j in range(num_perm):
                    X_test_perm = X_test.copy()
                    X_test_perm = self.perm_cov(X_test_perm, k)
                    pred_y = self.model.predict(X_test_perm)
                    ind_inf, ind_inf_mask = train_test_split(range(len(pred_y)), train_size=m_tmp, random_state=42)
                    metric_tmp = self.metric(y_test[ind_inf], pred_y[ind_inf])
                    metric_mask_tmp = self.metric(y_test[ind_inf_mask], pred_y_mask[ind_inf_mask])
                    p_value_tmp = self.pb_ttest(metric_tmp, metric_mask_tmp)
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
                    # training for null model
                    history = self.model.fit(x=X_train, y=y_train, **fit_params)
                    
                    # training for mask model
                    Z_train = self.alter_feat(X_train, k)
                    history_mask = self.model_alter.fit(x=Z_train, y=y_train, **fit_params)
                    
                    ## evaluation for the mask model
                    Z_test = self.alter_feat(X_test, k)
                    pred_y_mask = self.model_alter.predict(Z_test)
                    metric_mask_tmp = self.metric(y_test, pred_y_mask)

                    # evaluation
                    for j in range(num_perm):
                        X_test_perm = X_test.copy()
                        X_test_perm = self.perm_cov(X_test_perm, k)
                        ## compute the metric based on null model
                        pred_y = self.model.predict(X_test_perm)
                        metric_tmp = self.metric(y_test, pred_y)
                        ## compute the p-value based on the diff metrics
                        p_value_tmp = self.pb_ttest(metric_tmp, metric_mask_tmp, 
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