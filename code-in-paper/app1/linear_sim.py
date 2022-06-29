import numpy as np
from sklearn.model_selection import train_test_split

n, d = 10000, 100
## generate input data 
X = np.random.randn(n, d)
## generate beta: Null hypothesis is true
true_beta = np.random.randn(d)
true_beta[:10] = 0.

LH_diff, LH_true, LH_mask = [], [], []
for i in range(500):
	# generate labels
	y = np.sign(np.dot(X, true_beta) + .1*np.random.randn(n))

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=42)

	X_train_mask, X_test_mask = X_train.copy(), X_test.copy()
	X_train_mask[:,:10], X_test_mask[:,:10] = 0., 0.

	from sklearn.model_selection import KFold
	from sklearn.linear_model import LogisticRegression
	from sklearn.model_selection import GridSearchCV
	from sklearn.metrics import roc_auc_score, log_loss

	logit_param_grid = {'C': [20, 40, 60, 80, 100, 120, 140]}

	## cv for original dataset
	logit_grid = GridSearchCV(LogisticRegression(solver='lbfgs', tol=.001, max_iter=10000), logit_param_grid, 
							  scoring='neg_log_loss', cv=5, n_jobs=5, return_train_score=True, verbose=1)
	logit_grid.fit(X_train, y_train)
	print('best C for orginal LogisticRegression is %.3f' %logit_grid.best_params_['C'])

	## cv for original dataset
	logit_grid_mask = GridSearchCV(LogisticRegression(solver='lbfgs', tol=.001, max_iter=10000), logit_param_grid, 
							  scoring='neg_log_loss', cv=5, n_jobs=5, return_train_score=True, verbose=1)
	logit_grid_mask.fit(X_train_mask, y_train)
	print('best C for orginal LogisticRegression is %.3f' %logit_grid.best_params_['C'])

	pred, pred_mask = logit_grid.predict_proba(X_test), logit_grid_mask.predict_proba(X_test_mask)
	score, score_mask = log_loss(y_test, pred), log_loss(y_test, pred_mask)

	LH_true.append(score)
	LH_mask.append(score_mask)
	LH_diff_tmp = score - score_mask
	LH_diff.append(LH_diff_tmp)

LH_diff, LH_true, LH_mask = np.array(LH_diff), np.array(LH_true), np.array(LH_mask)

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

sns.set(color_codes=True)
LHR_dis = sns.distplot(LH_diff, fit=norm, kde=False);
# LHR_dis.set(xlim=(-.03, .04))
plt.title('Histgram of the likelihhod ratio based on 350 samples of random split')
plt.ylabel('count')
plt.show()

sns.distplot(LH_true, fit=norm, kde=False);
sns.distplot(LH_mask, fit=norm, kde=False);
# plt.set(xlim=(None, .04))
plt.title('Histgram of two distributions of likelihhod based on 350 samples of random split')
plt.ylabel('count')
plt.show()