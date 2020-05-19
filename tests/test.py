import numpy as np
from sklearn.datasets import make_classification
from VarSVM import noneg_driftsvm

X, y = make_classification(n_features=4, random_state=0)
y = y * 2 - 1

n = len(X)
drift = .28*np.ones(n)

clf = noneg_driftsvm()
clf.fit(X=X, y=y, drift=drift)
clf.decision_function(X=X, drift=drift) * y