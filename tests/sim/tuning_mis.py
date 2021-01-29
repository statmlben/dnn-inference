## linear_sim.py

import numpy as np
import sim_data
from functools import partial
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import keras.backend as K
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time
# from DnnT import DnnT
from dnn_inference import DnnT
import keras
from likelihood_ratio_test import likelihood_ratio_test
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression

array32 = partial(np.array, dtype=np.float32)
np.random.seed(0)

L0, d0 = 2, 32
N, p = 4000, 10
print('the number of sample: %d; number of dim: %d' %(N, p))

## Generate data
X = np.random.randn(N,p)
y = .1*X[:,5] + .2*X[:,6]**2 + .3*X[:,7]**3 + .4*X[:,8]*X[:,9] + .3*np.random.randn(N)

def Reg_model(p, d, L=3, optimizer=Adam(lr=.0005)):
    model = Sequential()
    model.add(Dense(d, input_dim=p, activation='relu'))
    for l in range(L-2):
        model.add(Dense(d, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

es = EarlyStopping(monitor='val_loss', mode='min', verbose=0,
                    patience=50, restore_best_weights=True)

fit_params = {'callbacks': [es],
              'epochs': 2000,
              'batch_size': 64,
              'validation_split': .2,
              'verbose': 1}

model_full = Reg_model(p=p, d=d0, L=L0)
his = model_full.fit(X, y,**fit_params)

## plot y_fit and y_predict
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set()
y_pred = model_full.predict(X).flatten()
d = {'y_true': list(y), 'y_predict': y_pred}
df = pd.DataFrame(data=d)
sns.scatterplot(data=df, x="y_true", y="y_predict", s=20)
plt.show()
