import numpy as np
import sim_data
from functools import partial
from keras.models import Sequential, Model, Input
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import keras.backend as K
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time
from dnn_inference import DnnT

p, L0, d0, K0 = 100, 3, 128, 5
tau, x_max, pho = 2., .4, .00
N = 6000

X = sim_data.gen_X(n=N, p=p, pho=pho, x_max=x_max, distribution='normal')
X0 = X.copy()
X0[:,:K0] = 0.
y = sim_data.gen_Y(p=p, d=d0, L=L0, X=X0, tau=tau, K0=K0, noise=1.)

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X0, y)
y_pred_linear = reg.predict(X0)

## report r_square for linear model
SSR = np.mean((y_pred - y)**2)
SST = np.mean((y - y.mean())**2)
r_sqaure_linear = 1 - SSR/SST
## report r_square for NN

def Reg_model(p, d, L=3):
	# Create a simple model.
	inputs = Input(shape=(p,))
	hidden_layer = Dense(d, use_bias=False, input_dim=p, activation='relu')(inputs)
	for l in range(L-2):
		hidden_layer = Dense(d, use_bias=False, activation='relu')(hidden_layer)
	outputs = Dense(1, use_bias=False)(hidden_layer)
	# outputs = Dense(1, use_bias=False, activation='relu')(hidden_layer)
	model = Model(inputs, outputs)
	# model.compile(loss='mean_squared_error', optimizer=optimizer)
	return model

model = Reg_model(p=p, d=d0, L=L0)
model.compile(loss='mean_squared_error', optimizer=Adam(lr=.0005))
model.fit(X0, y, epochs=100, batch_size=128)

y_pred_nn = model.predict(X0).flatten()
SSR_nn = np.mean((y_pred_nn - y)**2)
r_sqaure_nn = 1 - SSR_nn/SST

print('r_square for linear model: %.3f; nerual network: %.3f' %(r_sqaure_linear, r_sqaure_nn))


y_pred = []
y_pred.extend(y_pred_linear)
y_pred.extend(y_pred_nn)
y_pred = np.array(y_pred)

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set()

d = {'y_true': list(y)*2, 'y_predict': y_pred, 'method': ['linear']*N + ['nerual network']*N}
df = pd.DataFrame(data=d)
sns.scatterplot(data=df, x="y_true", y="y_predict", s=20, hue="method", style='method')
plt.show()

