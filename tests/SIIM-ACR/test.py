import numpy as np
import keras
from keras.applications import VGG19
from keras.layers import Dense, Dropout
from keras.models import Model
from keras import models
from keras import layers
from keras import optimizers
from tensorflow.python.keras import backend as K
import time

K.clear_session()

data = np.load('/home/ben/data/SIIM-ACR/siim.npz')
mask, X, y = data['name1'], data['name2'], data['name3']
y = keras.utils.to_categorical(y, 2)
# Create the base model of VGG19
vgg19 = VGG19(weights='imagenet', include_top=False, input_shape = (256, 256, 3), classes = 2)

def dense_nn():
	model = models.Sequential()
	model.add(layers.Dense(512, activation='relu', input_dim= 8*8*512))
	model.add(layers.Dense(256, activation='relu'))
	model.add(layers.Dense(2, activation="softmax"))

	# Compile the model
	model.compile(loss=keras.losses.categorical_crossentropy,
				  optimizer=keras.optimizers.Adam(.001),
				  metrics=['accuracy'])
	return model


## training for the original dataset
# Preprocessing the input
# convert class vectors to binary class matrices

tic = time.perf_counter()
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from pre_data import gen_feature

model_full, model_mask = dense_nn(), dense_nn()

es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20, restore_best_weights=True)

fit_params = {'callbacks': [es],
			  'epochs': 100,
			  'batch_size': 32,
			  'validation_split': .2,
			  'verbose': 1}

for ratio_tmp in [.2, .3, .4]:
	m_tmp = int(len(X)*ratio_tmp)
	n_tmp = len(X) - 2*m_tmp
	X_train, X_test, mask_train, mask_test, y_train, y_test = train_test_split(X, mask, y, train_size=n_tmp, random_state=42)
	X_mean = np.mean(X_train, axis=0)
	Z_train, Z_test = X_train*mask_train + (1.-mask_train)*X_mean, X_test*mask_test + (1.-mask_test)*X_mean
	## generta VGG features for image
	X_train_feature, X_test_feature = gen_feature(X_train, vgg19), gen_feature(X_test, vgg19)
	Z_train_feature, Z_test_feature = gen_feature(Z_train, vgg19), gen_feature(Z_test, vgg19)
	## fit both model_full and model_mask
	history = model_full.fit(x=X_train_feature, y=y_train, **fit_params)
	history_mask = model_mask.fit(x=Z_train_feature, y=y_train, **fit_params)
	## predict label
	if split == 'two_sample':
		ind_inf, ind_inf_mask = train_test_split(range(len(y_test)), train_size=m_tmp, random_state=42)
	else:
		ind_inf, ind_inf_mask = range(len(y_test)), range(len(y_test))
	label_test = np.argmax(y_test, 1)
	## predict for full model
	pred_y_full = model_full.predict(X_test_feature)
	pred_label_full = np.argmax(pred_y_full, 1)
	metric_full = 1. - 1.*(pred_label_full[ind_inf] == label_test[ind_inf])
	## predict for full model
	pred_y_mask = model_mask.predict(Z_test_feature)
	pred_label_mask = np.argmax(pred_y_mask, 1)
	metric_mask = 1. - 1.*(pred_label_mask[ind_inf_mask] == label_test[ind_inf_mask])
	## construct statistic
	diff = metric_full - metric_mask

toc = time.perf_counter()
print('testing time: %.3f' %(toc-tic))

## training for the masked dataset
X_stacked = np.stack((X,)*3, axis=-1)
X_pro = preprocess_input(X_stacked)

X_features = vgg19.predict(X_pro, batch_size=32, verbose=1)
X_features = np.reshape(X_features, (len(X_features), 8*8*512))


X_features = vgg19.predict(X_pro, batch_size=16, verbose=1)
X_features = np.reshape(X_features, (len(X_features), 8*8*512))

# convert class vectors to binary class matrices


tic = time.perf_counter()
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

model = dense_nn()

es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20, restore_best_weights=True)

fit_params = {'callbacks': [es],
			  'epochs': 100,
			  'batch_size': 32,
			  'validation_split': .2,
			  'verbose': 1}

for ratio_tmp in [.2, .3, .4]:
	m_tmp = int(len(X_features)*ratio_tmp)
	n_tmp = len(X_features) - 2*m_tmp
	X_train, X_test, y_train, y_test = train_test_split(X_features, y, train_size=n_tmp, random_state=42)
	history = model.fit(x=X_train, y=y_train, **fit_params)

toc = time.perf_counter()
print('testing time: %.3f' %(toc-tic))
