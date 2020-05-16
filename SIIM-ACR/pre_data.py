import numpy as np
from keras.applications.vgg19 import preprocess_input


def gen_feature(X, pre_trained_model):
	X_stacked = np.stack((X,)*3, axis=-1)
	X_pro = preprocess_input(X_stacked)

	X_features = pre_trained_model.predict(X_pro, batch_size=32, verbose=1)
	X_features = np.reshape(X_features, (len(X_features), 8*8*512))
	return X_features