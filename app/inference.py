from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras import backend as K
import numpy as np

batch_size = 16
num_classes = 2
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x = np.vstack((x_train, x_test))
y = np.hstack((y_train, y_test))
ind = (y == 9) + (y == 7)
x, y = x[ind], y[ind]
x = x.astype('float32')
x += .01*np.random.randn(14251, 28, 28)
y[y==7] = 0
y[y==9] = 1

LHR, LH_true, LH_blank = [], [], []

for i in range(140):
	# K.clear_session()
	print('Split for %s iteration' %i)
	from sklearn.model_selection import train_test_split
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=i)
	x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=.3, random_state=i)

	## blank the image
	x_train_blank, x_valid_blank, x_test_blank = x_train.copy(), x_valid.copy(), x_test.copy()
	x_train_blank[:,19:28, 21:28], x_valid_blank[:,19:28, 21:28], x_test_blank[:,19:28, 21:28] = 0, 0, 0
	# test 4: x_train_blank[:,21:28,4:13], x_valid_blank[:,21:28,4:13], x_test_blank[:,21:28,4:13] = 0, 0, 0
	# test 3: x_train_blank[:,19:28,13:20], x_valid_blank[:,19:28,13:20], x_test_blank[:,19:28,13:20] = 0, 0, 0

	if K.image_data_format() == 'channels_first':
	  x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	  x_valid = x_valid.reshape(x_valid.shape[0], 1, img_rows, img_cols)
	  x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	  x_train_blank = x_train_blank.reshape(x_train_blank.shape[0], 1, img_rows, img_cols)
	  x_valid_blank = x_valid_blank.reshape(x_valid_blank.shape[0], 1, img_rows, img_cols)
	  x_test_blank = x_test_blank.reshape(x_test_blank.shape[0], 1, img_rows, img_cols)
	  input_shape = (1, img_rows, img_cols)
	else:
	  x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	  x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols, 1)
	  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	  x_train_blank = x_train_blank.reshape(x_train_blank.shape[0], img_rows, img_cols, 1)
	  x_valid_blank = x_valid_blank.reshape(x_valid_blank.shape[0], img_rows, img_cols, 1)
	  x_test_blank = x_test_blank.reshape(x_test_blank.shape[0], img_rows, img_cols, 1)
	  input_shape = (img_rows, img_cols, 1)

	x_train = x_train.astype('float32')
	x_valid = x_valid.astype('float32')
	x_test = x_test.astype('float32')

	x_train /= 255
	x_valid /= 255
	x_test /= 255

	x_train_blank = x_train_blank.astype('float32')
	x_valid_blank = x_valid_blank.astype('float32')
	x_test_blank = x_test_blank.astype('float32')

	x_train_blank /= 255
	x_valid_blank /= 255
	x_test_blank /= 255
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_valid = keras.utils.to_categorical(y_valid, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	def cnn():
	  model = Sequential()
	  model.add(Conv2D(32, kernel_size=(3, 3),
	           activation='relu',
	           input_shape=input_shape))
	  model.add(Conv2D(64, (3, 3), activation='relu'))
	  model.add(MaxPooling2D(pool_size=(2, 2)))
	  model.add(Dropout(0.25))
	  model.add(Flatten())
	  model.add(Dense(128, activation='relu'))
	  model.add(Dropout(0.5))
	  model.add(Dense(num_classes, activation='softmax'))
	  return model

	from keras.callbacks import EarlyStopping
	true_ = cnn()
	es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=5)

	# true_.compile(loss=keras.losses.binary_crossentropy,
	#         optimizer=keras.optimizers.Adadelta(),
	#         metrics=['accuracy'])

	# true_.fit(x_train, y_train,
	#       batch_size=batch_size,
	#       epochs=epochs,
	#       verbose=1,
	#       callbacks=[es],
	#       validation_data=(x_valid, y_valid))

	# score = true_.evaluate(x_test, y_test, verbose=0)
	score = [0, 0]

	## fit the blank model
	blank_ = cnn()
	es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=5)
	blank_.compile(loss=keras.losses.binary_crossentropy,
	        optimizer=keras.optimizers.Adadelta(),
	        metrics=['accuracy'])
	blank_.fit(x_train_blank, y_train,
	      batch_size=batch_size,
	      epochs=epochs,
	      verbose=1,
	      callbacks=[es],
	      validation_data=(x_valid_blank, y_valid))

	score_blank = blank_.evaluate(x_test_blank, y_test, verbose=0)

	# print('Test loss:', score[0])
	# print('Test accuracy:', score[1])
	print('likelihood for true_: %.4f; likelihhod for blank_: %.4f; likelihhod ratio: %.4f' %(score[0], score_blank[0], score[0] - score_blank[0]))
	LHR.append(score_blank[0] - score[0])
	LH_true.append(score[0])
	LH_blank.append(score_blank[0])
	# from matplotlib import pyplot as plt
	# plt.axis('off')
	# plt.imshow(x_train_blank[0,:,:,0])
	# plt.imshow(x_train[0,:,:,0])
	# # plt.imshow(masked, 'gray', alpha=0.5)
	# plt.show()

# plt.show()

LH_true, LH_blank = np.array(LH_true), np.array(LH_blank)

from matplotlib import pyplot as plt
plt.axis('off')
masked = np.zeros((28, 28))
# masked[7:16,9:16] = 1.
masked[19:28,13:20] = 1.
plt.imshow(x_train[11,:,:,0])
# plt.imshow(tmp[2,:,:,0])
# plt.imshow(x_train[0,:,:,0])
plt.imshow(masked, 'gray', alpha=0.5)
plt.show()