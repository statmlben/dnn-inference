import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
# from mlxtend.plotting import plot_confusion_matrix

from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.utils import to_categorical

path = '/home/statmlben/dataset/facial-expression-recognition/'
os.listdir(path)

data = pd.read_csv(path+'icml_face_data.csv')
data.head()

def prepare_data(data):
    """ Prepare data for modeling 
        input: data frame with labels und pixel data
        output: image and label array """
    
    image_array = np.zeros(shape=(len(data), 48, 48))
    image_label = np.array(list(map(int, data['emotion'])))
    
    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, ' pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        image_array[i] = image
        
    return image_array, image_label

def plot_examples(label=0):
    fig, axs = plt.subplots(1, 5, figsize=(25, 12))
    fig.subplots_adjust(hspace = .2, wspace=.2)
    axs = axs.ravel()
    for i in range(5):
        idx = data[data['emotion']==label].index[i]
        axs[i].imshow(train_images[idx][:,:,0], cmap='gray')
        axs[i].set_title(emotions[train_labels[idx].argmax()])
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])
        
def plot_all_emotions():
    fig, axs = plt.subplots(1, 7, figsize=(30, 12))
    fig.subplots_adjust(hspace = .2, wspace=.2)
    axs = axs.ravel()
    for i in range(7):
        idx = data[data['emotion']==i].index[i]
        axs[i].imshow(train_images[idx][:,:,0], cmap='gray')
        axs[i].set_title(emotions[train_labels[idx].argmax()])
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])
        
def plot_image_and_emotion(test_image_array, test_image_label, pred_test_labels, image_number):
    """ Function to plot the image and compare the prediction results with the label """
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
    
    bar_label = emotions.values()
    
    axs[0].imshow(test_image_array[image_number], 'gray')
    axs[0].set_title(emotions[test_image_label[image_number]])
    
    axs[1].bar(bar_label, pred_test_labels[image_number], color='orange', alpha=0.7)
    axs[1].grid()
    
    plt.show()
    
def plot_compare_distributions(array1, array2, title1='', title2=''):
    df_array1 = pd.DataFrame()
    df_array2 = pd.DataFrame()
    df_array1['emotion'] = array1.argmax(axis=1)
    df_array2['emotion'] = array2.argmax(axis=1)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
    x = emotions.values()
    
    y = df_array1['emotion'].value_counts()
    keys_missed = list(set(emotions.keys()).difference(set(y.keys())))
    for key_missed in keys_missed:
        y[key_missed] = 0
    axs[0].bar(x, y.sort_index(), color='orange')
    axs[0].set_title(title1)
    axs[0].grid()
    
    y = df_array2['emotion'].value_counts()
    keys_missed = list(set(emotions.keys()).difference(set(y.keys())))
    for key_missed in keys_missed:
        y[key_missed] = 0
    axs[1].bar(x, y.sort_index())
    axs[1].set_title(title2)
    axs[1].grid()
    
    plt.show()


emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
			4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

train_image_array, train_image_label = prepare_data(data[data[' Usage']=='Training'])
val_image_array, val_image_label = prepare_data(data[data[' Usage']=='PrivateTest'])
test_image_array, test_image_label = prepare_data(data[data[' Usage']=='PublicTest'])

image_array = np.vstack((train_image_array, val_image_array, test_image_array))
labels = np.hstack((train_image_label, val_image_label, test_image_label))

image_array = image_array.reshape((image_array.shape[0], 48, 48, 1))

# train_images = train_image_array.reshape((train_image_array.shape[0], 48, 48, 1))
# train_images = train_images.astype('float32')/255
# val_images = val_image_array.reshape((val_image_array.shape[0], 48, 48, 1))
# val_images = val_images.astype('float32')/255
# test_images = test_image_array.reshape((test_image_array.shape[0], 48, 48, 1))
# test_images = test_images.astype('float32')/255

# train_labels = to_categorical(train_image_label)
# val_labels = to_categorical(val_image_label)
# test_labels = to_categorical(test_image_label)

# plot_all_emotions()

## detect the facial landmarks
import cv2
import dlib

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # read the image
# img = cv2.imread("face.jpg")

# # Convert image into grayscale
# gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

import imutils

left_eye, right_eye, nose, mouth = [], [], [], []
valid_ind = []

for img_tmp in image_array:
    img_tmp = np.array(img_tmp, dtype=np.uint8)

    # Use detector to find landmarks
    faces = detector(img_tmp, 1)
    
    if len(faces) == 1:
        landmarks = predictor(image=img_tmp, box=faces[0])
        left_eye_tmp = [landmarks.part(38).x, landmarks.part(38).y] + .3*np.random.randn(2)
        right_eye_tmp = [landmarks.part(44).x, landmarks.part(44).y] + .3*np.random.randn(2)
        nose_tmp = [landmarks.part(30).x, landmarks.part(30).y] + .3*np.random.randn(2)
        mouth_tmp = [landmarks.part(66).x, landmarks.part(66).y] + .3*np.random.randn(2)
        if (landmarks.part(30).y > 22) \
            and (landmarks.part(30).y < 35) \
            and (landmarks.part(30).y - landmarks.part(40).y > 3) \
            and (landmarks.part(30).y - landmarks.part(46).y > 3) \
            and (landmarks.part(30).x - landmarks.part(39).x > 2) \
            and (landmarks.part(42).x - landmarks.part(30).x > 2) \
            and (landmarks.part(66).y - landmarks.part(30).y > 1) \
            and (abs(landmarks.part(27).x - 25) < 3):
        # if ( (left_eye_tmp[1]<=23) and (right_eye_tmp[1]<=23) and (nose_tmp[1]<=33) and (nose_tmp[1]>=23) and (mouth_tmp[1]>=33) ):
            valid_ind.append(1)
            # append data
            left_eye.append(left_eye_tmp)
            right_eye.append(right_eye_tmp)
            nose.append(nose_tmp)
            mouth.append(mouth_tmp)
        else:
            valid_ind.append(0)
    else:
        valid_ind.append(0)

valid_ind = np.array(valid_ind)
left_eye = np.array(left_eye, dtype=np.float32)
right_eye = np.array(right_eye, dtype=np.float32)
nose = np.array(nose, dtype=np.float32)
mouth = np.array(mouth, dtype=np.float32)

valid_image_array = image_array[np.where(valid_ind==1)[0]]
valid_label = labels[np.where(valid_ind==1)[0]]

n = len(left_eye)
df = {'x': [], 'y': [], 'keypoints': []}
df['x'].extend(left_eye[:,0])
df['y'].extend(48 - left_eye[:,1])
df['keypoints'].extend(['left_eye']*n)

df['x'].extend(right_eye[:,0])
df['y'].extend(48 - right_eye[:,1])
df['keypoints'].extend(['right_eye']*n)

df['x'].extend(nose[:,0])
df['y'].extend(48 - nose[:,1])
df['keypoints'].extend(['nose']*n)

df['x'].extend(mouth[:,0])
df['y'].extend(48 - mouth[:,1])
df['keypoints'].extend(['mouth']*n)

df = pd.DataFrame(df)

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sns.set(style="whitegrid")

lm = sns.scatterplot(data=df, x="x", y="y", 
            # col='keypoints', kind="scatter",
            hue="keypoints", 
            s=20, alpha=.4)
lm.set(xlim=(10, 43))
rect_left_eye = patches.Rectangle((12, 26), 8, 12, linewidth=3, 
                         edgecolor='b', facecolor='b',alpha=0.4)
rect_right_eye = patches.Rectangle((30, 26), 8, 12, linewidth=3, 
                         edgecolor='y', facecolor='y',alpha=0.4)
rect_nose = patches.Rectangle((20, 16), 8, 9, linewidth=3, 
                         edgecolor='g', facecolor='g',alpha=0.4)
rect_mouth = patches.Rectangle((18, 3), 11, 12, linewidth=3, 
                         edgecolor='r', facecolor='r',alpha=0.4)
lm.add_patch(rect_left_eye)
lm.add_patch(rect_right_eye)
lm.add_patch(rect_nose)
lm.add_patch(rect_mouth)
plt.show()

for i in range(7):
    plt.axis('off')
    ind_tmp = np.where(valid_label == i)[0][203]
    plt.imshow(valid_image_array[ind_tmp], cmap='gray', vmin=0, vmax=255)
    ax = plt.gca()
    # plt.scatter(left_eye[ind_tmp,0], left_eye[ind_tmp,1], s=100, c='blue', marker='o')
    # plt.scatter(right_eye[ind_tmp,0], right_eye[ind_tmp,1], s=100, c='yellow', marker='o')
    # plt.scatter(nose[ind_tmp,0], nose[ind_tmp,1], s=100, c='green', marker='o')
    # plt.scatter(mouth[ind_tmp,0], mouth[ind_tmp,1], s=100, c='red', marker='o')
    rect_left_eye = patches.Rectangle((12, 14), 12, 8, linewidth=3, 
                         edgecolor='b', facecolor='b',alpha=0.5)
    rect_right_eye = patches.Rectangle((30, 14), 12, 8, linewidth=3, 
                             edgecolor='y', facecolor='y',alpha=0.5)
    rect_nose = patches.Rectangle((20, 25), 9, 8, linewidth=3, 
                             edgecolor='g', facecolor='g',alpha=0.5)
    rect_mouth = patches.Rectangle((18, 35), 12, 11, linewidth=3, 
                             edgecolor='r', facecolor='r',alpha=0.5)
    ax.add_patch(rect_left_eye)
    ax.add_patch(rect_right_eye)
    ax.add_patch(rect_nose)
    ax.add_patch(rect_mouth)
    plt.show()

# emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
#             4: 'Sad', 5: 'Surprise', 6: 'Neutral'}


np.save('valid_image_array', valid_image_array)
np.save('valid_label', valid_label)
# np.save('left_eye', left_eye)
# np.save('right_eye', right_eye)
# np.save('nose', nose)
# np.save('mouth', mouth)
