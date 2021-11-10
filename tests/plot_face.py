import numpy as np
import pandas as pd
import seaborn as sns

image_array_labeled = np.load('train_image_array_labeled.npy')
left_eye = np.load('train_left_eye.npy')
right_eye = np.load('train_right_eye.npy')
nose = np.load('train_nose.npy')
mouth = np.load('train_mouth.npy')

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

import matplotlib.pyplot as plt
sns.set()
sns.relplot(data=df, x="x", y="y", 
			col='keypoints', 
			hue="keypoints", kind="scatter", alpha=.7)
plt.show()

for i in range(10):
	plt.figure()
	plt.imshow(image_array_labeled[i])
	plt.scatter(left_eye[i,0], left_eye[i,1], s=500, c='red', marker='o')
	plt.scatter(right_eye[i,0], right_eye[i,1], s=500, c='red', marker='o')
	plt.scatter(mouth[i,0], mouth[i,1], s=500, c='yellow', marker='o')
	plt.scatter(nose[i,0], nose[i,1], s=500, c='blue', marker='o')
	plt.show()