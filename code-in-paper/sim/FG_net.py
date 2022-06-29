import numpy as np
import pandas as pd

from os import listdir
from os.path import isfile, join

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['figure.figsize'] = (12,12)

# load the dataset
face_images_db = np.load('./dataset/face_images/face_images.npz')['face_images']
facial_keypoints_df = pd.read_csv('./dataset/face_images/facial_keypoints.csv')

(im_height, im_width, num_images) = face_images_db.shape
num_keypoints = facial_keypoints_df.shape[1] / 2

print('number of images = %d' %(num_images))
print('image dimentions = (%d,%d)' %(im_height,im_width))
print('number of facial keypoints = %d' %(num_keypoints))

# show a random subset of images from the dataset
num_fig_rows = 7
num_fig_cols = 5

num_plots = num_fig_rows * num_fig_cols

rand_inds_vec = np.random.choice(face_images_db.shape[2],num_plots,replace=False)
rand_inds_mat = rand_inds_vec.reshape((num_fig_rows,num_fig_cols))

plt.close('all')
fig, ax = plt.subplots(nrows=num_fig_rows,ncols=num_fig_cols,figsize=(14,18))

for i in range(num_fig_rows):
    for j in range(num_fig_cols):
        curr_ind = rand_inds_mat[i][j]
        curr_image = face_images_db[:,:,curr_ind]
    
        x_feature_coords = np.array(facial_keypoints_df.iloc[curr_ind,0::2].tolist())
        y_feature_coords = np.array(facial_keypoints_df.iloc[curr_ind,1::2].tolist())
    
        ax[i][j].imshow(curr_image, cmap='gray')
        ax[i][j].scatter(x_feature_coords,y_feature_coords,c='r',s=15)
        ax[i][j].set_axis_off()
        ax[i][j].set_title('image index = %d' %(curr_ind),fontsize=10)
plt.show()



# show the scatter plot of all locations of several selected keypoints
# create a map of what color you want to display different keypoints
fraction_present_keypoints = 1.0 - facial_keypoints_df.iloc[:,0::2].isnull().mean(axis=0)
keypointColors = {}
keypointColors['left_eye_center'] = 'yellow'
keypointColors['right_eye_center'] = 'green'
keypointColors['nose_tip'] = 'red'
keypointColors['mouth_center_bottom_lip'] = 'blue'

# get all inds of image where the requested keypoints are present
keypoint_present_inds = np.ones(num_images) == 1
for key in keypointColors.keys():
    keypoint_present_inds = keypoint_present_inds & (facial_keypoints_df.isnull()[key+'_x'] == False)
keypoint_present_inds = np.nonzero(keypoint_present_inds.values)[0]

plt.figure(figsize=(10,10))
for key,value in keypointColors.items():
    x_feature_coords = im_width  - np.array(facial_keypoints_df.loc[keypoint_present_inds,key+'_x'].tolist())
    y_feature_coords = im_height - np.array(facial_keypoints_df.loc[keypoint_present_inds,key+'_y'].tolist())
    plt.scatter(x_feature_coords,y_feature_coords,c=value,s=10,alpha=0.5)
plt.xlim(0,im_width); plt.ylim(0,im_height)
plt.title('4 keypoints (eyes, nose, mouth) scatter plot', fontsize=20)
plt.show()