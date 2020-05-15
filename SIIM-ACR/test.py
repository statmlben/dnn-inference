import matplotlib.pyplot as plt
import matplotlib.patches as patches

from collections import defaultdict, deque
import datetime
import pickle
import time
import torch.distributed as dist
import errno

import collections
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageFile
import pandas as pd
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

def rle2mask(rle, width, height):
	mask= np.zeros(width* height)
	array = np.asarray([int(x) for x in rle.split()])
	starts = array[0::2]
	lengths = array[1::2]

	current_position = 0
	for index, start in enumerate(starts):
		current_position += start
		mask[current_position:current_position+lengths[index]] = 1
		current_position += lengths[index]

	return mask.reshape(width, height)

sub = pd.read_csv("~/data/SIIM-ACR/train-rle-sample.csv")
sub[sub.EncodedPixels!=' -1']

idx=5
img_path = sub.iloc[idx].ImageId
im = Image.open('~/data/SIIM-ACR/' + img_path + '.dcm').convert("RGB")
im = Image.open(img_path + '.png').convert("RGB")

fig,ax = plt.subplots(1)
ax.imshow(im)
ax.imshow(rle2mask(sub.iloc[idx].EncodedPixels, 1024, 1024).T, alpha=0.5)
plt.show()


idx=1322
img_path = sub.iloc[idx].ImageId
im = Image.open('../input/siim-png-images/input/test_png/' + img_path + '.png').convert("RGB")
fig,ax = plt.subplots(1)
ax.imshow(im)
ax.imshow(rle2mask(sub.iloc[idx].EncodedPixels, 1024, 1024).T, alpha=0.5)
plt.show()