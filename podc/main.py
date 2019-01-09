from data import VideoDataGenerator
from deeplearning import Model
import os
from sklearn.model_selection import KFold, train_test_split
import numpy as np


names = ["positive", "negative"]
videos_dir = "/home/keo7/Videos"

labels_dict = {}

for n in names:
    fn = "videos of %s sliding sign from reproducibility study" % (n)
    files = os.listdir(os.path.join(videos_dir, fn))
    if n == "negative":
        n = 0
    else:
        n = 1
    for i in files:
        labels_dict[i] = n
    

max_frames = 750
width = 64
height = 64

data_dir = "/home/keo7/Data/slidingsign"
filenames = np.array(os.listdir(data_dir))

tra_vdg = VideoDataGenerator(data_dir, filenames, labels_dict, 8, max_frames=max_frames, height=height, width=width, rotation_range=1, shear_range=1, n_jobs=-1)

import sys
from hurry.filesize import size
for X, y in tra_vdg:
    print(size(sys.getsizeof(X)))
    print([np.mean(x) for x in X])
    exit(0)
exit(0)

for train, test in KFold(n_splits=10).split(filenames):
    train, val = train_test_split(train, train_size=0.8, test_size=0.2)
    tra_vdg = VideoDataGenerator(data_dir, filenames[train], labels_dict, 4, max_frames=max_frames, height=height, width=width, rotation_range=1, shear_range=1, n_jobs=-1)
    val_vdg = VideoDataGenerator(data_dir, filenames[val], labels_dict, len(val), max_frames=max_frames, height=height, width=width, n_jobs=-1)
    model = Model(width=width, height=height, max_frames=max_frames)
    model.train_model(tra_vdg, val_vdg, "/tmp/model.h5", epochs=10)
    break
    
