from data import VideoDataGenerator
from deeplearning import Model
from sklearn.model_selection import KFold, train_test_split
import numpy as np
from utils import get_labels, get_max_frames
import os

videos_dir = "/home/keo7/Videos"
data_dir = "/home/keo7/Data/slidingsign"
labels_dict = get_labels(videos_dir)

# get_max_frames(data_dir)

max_frames = 150
width = 48
height = 48

filenames = np.array(os.listdir(data_dir))

for train, test in KFold(n_splits=10).split(filenames):
    train, val = train_test_split(train, train_size=0.8, test_size=0.2)
    tra_vdg = VideoDataGenerator(data_dir, filenames[train], labels_dict, 8, max_frames=max_frames, height=height, width=width, rotation_range=1, shear_range=1, n_jobs=-1)
    val_vdg = VideoDataGenerator(data_dir, filenames[val], labels_dict, 4, max_frames=max_frames, height=height, width=width, n_jobs=-1)
    model = Model(width=width, height=height, max_frames=max_frames)
    model.train_model(tra_vdg, val_vdg, "/tmp/model.h5", epochs=10)
    break
    
