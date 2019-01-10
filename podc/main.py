from data import VideoDataGenerator
from deeplearning import Classifier
from sklearn.model_selection import KFold, train_test_split
import numpy as np
from utils import get_labels, get_max_frames
import os
import json

home_dir = os.path.expanduser("~")

data_dir = os.path.join(home_dir, "Data/slidingsign")
videos_dir = os.path.join(data_dir, "videos")

with open(os.path.join(data_dir, "labels.json"), "r") as infile:
    labels = json.load(infile)

max_frames = 150
width = 200
height = 200

filenames = np.array(os.listdir(data_dir))


for train, test in KFold(n_splits=10).split(filenames):
    train, val = train_test_split(train, train_size=0.8, test_size=0.2)
    tra_vdg = VideoDataGenerator(data_dir, filenames[train], labels_dict, 2, max_frames=max_frames, height=height, width=width, rotation_range=1, shear_range=1, n_jobs=-1)
    val_vdg = VideoDataGenerator(data_dir, filenames[val], labels_dict, 1, max_frames=max_frames, height=height, width=width, n_jobs=-1)
    model = Classifier(width=width, height=height, max_frames=max_frames)
    model.train(tra_vdg, val_vdg, "/tmp/model.h5", epochs=10)
    
    
