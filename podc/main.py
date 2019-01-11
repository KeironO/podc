from data import VideoDataGenerator
from deeplearning import Classifier
from sklearn.model_selection import LeaveOneOut, train_test_split
import numpy as np
from utils import get_labels, get_max_frames
import os
import json

home_dir = os.path.expanduser("~")

data_dir = os.path.join(home_dir, "Data/slidingsign")
videos_dir = os.path.join(data_dir, "videos")

with open(os.path.join(data_dir, "labels.json"), "r") as infile:
    labels = json.load(infile)

max_frames = 100
width = 128
height = 128

filenames = np.array(os.listdir(videos_dir))

y_true = []
y_pred = []

from matplotlib.animation import ArtistAnimation
import matplotlib.pyplot as plt

filenames = filenames[0:4]

dg = VideoDataGenerator(videos_dir, filenames, labels, 2, max_frames=max_frames, height=height, width=width, rotation_range=1, shear_range=1, n_jobs=-1)

for l in range(10):
    for X, y in dg:
        for i in range(X.shape[0]):
            d = X[i]
            frames = []
            fig = plt.figure()
            for j in range(d.shape[0]):
                frames.append([plt.imshow(d[j])])
            ani = ArtistAnimation(fig, frames, interval=50)
            ani.save("/home/keo7/Pictures/test/lot%i-sample%i.mp4" % (l, i), fps=30)
            plt.close()


exit(0)

for train, test in LeaveOneOut().split(filenames):
    train, val = train_test_split(train, train_size=0.8, test_size=0.2)
    tra_vdg = VideoDataGenerator(videos_dir, filenames[train], labels, 2, max_frames=max_frames, height=height, width=width, rotation_range=1, shear_range=1, n_jobs=-1)
    val_vdg = VideoDataGenerator(videos_dir, filenames[val], labels, 1, max_frames=max_frames, height=height, width=width, n_jobs=-1)
    tes_vdg = VideoDataGenerator(videos_dir, filenames[test], labels, 1, max_frames=max_frames, height=height, width=width, n_jobs=-1)

    model = Classifier(width=width, height=height, max_frames=max_frames)
    model.train(tra_vdg, val_vdg, "/tmp/model.h5", epochs=100, patience=15)
    t, p = model.predict(tes_vdg)
    y_true.extend(t.tolist())
    y_pred.extend(p)
    

y_true = [item for sublist in y_true for item in sublist]
y_pred = [int(item) for sublist in y_pred for item in sublist]

out_json_tmp = {
    "y_true" : y_true,
    "y_pred" : y_pred
}

with open("results.json", "w") as outfile:
    json.dump(out_json_tmp, outfile, indent=4)
    
    
