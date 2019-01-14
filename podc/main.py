from data import VideoDataGenerator
from deeplearning import Classifier
from sklearn.model_selection import LeaveOneOut, train_test_split
import numpy as np
from utils import get_labels, get_max_frames
import os
import json
from sklearn.utils import class_weight

home_dir = os.path.expanduser("~")

data_dir = os.path.join(home_dir, "Data/slidingsign")
videos_dir = os.path.join(data_dir, "videos")

with open(os.path.join(data_dir, "labels.json"), "r") as infile:
    labels = json.load(infile)



max_frames = 100
width = 75
height = 75

filenames = np.array(os.listdir(videos_dir))

classes = [x[1] for x in labels.items()]

class_weights = class_weight.compute_class_weight("balanced", np.unique(classes), classes)


y_true = []
y_pred = []

for train, test in LeaveOneOut().split(filenames):
    train, val = train_test_split(train, train_size=0.9, test_size=0.1)
    tra_vdg = VideoDataGenerator(videos_dir, filenames[train], labels, 2, max_frames=max_frames, height=height, width=width, rotation_range=1, shear_range=1, n_jobs=-1)
    val_vdg = VideoDataGenerator(videos_dir, filenames[val], labels, 2, max_frames=max_frames, height=height, width=width, n_jobs=-1)
    tes_vdg = VideoDataGenerator(videos_dir, filenames[test], labels, 8, max_frames=max_frames, height=height, width=width, n_jobs=-1)

    model = Classifier(width=width, height=height, max_frames=max_frames)
    model.train(tra_vdg, val_vdg, "/tmp/model.h5", class_weights, epochs=100, patience=20)
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
    
    
