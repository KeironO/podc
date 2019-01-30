from data import VideoDataGenerator
from deeplearning import Classifier
from sklearn.model_selection import LeaveOneOut, KFold, train_test_split
import numpy as np
from utils import get_labels, get_max_frames, VGG19v1, SmolNet, VGG16v1
import os
import json
from collections import Counter
from sklearn.utils import class_weight
from keras.models import load_model
from sklearn.metrics import confusion_matrix


home_dir = os.path.expanduser("~")

data_dir = os.path.join(home_dir, "Data/slidingsign")
videos_dir = os.path.join(data_dir, "videos")

with open(os.path.join(data_dir, "labels.json"), "r") as infile:
    labels = json.load(infile)

def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return  {cls: float(majority/count) for cls, count in counter.items()}

max_frames = 100
width = 64
height = 64

filenames = np.array(os.listdir(videos_dir))

classes = [x[1] for x in labels.items()]

class_weights = get_class_weights(classes)

y_true = []
y_pred = []



train, test = train_test_split(filenames, train_size=0.8, test_size=0.2)
val, test = train_test_split(test, train_size=0.5, test_size=0.5)

tra_vdg = VideoDataGenerator(videos_dir, train, labels, 2, max_frames=max_frames, height=height, width=width, rotation_range=1, shear_range=1, n_jobs=-1, upsample=True)
val_vdg = VideoDataGenerator(videos_dir, val, labels, 2, max_frames=max_frames, height=height, width=width, n_jobs=-1)
tes_vdg = VideoDataGenerator(videos_dir, test, labels, 1, max_frames=max_frames, height=height, width=width, n_jobs=-1)

clf = VGG19v1(max_frames, width, height, output_dir="/tmp/").model

#clf = load_model("/tmp/model.h5")

model = Classifier(width=width, height=height, max_frames=max_frames, clf=clf)
model.train(tra_vdg, val_vdg, "/tmp/model.h5", epochs=100, patience=20)

#model = load_model("/tmp/model.h5")
y_true, y_pred = model.predict(tra_vdg) 

X, y = fhc.load_data()

idg = ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2
)

for x, _y in idg.flow(X, y, batch_size=32):
    print(x)
    break

y_true = [item for sublist in y_true for item in sublist]
y_pred = [int(item) for sublist in y_pred for item in sublist]

out_json_tmp = {
    "y_true" : y_true,
    "y_pred" : y_pred
}

print(out_json_tmp)

print(confusion_matrix(out_json_tmp["y_true"], out_json_tmp["y_pred"]))

with open("results.json", "w") as outfile:
    json.dump(out_json_tmp, outfile, indent=4)
    
    
