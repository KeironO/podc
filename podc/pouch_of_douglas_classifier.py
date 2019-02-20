'''
Copyright (c) 2019 Keiron O'Shea

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public
License as published by the Free Software Foundation; either
version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public
License along with this program; if not, write to the
Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
Boston, MA 02110-1301 USA
'''

from data import VideoDataGenerator, VideoDataLoader
import os
import numpy as np
import json
from sklearn.model_selection import KFold, train_test_split
from deeplearning import CheapoKeepo
from utils import Inference, visualise_video_data
from random import shuffle

home_dir = os.path.expanduser("~")
slidingsign_dir = os.path.join(home_dir, "Data/podc/slidingsign/")
data_dir = os.path.join(slidingsign_dir, "videos")
results_dir = os.path.join(slidingsign_dir, "results")

with open(os.path.join(slidingsign_dir, "labels.json"), "r") as infile:
    labels = json.load(infile)

video_ids = np.array(list(labels.keys()))

parameter_grid = {
    "data": {
        "height": 64,
        "width": 85,
        "max_frames": 80
    },
    "training": {
        "train_batch_size": 16,
        "val_batch_size": 8,
        "test_batch_size": 8,
        "epochs": 1000,
        "patience": 50
    },
    "model": {
        "vgg16_weights_fp": os.path.join(
            home_dir,
            ".keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
            ),
        "lstm_count": 64,
        "lstm_dropout": 0.2
    }
}

shuffle(video_ids)

vdl = VideoDataLoader(data_dir, labels, video_ids)
X, y = vdl.get_data(
    height=parameter_grid["data"]["height"],
    width=parameter_grid["data"]["width"],
    n_frames=parameter_grid["data"]["max_frames"]
)

y_true = []
y_pred = []

kf = KFold(n_splits=10)

for train_index, test_index in kf.split(video_ids):
    train_index, val_index = train_test_split(train_index, test_size=0.2)

    train_vg = VideoDataGenerator(
        X[train_index], y[train_index],
        height=parameter_grid["data"]["height"],
        width=parameter_grid["data"]["height"],
        max_frames=parameter_grid["data"]["max_frames"],
        batch_size=parameter_grid["training"]["train_batch_size"],
        upsample=True,
        shuffle=True,
        shear_range=0.2,
        rotation_range=20,
        horizontal_flip=True,
        gaussian_blur=True,
        n_jobs=-1
        )

    val_vg = VideoDataGenerator(
        X[val_index],
        y[val_index],
        height=parameter_grid["data"]["height"],
        width=parameter_grid["data"]["height"],
        max_frames=parameter_grid["data"]["max_frames"],
        batch_size=parameter_grid["training"]["val_batch_size"],
        n_jobs=-1
        )

    test_vg = VideoDataGenerator(
        X[test_index], y[test_index],
        height=parameter_grid["data"]["height"],
        width=parameter_grid["data"]["height"],
        max_frames=parameter_grid["data"]["max_frames"],
        batch_size=parameter_grid["training"]["test_batch_size"],
        n_jobs=-1
        )

    clf = CheapoKeepo(
        parameter_grid["data"]["height"],
        parameter_grid["data"]["width"],
        results_dir,
        max_frames=parameter_grid["data"]["max_frames"])

    clf.fit(
        train_vg,
        val_vg,
        epochs=parameter_grid["training"]["epochs"],
        patience=parameter_grid["training"]["patience"]
        )

    ground_truths, model_predictions = clf.predict_pod(test_vg)
    y_true.extend(ground_truths)
    y_pred.extend(model_predictions)
    

print(y_true, y_pred)

inf = Inference(y_true, y_pred)

with open(os.path.join(results_dir + "/10kf_results.json"), "w") as outfile:
    json.dump(str(inf.to_dict()), outfile, indent=4)
