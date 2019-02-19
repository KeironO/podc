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
from deeplearning import VGG19v1
from utils import Inference, visualise_video_data

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
        "width": 64,
        "max_frames": 20
    },
    "training": {
        "train_batch_size": 8,
        "val_batch_size": 2,
        "test_batch_size": 1,
        "epochs": 1000,
        "patience": 50
    },
    "model": {
        "hid_states": {
            "filter": 128,
            "kernel_size": (4, 4),
            "recurrent_dropout": 0.5,
            "dropout": 0.5
        },
        "conv_hid_states": {
            "filter": 64,
            "kernel_size": (4, 4)
        },
        "conv_acts": {
            "filter": 64,
            "kernel_size": (4, 4)
        },
        "eunice": {
            "filter": 1,
            "kernel_size": (4, 4)
        },
        "nn": {
            "filter": 128,
            "kernel_size": (4, 4),
            "recurrent_dropout": 0.5,
            "dropout": 0.5
        },
        "opt": {
            "lr": 0.01,
            "beta_1": 0.9,
            "beta_2": 0.9,
            "decay": 0.0
        }
    }
}

from random import shuffle


shuffle(video_ids)

vdl = VideoDataLoader(data_dir, labels, video_ids[0:16])
X, y = vdl.get_data(
    parameter_grid["data"]["height"],
    parameter_grid["data"]["width"],
    parameter_grid["data"]["max_frames"]
)

train_vg = VideoDataGenerator(X, y,
        batch_size=parameter_grid["training"]["train_batch_size"],
        upsample=True,
        shuffle=True,
        shear_range=0.2,
        rotation_range=0.2,
        vertical_flip=True,
        n_jobs=-1
        )


for X, y in train_vg:
    for i in range(X.shape[0]):
        visualise_video_data(X[i])
        break
    break
exit(0)

for X,y in train_vg:
    print(X.shape)
    exit(0)
y_true = []
y_pred = []

kf = KFold(n_splits=10)

for train_index, test_index in kf.split(video_ids):
    train_index, val_index = train_test_split(train_index, test_size=0.2)

    train_vg = VideoDataGenerator(
        data_dir,
        video_ids[train_index],
        labels,
        height=parameter_grid["data"]["height"],
        width=parameter_grid["data"]["height"],
        max_frames=parameter_grid["data"]["max_frames"],
        batch_size=parameter_grid["training"]["train_batch_size"],
        upsample=True,
        shuffle=True,
        shear_range=0.2,
        rotation_range=0.2,
        vertical_flip=True,
        n_jobs=-1)

    val_vg = VideoDataGenerator(
        data_dir,
        video_ids[val_index],
        labels,
        height=parameter_grid["data"]["height"],
        width=parameter_grid["data"]["height"],
        max_frames=parameter_grid["data"]["max_frames"],
        batch_size=parameter_grid["training"]["val_batch_size"],
        n_jobs=-1)

    test_vg = VideoDataGenerator(
        data_dir,
        video_ids[test_index],
        labels,
        height=parameter_grid["data"]["height"],
        width=parameter_grid["data"]["height"],
        max_frames=parameter_grid["data"]["max_frames"],
        batch_size=parameter_grid["training"]["test_batch_size"],
        n_jobs=-1)

    clf = VGG19v1(
        parameter_grid["data"]["height"],
        parameter_grid["data"]["width"],
        results_dir,
        max_frames=parameter_grid["data"]["max_frames"],
        model_parameters=parameter_grid["model"]
        )

    clf.fit(
        train_vg,
        val_vg,
        epochs=parameter_grid["training"]["epochs"],
        patience=parameter_grid["training"]["patience"])

    ground_truths, model_predictions = clf.predict_pod(test_vg)
    y_true.extend(ground_truths)
    y_pred.extend(model_predictions)
    break

inf = Inference(y_true, y_pred)

with open(results_dir + "/10kf_results.json", "w") as outfile:
    json.dump(inf.to_dict(), outfile, indent=4)
