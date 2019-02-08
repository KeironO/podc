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

from data import VideoDataGenerator
import os
import numpy as np
import json
from sklearn.model_selection import KFold, train_test_split
from collections import Counter
from utils import VGG16v1
from deeplearning import Classifier

home_dir = os.path.expanduser("~")
slidingsign_dir = os.path.join(home_dir, "Data/podc/slidingsign/")
data_dir = os.path.join(slidingsign_dir, "videos")
results_dir = os.path.join(slidingsign_dir, "results")

_WIDTH = 224
_HEIGHT = 224

with open(os.path.join(slidingsign_dir, "labels.json"), "r") as infile:
    labels = json.load(infile)

del labels["20_0001.AVI"]
del labels["20_0000.AVI"]

video_ids = np.array(list(labels.keys()))

kf = KFold(n_splits=10)

for train_index, test_index in kf.split(video_ids):
    train_index, val_index = train_test_split(train_index, test_size=0.2)

    train_vg = VideoDataGenerator(
        data_dir,
        video_ids[train_index],
        labels,
        height=_HEIGHT,
        width=_WIDTH,
        batch_size=32,
        upsample=True,
        shuffle=True,
        n_jobs=-1
        )

    val_vg = VideoDataGenerator(
        data_dir,
        video_ids[val_index],
        labels,
        height=_HEIGHT,
        width=_WIDTH,
        batch_size=int(len(val_index)/4),
        n_jobs=-1
    )

    test_vg = VideoDataGenerator(
        data_dir,
        video_ids[test_index],
        labels,
        height=_HEIGHT,
        width=_WIDTH,
        batch_size=int(len(test_index)/4),
        n_jobs=-1
    )

    model = VGG16v1(_HEIGHT, _WIDTH, results_dir, max_frames=100)

    clf = Classifier(_HEIGHT, _WIDTH, 100, model)

    clf.fit(train_vg, val_vg, )
