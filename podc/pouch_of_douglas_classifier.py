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
from deeplearning import VGG16v1

home_dir = os.path.expanduser("~")
slidingsign_dir = os.path.join(home_dir, "Data/podc/slidingsign/")
data_dir = os.path.join(slidingsign_dir, "videos")
results_dir = os.path.join(slidingsign_dir, "results")

_WIDTH = 64
_HEIGHT = 64
_MAX_FRAMES = 20

with open(os.path.join(slidingsign_dir, "labels.json"), "r") as infile:
    labels = json.load(infile)

video_ids = np.array(list(labels.keys()))

kf = KFold(n_splits=5)

for train_index, test_index in kf.split(video_ids):
    train_index, val_index = train_test_split(train_index, test_size=0.2)

    train_vg = VideoDataGenerator(
        data_dir,
        video_ids[train_index],
        labels,
        height=_HEIGHT,
        width=_WIDTH,
        max_frames=_MAX_FRAMES,
        batch_size=8,
        upsample=True,
        shuffle=True,
        n_jobs=-1)

    val_vg = VideoDataGenerator(
        data_dir,
        video_ids[val_index],
        labels,
        height=_HEIGHT,
        width=_WIDTH,
        max_frames=_MAX_FRAMES,
        batch_size=8,
        n_jobs=-1)

    test_vg = VideoDataGenerator(
        data_dir,
        video_ids[test_index],
        labels,
        height=_HEIGHT,
        width=_WIDTH,
        max_frames=_MAX_FRAMES,
        batch_size=2,
        n_jobs=-1)

    clf = VGG16v1(
        _HEIGHT, _WIDTH, results_dir, n_classes=1, max_frames=_MAX_FRAMES)

    clf.fit(train_vg, val_vg, epochs=1000, patience=50)
    pred = clf.predict_pod(test_vg)
    print(pred)
    exit(0)
    break
