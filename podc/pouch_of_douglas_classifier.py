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

home_dir = os.path.expanduser("~")
slidingsign_dir = os.path.join(home_dir, "Data/podc/slidingsign/")
data_dir = os.path.join(slidingsign_dir, "videos")
results_dir = os.path.join(slidingsign_dir, "results")

_WIDTH = 224
_HEIGHT = 224

with open(os.path.join(slidingsign_dir, "labels.json"), "r") as infile:
    labels = json.load(infile)

print(labels)

del labels["20_0001.AVI"]
del labels["20_0000.AVI"]

video_ids = np.array(list(labels.keys()))

kf = KFold(n_splits=10)

from collections import Counter

for train_index, test_index in kf.split(video_ids):
    train_index, val_index = train_test_split(train_index, test_size=0.2)

    tvg = VideoDataGenerator(
        data_dir,
        video_ids,
        labels,
        height=224,
        width=224,
        batch_size=32,
        upsample=True,
        n_jobs=-1
        )

    for X, y in tvg:
        print(Counter(y))
