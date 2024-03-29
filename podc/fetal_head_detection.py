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

from data import FHCDataGenerator
from scipy.ndimage.measurements import center_of_mass
import pandas as pd
import json
import os
import numpy as np
from deeplearning import VGG16FHC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

home_dir = os.path.expanduser("~")
data_dir = os.path.join(home_dir, "Data/FHC/training_set")
results_dir = os.path.join(home_dir, "Data/FHC/results")

_WIDTH = 224
_HEIGHT = 224

n_classes = 1

ids = pd.read_csv(
    os.path.join(data_dir, "training.csv"), index_col=0).index.values
ids = np.array([x.split(".")[0] for x in ids])

train, test = train_test_split(ids, train_size=0.8)

val, test = train_test_split(test, train_size=0.5)

clf = VGG16FHC(_HEIGHT, _WIDTH, "/tmp/", n_classes).model

fill_elipsoid = True

# Data Generators
fhc_train = FHCDataGenerator(
    data_dir,
    train,
    _HEIGHT,
    _WIDTH,
    n_classes,
    zoom_range=.8,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.8,
    rotation_range=0.8)
fhc_val = FHCDataGenerator(data_dir, val, _HEIGHT, _WIDTH, n_classes)
fhc_test = FHCDataGenerator(data_dir, test, _HEIGHT, _WIDTH, n_classes)


model_fp = os.path.join(results_dir, "best_model.md5")

# Callbacks
es = EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=35,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False
    )

mc = ModelCheckpoint(
    model_fp,
    monitor="val_loss",
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    period=1
    )

# Do the training
history = clf.fit_generator(
    fhc_train, epochs=1000, validation_data=fhc_val, callbacks=[es, mc]
    )


history = history.history

history_fp = os.path.join(results_dir, "history.json")
history_lossplot_fp = os.path.join(results_dir, "history_plot.png")

with open(history_fp, "w") as outfile:
    json.dump(history, outfile, indent=4)

plt.figure()
plt.plot(history["loss"], label="train loss")
plt.plot(history["val_loss"], label="validation loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xlim([0, len(history["loss"])])
plt.tight_layout()
plt.savefig(history_lossplot_fp)
plt.clf()

# Load best model
clf = load_model(model_fp)

# Cheap Evaluator
count = 0
for X, y_true in fhc_test:
    y_pred = clf.predict(X)
    for indx, y_p in enumerate(y_pred):

        fig, axs = plt.subplots(figsize=[15, 5], ncols=3)

        axs[0].imshow(X[indx][:, :, 0])
        axs[0].axis("off")
        axs[0].set_title("Input Data")

        axs[1].imshow(y_true[indx][:, :, 0])
        axs[1].axis("off")
        axs[1].set_title("Segmentation Ground Truth")

        axs[2].imshow(y_p[:, :, 0])
        axs[2].axis("off")
        axs[2].set_title("Segmentation Prediction")

        plt.tight_layout()
        plt.savefig("%s/pred_%i.png" % (results_dir, count))
        plt.clf()
        count += 1
