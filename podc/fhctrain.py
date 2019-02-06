from data import FHCDataGenerator
import pandas as pd
import json
import os
import numpy as np
from utils import VGG16FHC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import *
from keras.models import load_model

home_dir = os.path.expanduser("~")
data_dir = os.path.join(home_dir, "Data/FHC/training_set")
results_dir = os.path.join(home_dir, "Data/FHC/results")

_WIDTH = 128
_HEIGHT = 128

n_classes = 1

ids = pd.read_csv(os.path.join(data_dir, "training.csv"), index_col=0).index.values
ids = np.array([x.split(".")[0] for x in ids])

train, test = train_test_split(ids, train_size=0.8)

val, test = train_test_split(test, train_size=0.5)

clf = VGG16FHC(_HEIGHT, _WIDTH, "/tmp/", n_classes).model

# Data Generators
fhc_train = FHCDataGenerator(data_dir, train, _HEIGHT, _WIDTH, n_classes, zoom_range=.8, horizontal_flip=True, vertical_flip=True, shear_range=0.8, rotation_range=0.8)
fhc_val = FHCDataGenerator(data_dir, val, _HEIGHT, _WIDTH, n_classes)
fhc_test = FHCDataGenerator(data_dir, test, _HEIGHT, _WIDTH, n_classes)

model_fp = os.path.join(results_dir, "best_model.md5")

# Callbacks
es = EarlyStopping(monitor="val_loss", min_delta=0, patience=35, verbose=0, mode="auto", baseline=None, restore_best_weights=False)
mc = ModelCheckpoint(model_fp, monitor="val_loss", verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

# Do the training
history = clf.fit_generator(fhc_train, epochs=10, validation_data=fhc_val, callbacks=[es, mc])

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

# Load best models
clf = load_model(model_fp)

from scipy.ndimage.measurements import center_of_mass

# Cheap Evaluator
count = 0
for X, y_true in fhc_test:
    y_pred = clf.predict(X)
    for indx, y_p in enumerate(y_pred):

        fig, axs = plt.subplots(figsize=[20,5], ncols=4)

        axs[0].imshow(X[indx][:, :, 0])
        axs[0].axis("off")
        axs[0].set_title("Input Data")

        axs[1].imshow(y_true[indx][:, :, 0])
        axs[1].axis("off")
        axs[1].set_title("Segmentation Ground Truth")


        axs[2].imshow(y_p[:, :, 0])
        axs[2].axis("off")
        axs[2].set_title("Segmentation Prediction")

        axs[3].imshow(y_p[:, :, 0] >= np.percentile(y_p[:, :, 0], 80))
        axs[3].axis("off")
        axs[3].set_title("Segmentation Prediction (80%)")

        plt.tight_layout()
        plt.savefig("%s/pred_%i.png" % (results_dir, count))
        plt.clf()
        count += 1
        