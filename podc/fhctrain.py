from data import FHCDataGenerator
import pandas as pd
import os
import numpy as np
from utils import VGG19FHC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import *
from keras.models import load_model

data_dir = "/home/keo7/Data/FHC/training_set"

_WIDTH = 128
_HEIGHT = 128

ids = pd.read_csv(os.path.join(data_dir, "training.csv"), index_col=0).index.values

ids = np.array([x.split(".")[0] for x in ids])

train, test = train_test_split(ids, train_size=0.9)

val, test = train_test_split(test, train_size=0.5)

clf = VGG19FHC(0, _HEIGHT, _WIDTH, "/tmp/").model

# Data Generators
fhc_train = FHCDataGenerator(data_dir, train, _HEIGHT, _WIDTH, zoom_range=.8, horizontal_flip=True, vertical_flip=True, shear_range=0.8, rotation_range=0.8)
fhc_val = FHCDataGenerator(data_dir, val, _HEIGHT, _WIDTH)
fhc_test = FHCDataGenerator(data_dir, test, _HEIGHT, _WIDTH)

# Callbacks
es = EarlyStopping(monitor="val_loss", min_delta=0, patience=20, verbose=0, mode="auto", baseline=None, restore_best_weights=False)
mc = ModelCheckpoint("/tmp/best.md5", monitor="val_loss", verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

# Do the training
clf.fit_generator(fhc_train, epochs=100, validation_data=fhc_val, callbacks=[es, mc])
# Load best models
clf = load_model("/tmp/best.md5")

# Cheap Evaluator
count = 0
for X, y_true in fhc_test:
    y_pred = clf.predict(X)
    for indx, y_p in enumerate(y_pred):
        fig, axs = plt.subplots(figsize=[20,5], ncols=4)

        axs[0].imshow(X[indx][:, :, 1])
        axs[0].axis("off")
        axs[0].set_title("Input Data")

        axs[1].imshow(y_true[indx][:, :, 1])
        axs[1].axis("off")
        axs[1].set_title("Segmentation Ground Truth")

        axs[2].imshow(y_p[:, :, 1])
        axs[2].axis("off")
        axs[2].set_title("Segmentation Prediction")

        axs[3].imshow(y_p[:, :, 1] >= np.percentile(y_p[:, :, 1], 80))
        axs[3].axis("off")
        axs[3].set_title("Segmentation Prediction (80%)")

        plt.tight_layout()
        plt.savefig("/tmp/%i.png" % count)
        plt.clf()
        count += 1
