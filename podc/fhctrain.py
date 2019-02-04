from data import FHCDataGenerator
import pandas as pd
import os
import numpy as np
from utils import VGG19FHC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data_dir = "/home/keo7/Data/FHC/training_set"

_WIDTH = 224
_HEIGHT = 224

ids = pd.read_csv(os.path.join(data_dir, "training.csv"), index_col=0).index.values

ids = np.array([x.split(".")[0] for x in ids])

train, test = train_test_split(ids, train_size=0.8)

val, test = train_test_split(test, train_size=0.5)

clf = VGG19FHC(0, _HEIGHT, _WIDTH, "/tmp/").model

fhc_train = FHCDataGenerator(data_dir, train, _HEIGHT, _WIDTH, _HEIGHT, _WIDTH, rotation_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, vertical_flip=True)
fhc_val = FHCDataGenerator(data_dir, val, _HEIGHT, _WIDTH, _HEIGHT, _WIDTH)
fhc_test = FHCDataGenerator(data_dir, test, _HEIGHT, _WIDTH, _HEIGHT, _WIDTH)

clf.fit_generator(fhc_train, epochs=100, validation_data=fhc_val)

for X, y_true in fhc_test:
    y_pred = clf.predict(X)[0]

    fig, axs = plt.subplots(figsize=[15,8], ncols=3)
    axs[0].imshow(y_pred[:, :, 0])
    axs[1].imshow(X[0][:, :, 0])
    axs[2].imshow(y_true[0][:, :, 0])
    plt.tight_layout()
    plt.show()

    
    break
