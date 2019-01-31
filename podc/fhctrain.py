from data import FHCDataGenerator
import pandas as pd
import os
import numpy as np
from utils import VGG19FHC


data_dir = "/home/keo7/Data/FHC/training_set"

_WIDTH = 800
_HEIGHT = 540

scale = 0.25

ids = pd.read_csv(os.path.join(data_dir, "training.csv"), index_col=0).index.values

ids = np.array([x.split(".")[0] for x in ids])

fhc = FHCDataGenerator(data_dir, ids, _HEIGHT*scale, _WIDTH*scale, _HEIGHT*scale, _WIDTH*scale)

for X, y in fhc:
    break

#clf = VGG19FHC(0, int(_HEIGHT*scale), int(_WIDTH*scale), "/tmp/").model


#clf.fit_generator(fhc, epochs=10)
'''
import random

colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(2)  ]

import matplotlib.pyplot as plt

for X, _ in fhc:
    y_pred = clf.predict(X)[0]

    y_pred = y_pred.reshape(clf.outputHeight, clf.outputWidth, 2).argmax( axis = 2)
    seg_img = np.zeros((clf.outputHeight, clf.outputWidth, 3))

    for c in range(2):
        seg_img[:,:,0] += ( (y_pred[:,: ] == c )*( colors[c][0] )).astype('uint8')
        seg_img[:,:,1] += ((y_pred[:,: ] == c )*( colors[c][1] )).astype('uint8')
        seg_img[:,:,2] += ((y_pred[:,: ] == c )*( colors[c][2] )).astype('uint8')

    plt.figure()
    plt.imshow(seg_img[:,:,0])
    plt.show()

    break
'''