from data import FHCDataGenerator
import pandas as pd
import os
import numpy as np

data_dir = "/home/keo7/Data/FHC/training_set"

_WIDTH = 800
_HEIGHT = 540

scale = 0.25

ids = pd.read_csv(os.path.join(data_dir, "training.csv"), index_col=0).index.values

ids = np.array([x.split(".")[0] for x in ids])

fhc = FHCDataGenerator(data_dir, ids, _HEIGHT*scale, _WIDTH*scale)

for X, y in fhc:
    
    break