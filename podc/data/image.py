from keras.utils import Sequence
import os
import numpy as np
from PIL import Image


class FHCDataGenerator(Sequence):
    def __init__(self, data_dir, ids, height, width, batch_size=32, shuffle=True):
        self.data_dir = data_dir
        self.ids = ids
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.on_epoch_end()


    def _join_paths(self, fns):
        return np.array([os.path.join(self.data_dir, x) for x in fns])
    
    def __len__(self):
        return int(np.floor(len(self.ids)/self.batch_size))
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, index):
        bs = self.batch_size
        indexes = self.indexes[index*bs : bs*(index+1)]
        return self._generate_data(indexes)

    def _generate_data(self, indexes):

        identifiers = self.ids[indexes]
        
        print(identifiers)
        