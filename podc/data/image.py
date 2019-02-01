from keras.utils import Sequence
import os
import numpy as np
from PIL import Image


class FHCDataGenerator(Sequence):
    def __init__(self, data_dir, ids, height, width, output_height, output_width, batch_size=32, shuffle=True):
        self.data_dir = data_dir
        self.ids = ids
        self.height = int(height)
        self.width = int(width)
        self.output_height = int(output_height)
        self.output_width = int(output_width)
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
        
        def _load_image(identifier):
            fp = os.path.join(self.data_dir, identifier + ".png")
            img = Image.open(fp)
            img = img.convert("RGB")
            img = img.resize((self.width, self.height))
            return np.array(img)
        
        def _load_annotations(identifier):

            def __colour_elipsoid(img):
                img = np.array(img)
                img = (img == 255).astype(int)

                return np.array(np.maximum.accumulate(img, 1) & np.maximum.accumulate(img[:, ::-1], 1)[:, ::-1])


            fp = os.path.join(self.data_dir, identifier + "_Annotation.png")
            img = Image.open(fp)
            img = __colour_elipsoid(img)

            img = Image.fromarray(img.astype("uint8"))

            img = img.resize((self.output_width, self.output_height))

            o = np.zeros((2, self.output_width, self.output_height))

            for i in range(2):
                o[i] = (img == i)

            img = np.array(o).reshape(self.output_width*self.output_height, 2)

            return img

        identifiers = self.ids[indexes]

        X = np.zeros((len(identifiers), self.height, self.width, 3))
        y = []

        for indx, identifier in enumerate(identifiers):
            X[indx] =_load_image(identifier)
            y.append(_load_annotations(identifier))

        return X, np.array(y)


        