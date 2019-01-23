import pandas as pd
import imageio
import os
from PIL import Image
import numpy as np

class MIAS():
    def __init__(self, image_dir, labels_fp, image_format, height=224, width=224, num_channels=1):
        self.image_dir = image_dir
        self.labels_fp = labels_fp
        self.image_format = image_format
        self.height = height
        self.width = width
        self.num_channels = num_channels

    def load_data(self, c):
        labels = self._get_labels(self.labels_fp)
        labels = labels[c]
        return self._get(labels)

    def _get(self, labels):

        X = np.empty((len(labels.index.values), self.height, self.width, self.num_channels))
        y = []
        for index, (filename, values) in enumerate(labels.iterrows()):
            l = values.values
            if len(l) == 1:
                l = l[0]
            
            fp = os.path.join(self.image_dir, filename + self.image_format)
            img = imageio.imread(fp)
            img = img.view(type=np.ndarray)
            img = Image.fromarray(img)
            img = img.resize((self.height, self.width))
            img = np.array(img)
            img = img.reshape(img.shape[0], img.shape[1], self.num_channels)

            X[index:, ] = img

            y.append(l)

        y = np.array(y)
            
        return X, y

    def _get_labels(self, fp):
        df = pd.read_csv(fp, index_col=0)
        return df


if __name__ == "__main__":
    mias = MIAS("/home/keo7/Data/MIAS/images", "/home/keo7/Data/MIAS/Info.txt", ".jpg")
    X, y = mias.load_data(["class"])

    print(X.shape)

#get_metadata("/home/kto/Data/MIAS/Info.txt")