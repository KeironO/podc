import imageio
from PIL import Image
import pandas as pd
import os
from keras.utils import Sequence
from keras_preprocessing.image import apply_affine_transform
import numpy as np
from keras import backend as K

ROW_AXIS = 0
COL_AXIS = 1
CHANNEL_AXIS = 2

class VideoDataGenerator(Sequence):
    def __init__(self, directory, filenames, labels, batch_size, shuffle=False, height=None, width=None, featurewise_center=False, rotation_range=False, brightness_range=False, shear_range=False, zoom_range=False, horizontal_flip=False, vertical_flip=False):
        self.directory = directory
        self.filenames = filenames
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.height = height
        self.width = width
        self.featurewise_center = featurewise_center
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

        self._generate_filepaths()
        self.on_epoch_end()

    def _generate_filepaths(self):
        self.filepaths = [os.path.join(self.directory, x) for x in self.filenames]


    def __len__(self):
        return int(np.floor(len(self.filepaths)/ self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.filepaths))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:self.batch_size*(index+1)]
        filepaths = [self.filepaths[indx] for indx in indexes]
        X, y = self.__data_generation(filepaths)
        return X, y     

    def __data_generation(self, filepaths):

        def ___random_rotation(x, rg, fill_mode="nearest", cval=0., interpolation_order=1):
            theta = np.random.uniform(-rg, rg)
            for index in range(x.shape[0]):
                x[index] = apply_affine_transform(x[index], theta=theta, row_axis=ROW_AXIS, col_axis=COL_AXIS, channel_axis=CHANNEL_AXIS, fill_mode=fill_mode, cval=cval)
            return x

        def __featurewise_centre(x):
            mean = np.mean(x, axis=(ROW_AXIS, COL_AXIS))
            x -= mean
            return x

        def __flip_axis(x, axis):
            x = np.asarray(x).swapaxes(axis, 0)
            x = x[::-1, ...]
            x = x.swapaxes(0, axis)
            return x
            
        def __random_shift(x, wrg, hrg, fill_mode="nearest", cval=0.):

            tx = np.random.uniform(-hrg, hrg) * x.shape[ROW_AXIS]
            ty = np.random.uniform(-wrg, wrg) * x.shape[COL_AXIS]

            for index in range(x.shape[0]):
                x[index] = apply_affine_transform(x[index], tx=tx, ty=ty, row_axis=ROW_AXIS, col_axis=COL_AXIS, channel_axis=CHANNEL_AXIS, fill_mode=fill_mode, cval=cval)
            return x

        def __random_zoom(x, zoom_range, fill_mode="nearest", cval=0.):
            zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
            for index in range(x.shape[0]):
                x[index] = apply_affine_transform(x[index], zx=zx, zy=zy, row_axis=ROW_AXIS, col_axis=COL_AXIS, channel_axis=CHANNEL_AXIS, fill_mode=fill_mode, cval=cval)
            return x

        def __random_shear(x, intensity, fill_mode="nearest", cval=0.):
            shear = np.random.uniform(-intensity, intensity)
            for index in range(x.shape[0]):
                x[index] = apply_affine_transform(x[index], shear=shear, row_axis=ROW_AXIS, col_axis=COL_AXIS, channel_axis=CHANNEL_AXIS, fill_mode=fill_mode, cval=cval)
            return x              

        def ___read(filepath):
            video = imageio.get_reader(filepath, "ffmpeg")
            meta = video.get_meta_data()
            
            if None not in [self.width, self.height]:
                frames = np.zeros((meta["nframes"], self.width, self.height, 1))
            else:
                frames = np.zeros((meta["nframes"], meta["size"][0], meta["size"][1], 1))
            for index, frame in enumerate(video):
                frame = frame.view(type=np.ndarray)
                frame = Image.fromarray(frame)
                if None not in [self.width, self.height]:
                    frame = frame.resize((self.width, self.height))
                frame = frame.convert("L")
                frame = np.array(frame)
                frame = frame.reshape(frame.shape[0], frame.shape[1], 1)
                frames[index] = frame
            return frames
        
        X = []
        y = []

        for filepath in filepaths:
            x = ___read(filepath)

            if self.featurewise_center != False:
                x = __featurewise_centre(x)
            
            if self.rotation_range != False:
                x = ___random_rotation(x, self.rotation_range)
            
            if self.horizontal_flip != False:
                x = __flip_axis(x, ROW_AXIS)
            
            if self.vertical_flip != False:
                x = __flip_axis(x, COL_AXIS)

            if self.shear_range != False:
                x = __random_shear(x, self.shear_range)

            if self.brightness_range != False:
                pass

            if self.zoom_range != False:
                x = __random_zoom(x, self.zoom_range)
            
            X.append(x)
            y.append(self.labels["".join(os.path.splitext(os.path.basename(filepath)))])
        return X, y


import matplotlib.pyplot as plt
import matplotlib.animation as animation
if __name__ == "__main__":
    names = ["positive", "negative"]

    videos_dir = "/home/keo7/Videos/"

    labels_dict = {}
    for n in names:
        fn = "videos of %s sliding sign from reproducibility study" % (n)
        files = os.listdir(os.path.join(videos_dir, fn))
        if n == "negative":
            n = 0
        else:
            n = 1
        
        for i in files:
            labels_dict[i] = n    


    data_dir = "/home/keo7/Data/slidingsign"
    filenames = os.listdir(data_dir)
    vdg = VideoDataGenerator(data_dir, filenames, labels_dict, 8, height=128, width=128, featurewise_center=True, rotation_range=1, horizontal_flip=True, shear_range=5)
    
    for X, y in vdg:
        print(len(y))
        for x in X:
            fig = plt.figure()
            frames = []
            for indx in range(x.shape[0]):
                frames.append([plt.imshow(x[indx].reshape(x.shape[1], x.shape[2]))])
            ani = animation.ArtistAnimation(fig, frames, interval=20, blit=True, repeat=False)
            plt.tight_layout()
            plt.show()
            plt.close("all")
            exit(0)