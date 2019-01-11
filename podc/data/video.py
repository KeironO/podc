import imageio
from PIL import Image
import pandas as pd
import os
import random
import sys
from keras.utils import Sequence
from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.image import apply_affine_transform
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from keras import backend as K
from joblib import Parallel, delayed, cpu_count
from hurry.filesize import size

# Logging details
import logging
#logging.basicConfig(format='%(asctime)s - %(message)s', stream=sys.stdout)


ROW_AXIS = 0
COL_AXIS = 1
CHANNEL_AXIS = 2

class VideoDataGenerator(Sequence):
    def __init__(self, directory, filenames, labels, batch_size, shuffle=False, height=224, width=224, max_frames=50, featurewise_center=False, gaussian_blur=False, rotation_range=False, brightness_range=False, shear_range=False, zoom_range=False, horizontal_flip=False, vertical_flip=False, n_jobs=1):
        self.directory = directory
        self.filenames = filenames
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.height = height
        self.width = width
        self.max_frames = max_frames
        self.featurewise_center = featurewise_center
        self.gaussian_blur = gaussian_blur
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.n_jobs = n_jobs
        self._generate_filepaths()
        self.on_epoch_end()

    def _generate_filepaths(self):
        self.filepaths = [os.path.join(self.directory, x) for x in self.filenames]


    def __len__(self):
        return int(np.floor(len(self.filepaths)/self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.filepaths))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indxs = self.indexes[index*self.batch_size:self.batch_size*(index+1)]

        filepaths = np.array([self.filepaths[k] for k in indxs])
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

        def __gaussian_blur(x):
            rg = random.randint(0, self.gaussian_blur)
            for index in range(x.shape[0]):
                x[index] = gaussian_filter(x[index], rg)
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
            
            frames = np.empty((self.max_frames, self.width, self.height, 3), dtype="uint8")
            for index, frame in enumerate(video):
                if index >= self.max_frames:
                    break
                frame = frame.view(type=np.ndarray)
                frame = Image.fromarray(frame)
                if None not in [self.width, self.height]:
                    frame = frame.resize((self.width, self.height))
                frame = np.array(frame)
                frame = frame.reshape(frame.shape[0], frame.shape[1], 3)

                frames[index] = frame
            frame_size = size(sys.getsizeof(frames))
            logging.info("DATALOAD !! LOADED %s SUCCESS (%s)" % (filepath, frame_size))
            return frames
        
        X = np.zeros((len(filepaths), self.max_frames, self.width, self.height, 3), dtype="uint8")
        y = []

        def _do(filepath):
            x = ___read(filepath)

            if self.featurewise_center != False:
                logging.info("DATAUG !! FEATUREWISE CENTERING %s" % (filepath))
                x = __featurewise_centre(x)
            
            if self.gaussian_blur != False:
                logging.info("DATAUG !! GAUSSIAN BLUR %s" % (filepath))
                x = __gaussian_blur(x)

            if self.rotation_range != False:
                logging.info("DATAUG !! RANDOM ROTATION %s" % (filepath))
                x = ___random_rotation(x, self.rotation_range)
            
            if self.horizontal_flip != False:
                logging.info("DATAUG !! HORIZONTAL FLIP %s" % (filepath))
                # coinflip
                if random.randint(0, 1) == 1:
                    x = __flip_axis(x, ROW_AXIS)
            
            if self.vertical_flip != False:
                logging.info("DATAUG !! VERTICAL FLIP %s" % (filepath))
                if random.randint(0, 1) == 1:
                    x = __flip_axis(x, COL_AXIS)

            if self.shear_range != False:
                logging.info("DATAUG !! SHEAR %s" % (filepath))
                x = __random_shear(x, self.shear_range)

            if self.brightness_range != False:
                pass

            if self.zoom_range != False:
                logging.info("DATAUG !! HORIZONTAL FLIP %s" % (filepath))
                x = __random_zoom(x, self.zoom_range)
            
            logging.info("RETURNING !! DATA %s (%s)" % (filepath, size(sys.getsizeof(x))))

            return x, self.labels["".join(os.path.splitext(os.path.basename(filepath)))]


    
        if self.n_jobs == -1:
            self.n_jobs = cpu_count()
        if self.n_jobs > 1:
            data = Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(_do)(fp) for fp in filepaths)
        else:
            data = [_do(fp) for fp in filepaths]

        for indx in range(len(data)):
            x, _y = data[indx]
            X[indx] = x
            y.append(_y)

        logging.info("RETURNING !! X RETURNED IN SIZE %s" % (size(sys.getsizeof(X))))
        return X, y