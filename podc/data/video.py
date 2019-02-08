'''
Copyright (c) 2019 Keiron O'Shea

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public
License as published by the Free Software Foundation; either
version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public
License along with this program; if not, write to the
Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
Boston, MA 02110-1301 USA
'''

import imageio
from PIL import Image
from os.path import join, basename, splitext
import random
from collections import Counter
from keras.utils import Sequence
from keras_preprocessing.image import apply_affine_transform
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from joblib import Parallel, delayed, cpu_count
from scipy.signal import convolve2d

ROW_AXIS = 0
COL_AXIS = 1
CHANNEL_AXIS = 2


class VideoDataGenerator(Sequence):
    def __init__(self,
                 directory,
                 filenames,
                 labels,
                 batch_size,
                 shuffle=False,
                 height=224,
                 width=224,
                 max_frames=50,
                 optical_flow=False,
                 upsample=False,
                 featurewise_center=False,
                 gaussian_blur=False,
                 rotation_range=False,
                 brightness_range=False,
                 shear_range=False,
                 zoom_range=False,
                 horizontal_flip=False,
                 vertical_flip=False,
                 n_jobs=-1):
        self.directory = directory
        self.filenames = filenames
        self.labels = labels

        if upsample:
            self._upsample()

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.height = height
        self.width = width
        self.optical_flow = optical_flow
        self.max_frames = max_frames
        self.upsample = upsample
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
        self.filepaths = np.array(
            [join(self.directory, x) for x in self.filenames]
            )

    def _upsample(self):
        labels = {x: self.labels[x] for x in self.filenames}

        counter = Counter(labels.values())

        smallest = min(counter, key=counter.get)
        largest = max(counter, key=counter.get)     

        smol = [x for x in labels if self.labels[x] == smallest]
        tmp_filenames = self.filenames.tolist()

        for i in range(counter[largest] - counter[smallest]):
            random_filename = random.choice(smol)
            tmp_filenames.append(random_filename)
        self.filenames = np.array(tmp_filenames)

    def __len__(self):
        return int(np.floor(len(self.filepaths) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.filepaths))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indxs = self.indexes[index * self.batch_size:self.batch_size * (
            index + 1)]

        X, y = self.__data_generation(self.filepaths[indxs])

        return X, y

    def __data_generation(self, filepaths):
        def ___random_rotation(x,
                               rg,
                               fill_mode="nearest",
                               cval=0.,
                               interpolation_order=1):
            theta = np.random.uniform(-rg, rg)
            for index in range(x.shape[0]):
                x[index] = apply_affine_transform(
                    x[index],
                    theta=theta,
                    row_axis=ROW_AXIS,
                    col_axis=COL_AXIS,
                    channel_axis=CHANNEL_AXIS,
                    fill_mode=fill_mode,
                    cval=cval)
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
                x[index] = apply_affine_transform(
                    x[index],
                    tx=tx,
                    ty=ty,
                    row_axis=ROW_AXIS,
                    col_axis=COL_AXIS,
                    channel_axis=CHANNEL_AXIS,
                    fill_mode=fill_mode,
                    cval=cval)
            return x

        def __optical_flow(x, window_size=4, tau=1e-5, mode="same"):
            # Lucas-Kande method
            for frame_index in range(x.shape[0]):
                frame0 = x[frame_index]
                try:
                    I2g = x[frame_index + 1]
                except IndexError:
                    break
                kX = np.array([[-1., 1.], [-1., 1.]])
                kY = np.array([[-1., -1.], [1., 1.]])
                kT = np.array([[1., 1.], [1., 1.]])

                w = int(np.floor(window_size / 2))
                fxT = np.zeros((self.height, self.width, 3), dtype=float)
                fyT = np.zeros((self.height, self.width, 3), dtype=float)
                ftT = np.zeros((self.height, self.width, 3), dtype=float)

                for channel_idx in range(3):
                    fx = convolve2d(
                        frame0[:, :, channel_idx],
                        kX,
                        boundary="symm",
                        mode=mode)

                    fy = convolve2d(
                        frame0[:, :, channel_idx],
                        kY,
                        boundary="symm",
                        mode=mode)

                    ft = convolve2d(
                        I2g[:, :, channel_idx], kT, boundary="symm", mode=mode)

                    ft += convolve2d(
                        frame0[:, :, channel_idx],
                        -kT,
                        boundary="symm",
                        mode=mode)

                    fxT[:, :, channel_idx] = fx
                    fyT[:, :, channel_idx] = fy
                    ftT[:, :, channel_idx] = ft

                u = np.zeros((self.height, self.width))
                v = np.zeros((self.height, self.width))

                for i in range(w, frame0.shape[0] - w):
                    for j in range(w, frame0.shape[1] - w):
                        Ix = fxT[i - w:i + w + 1, j - w:j + w + 1, :].flatten()
                        Iy = fyT[i - w:i + w + 1, j - w:j + w + 1, :].flatten()
                        It = ftT[i - w:i + w + 1, j - w:j + w + 1, :].flatten()
                        b = np.reshape(It, (It.shape[0], 1))
                        A = np.vstack((Ix, Iy)).T

                        nu = np.zeros((2, 1))
                        if np.min(abs(np.linalg.eigvals(np.matmul(A.T,
                                                                  A)))) >= tau:

                            nu = np.matmul(np.linalg.pinv(A), b)

                        u[i, j] = nu[0]
                        v[i, j] = nu[1]

                mask = np.square(u) + np.square(v) == 0.0
                rawX = x[frame_index]

                for channel_idx in range(3):
                    channel_X = rawX[:, :, channel_idx]
                    m = np.ma.masked_array(
                        data=channel_X, mask=mask, fill_value=0.0).filled()
                    rawX[:, :, channel_idx] = m

                x[frame_index] = rawX

            return x

        def __gaussian_blur(x):
            rg = random.randint(0, self.gaussian_blur)
            for index in range(x.shape[0]):
                x[index] = gaussian_filter(x[index], rg)
            return x

        def __random_zoom(x, zoom_range, fill_mode="nearest", cval=0.):
            zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
            for index in range(x.shape[0]):
                x[index] = apply_affine_transform(
                    x[index],
                    zx=zx,
                    zy=zy,
                    row_axis=ROW_AXIS,
                    col_axis=COL_AXIS,
                    channel_axis=CHANNEL_AXIS,
                    fill_mode=fill_mode,
                    cval=cval)
            return x

        def __random_shear(x, intensity, fill_mode="nearest", cval=0.):
            shear = np.random.uniform(-intensity, intensity)
            for index in range(x.shape[0]):
                x[index] = apply_affine_transform(
                    x[index],
                    shear=shear,
                    row_axis=ROW_AXIS,
                    col_axis=COL_AXIS,
                    channel_axis=CHANNEL_AXIS,
                    fill_mode=fill_mode,
                    cval=cval)
            return x

        def ___read(filepath):
            video = imageio.get_reader(filepath, "ffmpeg")

            frames = np.empty(
                (self.max_frames, self.width, self.height, 3), dtype="uint8")

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

            return frames

        X = np.zeros(
            (len(filepaths), self.max_frames, self.width, self.height, 3),
            dtype="uint8"
            )
        y = []

        def _do(filepath):
            x = ___read(filepath)

            if self.featurewise_center:
                x = __featurewise_centre(x)

            if self.gaussian_blur:
                x = __gaussian_blur(x)

            if self.rotation_range:
                x = ___random_rotation(x, self.rotation_range)

            if self.horizontal_flip:
                # coinflip
                if random.randint(0, 1) == 1:
                    x = __flip_axis(x, ROW_AXIS)

            if self.vertical_flip:
                if random.randint(0, 1) == 1:
                    x = __flip_axis(x, COL_AXIS)

            if self.shear_range:
                x = __random_shear(x, self.shear_range)

            if self.brightness_range:
                raise NotImplementedError("Brightness yet to be implemented")

            if self.zoom_range:
                x = __random_zoom(x, self.zoom_range)

            if self.optical_flow:
                x = __optical_flow(x, window_size=self.optical_flow)

            return x, self.labels["".join(
                splitext(basename(filepath)))]

        if self.n_jobs == -1:
            self.n_jobs = cpu_count()
        if self.n_jobs > 1:
            data = Parallel(
                n_jobs=self.n_jobs,
                prefer="threads")(delayed(_do)(fp) for fp in filepaths)
        else:
            data = [_do(fp) for fp in filepaths]

        for indx in range(len(data)):
            x, _y = data[indx]
            X[indx] = x
            y.append(_y)

        return X, y
