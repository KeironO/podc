from keras.utils import Sequence
import os
import numpy as np
from PIL import Image
from keras_preprocessing.image import apply_affine_transform
import random

class FHCDataGenerator(Sequence):
    def __init__(self, data_dir, ids, height, width, output_height, output_width, batch_size=32, shuffle=True, rotation_range=False, shear_range=False, zoom_range=False, horizontal_flip=False, vertical_flip=False):
        self.data_dir = data_dir
        self.ids = ids
        self.height = int(height)
        self.width = int(width)
        self.output_height = int(output_height)
        self.output_width = int(output_width)
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Transformation stuff

        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

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

        def __random_rotation(x, y, rg, fill_mode="nearest", cval=0., interpolation_order=1):
            theta = np.random.uniform(-rg, rg)
            x = apply_affine_transform(x, theta=theta, row_axis=0, col_axis=1, channel_axis=2, fill_mode=fill_mode, cval=cval)
            y = apply_affine_transform(y, theta=theta, row_axis=0, col_axis=1, channel_axis=2, fill_mode=fill_mode, cval=cval)

        def __flip_axis(x, y, axis):
            x = x.swapaxes(axis, 0)
            x = x[::-1, ...]
            x = x.swapaxes(0, axis)

            y = y.swapaxes(axis, 0)
            y = y[::-1, ...]
            y = y.swapaxes(0, axis)
            return x, y

        def __random_shift(x, y, wrg, hrg, fill_mode="nearest", cval=0.):
            tx = np.random.uniform(-hrg, hrg) * x.shape[0]
            ty = np.random.uniform(-wrg, wrg) * x.shape[1]

            x = apply_affine_transform(x, tx=tx, ty=ty, row_axis=0, col_axis=1, channel_axis=2, fill_mode=fill_mode, cval=cval)
            y = apply_affine_transform(y, tx=tx, ty=ty, row_axis=0, col_axis=1, channel_axis=2, fill_mode=fill_mode, cval=cval)

            return x, y

        def _load_image(identifier):    
            fp = os.path.join(self.data_dir, identifier + ".png")
            img = Image.open(fp)
            img = img.convert("RGB")
            img = img.resize((self.width, self.height))
            return np.array(img)

        def __random_zoom(x, y, zoom_range, fill_mode="nearest", cval=0.):
            zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
            x = apply_affine_transform(x, zx=zx, zy=zy, row_axis=0, col_axis=1, channel_axis=2, fill_mode=fill_mode, cval=cval)
            y = apply_affine_transform(y, zx=zx, zy=zy, row_axis=0, col_axis=1, channel_axis=2, fill_mode=fill_mode, cval=cval)
            return x, y

        def __random_shear(x, y, intensity, fill_mode="nearest", cval=0.):
            shear = np.random.uniform(-intensity, intensity)
            x = apply_affine_transform(x, shear=shear, row_axis=0, col_axis=1, channel_axis=2, fill_mode=fill_mode, cval=cval)
            y = apply_affine_transform(y, shear=shear, row_axis=0, col_axis=1, channel_axis=2, fill_mode=fill_mode, cval=cval)
            return x, y
        
        def _load_annotations(identifier):

            def __colour_elipsoid(img):
                img = np.array(img)
                img = (img == 255).astype(int)

                return np.array(np.maximum.accumulate(img, 1) & np.maximum.accumulate(img[:, ::-1], 1)[:, ::-1])


            fp = os.path.join(self.data_dir, identifier + "_Annotation.png")
            img = Image.open(fp)
            img = __colour_elipsoid(img)

            img = Image.fromarray(img.astype("uint8"))

            img = np.array(img.resize((self.output_width, self.output_height)))

            o = np.zeros((self.output_width, self.output_height, 2))
            # On the assumption that I'm only interested in /only/ the black and white segmentation.
            for i in range(2):
                o[:, :, i] = (img == i)
            img = np.array(o)
            return img

        identifiers = self.ids[indexes]

        X = np.zeros((len(identifiers), self.height, self.width, 3))
        y = []

        for indx, identifier in enumerate(identifiers):
            x = _load_image(identifier)
            _y = _load_annotations(identifier)

            if self.rotation_range != False:
                x, y = __random_rotation(x, y, self.rotation_range)
            
            if self.shear_range != False:
                x, y = __random_shear(x, y, self.shear_range)

            if self.zoom_range != False:
                x, y = __random_zoom(x, y, self.zoom_range)

            if self.horizontal_flip != False:
                if random.randint(0, 1) == 1:
                    x, y = __flip_axis(x, y, 1)
            
            if self.vertical_flip != False:
                if random.randint(0, 1) == 1:
                    x, y = __flip_axis(x, y, 0)
            
            x[indx] = x
            y.append(_y)

        return X, np.array(y)


        