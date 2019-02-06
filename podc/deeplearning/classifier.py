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


from keras.models import Sequential
from keras.layers import TimeDistributed, Dense, Activation, Dropout, Conv2D, LSTM, MaxPooling2D, Flatten, Bidirectional, Input, Masking, multiply, Reshape, Lambda
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, History, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.initializers import glorot_uniform
from keras.applications import ResNet50, InceptionV3
from keras.engine import Model
from keras import backend as K
import numpy as np

class Classifier(object):
    def __init__(self, height, width, max_frames, clf):
        self.max_frames = max_frames
        self.height = height
        self.width = width
        self.trained = False
        self.clf = clf
        #self.clf = self.build_clf()
    
    def train(self, generator, validation_data, model_fp, class_weights=False, epochs=10, patience=5):
        if self.trained != False:
            raise Exception("!! ERROR !! THIS MODEL HAS ALREADY BEEN TRAINED!")

        hi = History()
        mc = ModelCheckpoint(model_fp, monitor="val_loss", verbose=1, save_best_only=True)

        es = EarlyStopping(monitor="val_loss", verbose=1, patience=patience)
        
        if class_weights == False:
            history = self.clf.fit_generator(generator, epochs=epochs, callbacks=[hi, mc, es],
                                        validation_data=validation_data, verbose=1)
        else:
            history = self.clf.fit_generator(generator, epochs=epochs, callbacks=[hi, mc, es],
                                        validation_data=validation_data, verbose=1, class_weight=class_weights)

        self.clf.load_weights(model_fp)
        self.history = history
        self.trained = True

    def predict(self, generator):
        y_true = [y for _, y in generator]
        y_pred = self.clf.predict_generator(generator)
        for index, pred in enumerate(y_pred):
            if pred > 0.5:
                y_pred[index] = 1
            else:
                y_pred[index] = 0
        return y_true, y_pred


        