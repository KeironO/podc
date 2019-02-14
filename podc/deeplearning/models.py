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

from os.path import isfile, join
import numpy as np
import itertools
from keras.models import Sequential, Model
from keras.models import load_model as k_load_model
from keras.engine import Model
from keras.optimizers import SGD, Adam
from keras.applications import VGG16, VGG19, MobileNet, InceptionV3
from keras.layers import (Add, GlobalAveragePooling2D, Dense, Lambda, Multiply,
                          Dropout, GlobalMaxPool2D, MaxPooling1D, Flatten,
                          Input, LSTM, TimeDistributed, Activation, Conv2D,
                          Conv2DTranspose, ConvLSTM2D)
from keras.callbacks import (
    EarlyStopping,
    History,
    ModelCheckpoint,
)

class BaseModel:
    def __init__(self,
                 height: int,
                 width: int,
                 output_dir: str,
                 n_classes: int = 1,
                 max_frames: int = 1,
                 n_channels: int = 3,
                 output_type: str = "categorical") -> None:

        self.height = int(height)
        self.width = int(width)
        self.output_dir = output_dir
        self.n_classes = n_classes
        self.max_frames = max_frames
        self.trained = False
        self.model_fp = join(output_dir, "model.h5")

        if isfile(self.model_fp):
            self.model = self.generate_model()
        else:
            self.model = self.generate_model()
            self.save_model()
        self.output_type = output_type

    def save_model(self):
        self.model.save(self.model_fp)

    def load_model(self) -> Model:
        return k_load_model(self.model_fp)

    def fit(self,
            generator,
            val_generator,
            epochs,
            patience,
            class_weights=False) -> dict:

        hi = History()

        mc = ModelCheckpoint(
            self.model_fp, monitor="val_loss", verbose=1, save_best_only=True)

        es = EarlyStopping(monitor="val_loss", verbose=1, patience=patience)

        clf = self.model

        if class_weights:
            history = clf.fit_generator(
                generator,
                epochs=epochs,
                callbacks=[hi, mc, es],
                validation_data=val_generator,
                verbose=1,
                class_weight=class_weights)
        else:
            history = clf.fit_generator(
                generator,
                epochs=epochs,
                callbacks=[hi, mc, es],
                validation_data=val_generator,
                verbose=1)

        self.model.load_weights(self.model_fp)
        self.history = history
        self.trained = True

    def predict_pod(self, generator):
        y_true = []
        y_pred = []
        for X, y in generator:
            predictions = list(itertools.chain.from_iterable(self.model.predict(X)))

            y_true.extend(y)
            y_pred.extend(predictions)

        return y_true, y_pred

    def generate_model(self) -> RuntimeError:
        return RuntimeError("You only call this on a subclass of Model")


class VGG16FHC(BaseModel):
    def generate_model(self) -> Model:
        # This is a bastardised version of VGG16, VGG19 is probably a bit too
        # big to run on our hardware, which is likely to limit our experiment
        # going forward.

        img_input = Input(shape=(self.height, self.width, 3))

        cnn = VGG19(include_top=False, input_tensor=img_input)

        for layer in cnn.layers:
            # HACK: Ensuring that the top of the model (VGG16 classifier)
            # is set to train.
            layer.trainable = True

        x = Conv2D(filters=2, kernel_size=(1, 1))(cnn.output)

        x = Conv2DTranspose(
            filters=self.n_classes,
            kernel_size=(64, 64),
            strides=(32, 32),
            padding="same",
            activation="sigmoid")(x)

        model = Model(inputs=img_input, outputs=x)

        if self.n_classes == 1:
            loss_func = "binary_crossentropy"
        elif self.n_classes > 1:
            loss_func = "categorical_crossentropy"

        sgd = SGD()
        model.compile(loss=loss_func, optimizer=sgd)

        return model


class VGG19v1(BaseModel):
    def generate_model(self) -> Model:
        inp = Input(
            shape=(self.max_frames, self.height, self.width, 3), name="input")
        cnn = VGG16(include_top=False)

        for layer in cnn.layers:
            layer.trainable = False

        frame_acts = TimeDistributed(cnn)(inp)

        hid_states = ConvLSTM2D(
            512, (3, 3),
            padding="same",
            return_sequences=True,
            recurrent_dropout=0.2,
            dropout=0.2)(frame_acts)

        conv_hid_states = TimeDistributed(
            Conv2D(512, (1, 1), activation="relu", padding="same"))(hid_states)

        conv_acts = TimeDistributed(
            Conv2D(512, (1, 1), activation="relu", padding="same"))(frame_acts)

        acct = Activation(activation="tanh")(
            Add()([conv_acts, conv_hid_states]))

        eunice = TimeDistributed(
            Conv2D(1, (1, 1), activation="relu", padding="same"))(acct)

        att = Activation(activation="softmax")(eunice)

        nn = Multiply()([frame_acts, att])

        nn = ConvLSTM2D(
            512, (3, 3), padding="same", recurrent_dropout=0.2,
            dropout=0.2)(nn)

        nn = GlobalMaxPool2D()(nn)

        outputs = Dense(1, activation="sigmoid")(nn)

        opt = Adam(lr=1e-4, beta_1=0.9)

        model = Model(inputs=inp, outputs=outputs)
        model.compile(
            loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model
