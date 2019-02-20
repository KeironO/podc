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
from keras.callbacks import (EarlyStopping, History, ModelCheckpoint,
                             ReduceLROnPlateau)

class BaseModel:
    def __init__(self,
                 height: int,
                 width: int,
                 output_dir: str,
                 n_classes: int = 1,
                 max_frames: int = 1,
                 n_channels: int = 3,
                 model_parameters: dict = False,
                 output_type: str = "categorical") -> None:

        self.height = int(height)
        self.width = int(width)
        self.output_dir = output_dir
        self.n_classes = n_classes
        self.max_frames = max_frames
        self.model_parameters = model_parameters
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
            monitor="val_loss",
            class_weights=False) -> dict:

        hi = History()

        mc = ModelCheckpoint(
            self.model_fp, monitor=monitor, verbose=1, save_best_only=True)

        es = EarlyStopping(monitor=monitor, verbose=1, patience=patience)

        reduce_lr = ReduceLROnPlateau(
            monitor=monitor,
            factor=0.2,
            patience=int(patience / 2),
            min_lr=1e-9)

        clf = self.model

        if class_weights:
            history = clf.fit_generator(
                generator,
                epochs=epochs,
                callbacks=[hi, mc, es, reduce_lr],
                validation_data=val_generator,
                verbose=1,
                class_weight=class_weights)
        else:
            history = clf.fit_generator(
                generator,
                epochs=epochs,
                callbacks=[hi, mc, es, reduce_lr],
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


class CheapoKeepo(BaseModel):

    """
        This model I can train fairly effortlessly on very little hardware.

        It's primarily used for testing purposes.
    """

    default_parameters = {
        "lstm_count": 64,
        "lstm_dropout": 0.2
    }

    def generate_model(self) -> Model:
        if not self.model_parameters:
            self.model_parameters = self.default_parameters

        inps = Input(shape=(None, self.height, self.width, 3))

        cnn = VGG16(weights=None, include_top=False, pooling="avg")

        if "vgg16_weights_fp" in self.model_parameters:
            cnn.load_weights(self.model_parameters["vgg16_weights_fp"])

            for layer in cnn.layers:
                layer.trainable = False

        encoded_frame = TimeDistributed(Lambda(lambda x: cnn(x)))(inps)
        video = LSTM(self.model_parameters["lstm_count"])(encoded_frame)
        video = Dropout(self.model_parameters["lstm_dropout"])(video)

        opt = Adam(lr=0.005, decay=0.001)
        outputs = Dense(1, activation="linear")(video)

        model = Model(inputs=[inps], outputs=outputs)

        model.compile(optimizer=opt, loss="mean_squared_error")

        return model


class VGG16PouchOfDouglas(BaseModel):

    default_parameters = {
        "hid_states": {
            "filter": 512,
            "kernel_size": (3, 3),
            "recurrent_dropout": 0.2,
            "dropout": 0.2
        },
        "conv_hid_states": {
            "filter": 512,
            "kernel_size": (1, 1)
        },
        "conv_acts": {
            "filters": 512,
            "kernel_size": (1, 1)
        },
        "eunice": {
            "filters": 1,
            "kernel_size": (1, 1)
        },
        "nn": {
            "filter": 512,
            "kernel_size": (3, 3),
            "recurrent_dropout": 0.2,
            "dropout": 0.2
        },
        "opt": {
            "lr": 1e-4,
            "beta_1": 0.9,
            "beta_2": 0.9,
            "decay": 0.0
        }
    }

    def generate_model(self) -> Model:

        if not self.model_parameters:
            self.model_parameters = self.default_parameters

        inp = Input(
            shape=(self.max_frames, self.height, self.width, 3), name="input")

        cnn = VGG16(include_top=False, weights=None)

        if "vgg16_weights_fp" in self.model_parameters:
            cnn.load_weights(self.model_parameters["vgg16_weights_fp"])

            for layer in cnn.layers:
                layer.trainable = False

        frame_acts = TimeDistributed(cnn)(inp)

        lp = self.model_parameters["hid_states"]
        hid_states = ConvLSTM2D(
            lp["filter"],
            lp["kernel_size"],
            padding="same",
            return_sequences=True,
            recurrent_dropout=lp["recurrent_dropout"],
            dropout=lp["dropout"])(frame_acts)

        lp = self.model_parameters["conv_hid_states"]
        conv_hid_states = TimeDistributed(
            Conv2D(
                lp["filter"],
                lp["kernel_size"],
                activation="relu",
                padding="same"))(hid_states)

        lp = self.model_parameters["conv_acts"]
        conv_acts = TimeDistributed(
            Conv2D(
                lp["filter"],
                lp["kernel_size"],
                activation="relu",
                padding="same"))(frame_acts)

        acct = Activation(activation="tanh")(
            Add()([conv_acts, conv_hid_states]))

        lp = self.model_parameters["eunice"]
        eunice = TimeDistributed(
            Conv2D(
                lp["filter"],
                lp["kernel_size"],
                activation="relu",
                padding="same"))(acct)

        att = Activation(activation="softmax")(eunice)

        nn = Multiply()([frame_acts, att])

        lp = self.model_parameters["nn"]
        nn = ConvLSTM2D(
            lp["filter"],
            lp["kernel_size"],
            padding="same",
            recurrent_dropout=lp["recurrent_dropout"],
            dropout=lp["dropout"])(nn)

        nn = GlobalMaxPool2D()(nn)

        outputs = Dense(1, activation="sigmoid")(nn)

        lp = self.model_parameters["opt"]
        opt = Adam(
            lr=lp["lr"],
            beta_1=lp["beta_1"],
            beta_2=lp["beta_2"],
            decay=lp["decay"]
            )

        model = Model(inputs=inp, outputs=outputs)
        model.compile(
            loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model
