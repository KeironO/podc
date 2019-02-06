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
from keras.models import Sequential
from keras.models import load_model as k_load_model
from keras.engine import Model
from keras.optimizers import SGD
from keras.layers import (
    GlobalAveragePooling2D,
    Dense, 
    Dropout,
    GlobalMaxPool2D,
    Input,
    LSTM,
    TimeDistributed,
    Activation,
    Conv2D,
    Conv2DTranspose,
    ConvLSTM2D
    )
from keras.applications import VGG16, VGG19, MobileNet, InceptionV3


class BaseModel:
    def __init__(
        self, height: int,
        width: int,
        output_dir: str,
        n_classes: int,
        max_frames: int = 1,
        n_channels: int = 3,
        output_type: str = "categorical"
        ) -> None:

        self.height = int(height)
        self.width = int(width)
        self.output_dir = output_dir
        self.n_classes = n_classes
        self.max_frames = max_frames
        self.model_fp = join(output_dir, "base_model.h5")

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

    def generate_model(self) -> RuntimeError:
        raise RuntimeError("You only call this on a subclass of Model")


class VGG19Image(BaseModel):
    def generate_model(self) -> None:
        cnn = VGG19(
            include_top=False,
            weights="imagenet",
            input_shape=(self.height, self.width, 3)
            )

        x = Sequential()
        x.add(GlobalAveragePooling2D(input_shape=cnn.output_shape[1:], data_format=None))
        x.add(Dense(512, activation='relu'))
        x.add(Dropout(0.5))
        x.add(Dense(1, activation='sigmoid'))

        model = Model(inputs=cnn.input, outputs=x(cnn.output))
        opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0)
        
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model
        

class VGG19v1(BaseModel):
    def generate_model(self) -> None:
        inp = Input(
            shape=(self.max_frames, self.height, self.width, 3),
            name="input"
            )
        cnn = VGG19(include_top=False)

        for layer in cnn.layers:
            layer.trainable = False
        
        frame_acts = TimeDistributed(cnn)(inp)

        hid_states = ConvLSTM2D(
            512,
            (3, 3),
            padding="same",
            return_sequences=True,
            recurrent_dropout=0.2,
            dropout=0.2
            )(frame_acts)
        
        conv_hid_states = TimeDistributed(Conv2D(512, (1,1), activation="relu", padding="same"))(hid_states)

        conv_acts = TimeDistributed(Conv2D(512, (1, 1), activation="relu", padding="same"))(frame_acts)

        eunice = TimeDistributed(Conv2D(1, (1, 1), activation="relu", padding="same"))(Activation(activation="tanh")(Add()([conv_acts, conv_hid_states])))

        att = Activation(activation="softmax")(eunice)

        nn = Multiply()([frame_acts, att])
        nn = ConvLSTM2D(512, (3, 3), padding="same", recurrent_dropout=0.2, dropout=0.2)(nn)

        nn = GlobalMaxPool2D()(nn)

        outputs = Dense(1, activation="sigmoid")(nn)

        model = Model(inputs=inp, outputs=outputs)
        opt = Adam(lr = 1e-4, beta_1=0.9)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model

class SmolNet(BaseModel):

    def generate_model(self) -> None:

        base_model  = MobileNet(input_shape=(self.height,self.width,3), include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        cnn_model = Model(inputs=base_model.input, outputs=x)
    
        model = Sequential()
        model.add(TimeDistributed(cnn_model, input_shape=(self.max_frames, self.height, self.width, 3)))
        model.add(TimeDistributed(Flatten()))
    
        model.add(LSTM(1, return_sequences=True))

        opt = Adam(lr = 1e-4, beta_1=0.9)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model


class VGG16FHC(BaseModel):
    def generate_model(self) -> None:    
        #This is a bastardised version of VGG16, VGG19 is probably a bit too big to run on our hardware, which is likely to limit our experiment going forward.
    
        img_input = Input(shape=(self.height,self.width, 3))

        cnn = VGG19(include_top=False, input_tensor=img_input)

        x = Conv2D(filters=2, kernel_size=(1,1))(cnn.output)

        x = Conv2DTranspose(filters=self.n_classes, kernel_size=(64, 64), strides=(32, 32), padding="same", activation="sigmoid")(x)
        model = Model(inputs=img_input, outputs=x )

        for layer in model.layers[:15]:
            # Ensuring that the top of the model (VGG16 classifier) is set to train.
            layer.trainable = True
        
        if self.n_classes == 1:
            l = "binary_crossentropy"
        elif self.n_classes > 1:
            l = "categorical_crossentropy"
        
        sgd = SGD()

        model.compile(loss=l, optimizer=sgd)

        return model


class VGG16v1(BaseModel):
    def generate_model(self) -> None:
        inp = Input(shape=(self.max_frames, self.height, self.width, 3), name="input")

        cnn = VGG16(weights="imagenet", include_top=False)
        
        for layer in cnn.layers:
            layer.trainable = False

        tdcnn = TimeDistributed(cnn)(inp)
        tdcnn = TimeDistributed(Flatten())(tdcnn)
        tdcnn = MaxPooling1D(pool_size=self.max_frames)(tdcnn)

        tdcnn = Dropout(0.5)(tdcnn)
        tdcnn = Dense(512, activation="relu")(tdcnn)

        tdcnn = Flatten()(tdcnn)
        tdcnn = Dropout(0.5)(tdcnn)

        tdcnn = Dense(1, activation="softmax")(tdcnn)

        model = Model(inputs=inp, outputs=tdcnn)
        opt = Adam(lr=1e-4, beta_1=0.9)
        model.compile(
            loss="binary_crossentropy",
            optimizer=opt,
            metrics=["accuracy"]
            )
        return model     

class ResNet50v1(BaseModel):
    def generate_model(self) -> None:
        inp = Input(shape=(self.max_frames, self.height, self.width, 3), name="input")
        
        cnn = InceptionV3(weights="imagenet", include_top=False, pooling="avg", input_shape=(self.height, self.width, 3))
        
        for layer in cnn.layers:
            layer.trainable = False

        encoded_frames = TimeDistributed(Lambda(lambda x: cnn(x)))(inp)

        encoded_vid = LSTM(256)(encoded_frames)

        output = Dense(1, activation="sigmoid")(encoded_vid)
        model = Model(inputs=[inp], outputs=output)

        opt = Adam(lr = 1e-4, beta_1=0.9)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model
