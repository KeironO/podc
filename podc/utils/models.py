import os

from keras.models import Sequential
from keras.models import load_model as k_load_model
from keras.engine import Model
from keras.optimizers import *
from keras import backend as K
from keras.layers import *
from keras.applications import * 


from keras.utils import plot_model

class BaseModel:
    def __init__(self, max_frames, height, width, output_dir, n_classes, n_channels=3, output_type="categorical"):
        self.max_frames = int(max_frames)
        self.height = int(height)
        self.width = int(width)
        self.output_dir = output_dir
        self.n_classes = n_classes
        self.model_fp = os.path.join(output_dir, "base_model.h5")
        if os.path.isfile(self.model_fp):
            self.model = self.generate_model()
        else:
            self.model = self.generate_model()
            self.save_model()
        self.output_type = output_type

    def save_model(self):
        self.model.save(self.model_fp)

    def load_model(self):
        return k_load_model(self.model_fp)

    def generate_model(self):
        raise Exception("This isn't to be called!")


class VGG19Image(BaseModel):
    def generate_model(self):
        cnn = VGG19(include_top=False, weights="imagenet", input_shape=(self.height, self.width, 3))

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
    def generate_model(self):
        inp = Input(shape=(self.max_frames, self.height, self.width, 3), name="input")
        cnn = VGG19(include_top=False)

        for layer in cnn.layers:
            layer.trainable = False
        
        frame_acts = TimeDistributed(cnn)(inp)

        hid_states = ConvLSTM2D(512, (3, 3), padding="same", return_sequences=True, recurrent_dropout=0.2, dropout=0.2)(frame_acts)
        
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

    def generate_model(self):

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


class VGG19FHC(BaseModel):
    def generate_model(self):
        '''
            This is a bastardised version of VGG16, VGG19 is probably a bit too big to run on our hardware.

            You can find U-net details online.
            
        '''
        img_input = Input(shape=(self.height,self.width, 3))

        cnn = VGG16(include_top=False, input_tensor=img_input)

        x = Conv2D(filters=2, kernel_size=(1,1))(cnn.output)
        if self.n_classes == 1:
            out_filters = 2
        else:
            out_filters = self.n_classes

        x = Conv2DTranspose(filters=out_filters, kernel_size=(64, 64), strides=(32, 32), padding="same", activation="sigmoid")(x)
        model = Model(inputs=img_input, outputs=x )

        for layer in model.layers[:15]:
            layer.trainable = True
        

        model.compile(loss="categorical_crossentropy", optimizer="sgd")

        return model



class VGG16v1(BaseModel):
    def generate_model(self):
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
        opt = Adam(lr = 1e-4, beta_1=0.9)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model


        

class ResNet50v1(BaseModel):
    def generate_model(self):
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



if __name__ == "__main__":
    #model = ResNet50v1(100, 75, 75, output_dir="/tmp/", output_type="")
    #model = VGG19v1(100, 64, 64, output_dir="/tmp/")
    #print(model.model)
    pass