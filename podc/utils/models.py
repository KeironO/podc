import os

from keras.models import Sequential
from keras.models import load_model as k_load_model
from keras.engine import Model
from keras.optimizers import *
from keras import backend as K
from keras.layers import *
from keras.applications import * 

class BaseModel:
    def __init__(self, max_frames, height, width, output_dir, n_channels=3, output_type="categorical"):
        self.max_frames = max_frames
        self.height = height
        self.width = width
        self.output_dir = output_dir
        self.model_fp = os.path.join(output_dir, "base_model.h5")
        if os.path.isfile(self.model_fp):
            self.model = self.load_model()
        else:
            # Ignore pylint as this is only to be called by children of THIS class.
            self.model = self.generate_model()
            self.save_model()
        self.output_type = output_type

    def save_model(self):
        self.model.save(self.model_fp)

    def load_model(self):
        return k_load_model(self.model_fp)

    def generate_model(self):
        raise Exception("This isn't to be called!")

class ResNet50v1(BaseModel):
    def generate_model(self):
        inp = Input(shape=(self.max_frames, self.height, self.width, 3), name="input")
        
        cnn = InceptionV3(weights="imagenet", include_top=False, pooling="avg", input_shape=(self.height, self.width, 3))
        cnn.trainable = False
        encoded_frames = TimeDistributed(Lambda(lambda x: cnn(x)))(inp)

        encoded_vid = LSTM(256)(encoded_frames)

        output = Dense(1, activation="sigmoid")(encoded_vid)
        model = Model(inputs=[inp], outputs=output)

        opt = Adam(lr = 1e-4, beta_1=0.9)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model



if __name__ == "__main__":
    model = ResNet50v1(100, 75, 75, output_dir="/tmp/", output_type="")
    print(model.model)
