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
    def __init__(self, height, width, max_frames=10):
        self.max_frames = max_frames
        self.height = height
        self.width = width
        self.trained = False
        self.clf = self.build_clf()

    def build_clf(self):
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

    def train(self, generator, validation_data, model_fp, epochs=10, patience=5):
        if self.trained != False:
            raise Exception("!! ERROR !! THIS MODEL HAS ALREADY BEEN TRAINED!")

        hi = History()
        mc = ModelCheckpoint(model_fp, monitor="val_loss", verbose=1, save_best_only=True)

        es = EarlyStopping(monitor="val_loss", verbose=1, patience=patience)

        history = self.clf.fit_generator(generator, epochs=epochs, callbacks=[hi, mc, es],
                                    validation_data=validation_data, verbose=1)

        self.clf.load_weights(model_fp)
        self.history = history
        self.trained = True

    def predict(self, generator):
        y_true = [y for X, y in generator]
        y_pred = self.clf.predict_generator(generator)
        return y_true, y_pred


        