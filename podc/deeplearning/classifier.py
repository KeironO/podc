from keras.models import Sequential
from keras.layers import TimeDistributed, Dense, Activation, Dropout, Conv2D, LSTM, MaxPooling2D, Flatten, Bidirectional, Input, Masking, multiply, Reshape, Lambda
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, History, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.initializers import glorot_uniform
from keras.applications import ResNet50, InceptionV3
from keras.engine import Model
from keras import backend as K

class Classifier(object):
    def __init__(self, height, width, max_frames=10):
        self.max_frames = max_frames
        self.height = height
        self.width = width
        self.trained = False
        self.clf = self.build_clf()

    def build_clf(self):
        inp = Input(shape=(self.max_frames, self.height, self.width, 3), name="input")
        
        incept = InceptionV3(weights="imagenet", include_top="False", pooling="avg")
        incept.trainable = False

        td_incept = TimeDistributed(Lambda(lambda x: incept(x)))(inp)
        fin_lstm = LSTM(256)(td_incept)
        outputs = Dense(1, activation="softmax", kernel_initializer=glorot_uniform(seed=3), name="ol")(fin_lstm)

        model = Model(inputs=[inp], outputs=outputs)

        opt = Adam()

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
        