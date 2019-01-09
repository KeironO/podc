from keras.models import Sequential
from keras.layers import TimeDistributed, Dense, Activation, Dropout, Conv2D, LSTM, MaxPooling2D, Flatten, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, History, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau

class Model(object):
    def __init__(self, height, width, max_frames=10):
        self.max_frames = max_frames
        self.height = height
        self.width = width
        self.trained = False
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        # VGG Block Zero
        model.add(TimeDistributed(Conv2D(64, (3, 3), activation="relu", padding="same", trainable=False), name="0timedistconv0", input_shape=(self.max_frames, self.width, self.height, 3)))
        model.add(TimeDistributed(Conv2D(64, (3, 3), activation="relu", padding="same", trainable=False), name="0timedistconv1"))
        model.add(TimeDistributed(MaxPooling2D((2,2), strides=(2, 2)), name="0timedistmp"))
        # VGG Block One
        model.add(TimeDistributed(Conv2D(128, (3, 3), activation="relu", padding="same", trainable=False), name="1timedistconv0"))
        model.add(TimeDistributed(Conv2D(128, (3, 3), activation="relu", padding="same", trainable=False), name="1timedistconv1"))
        model.add(TimeDistributed(MaxPooling2D((2,2), strides=(2, 2)), name="1timedistmp"))
        # VGG Block Two
        model.add(TimeDistributed(Conv2D(256, (3, 3), activation="relu", padding="same", trainable=False), name="2timedistconv0"))
        model.add(TimeDistributed(Conv2D(256, (3, 3), activation="relu", padding="same", trainable=False), name="2timedistconv1"))
        model.add(TimeDistributed(Conv2D(256, (3, 3), activation="relu", padding="same", trainable=False), name="2timedistconv2"))
        model.add(TimeDistributed(MaxPooling2D((2,2), strides=(2, 2)), name="2timedistmp"))
        # VGG Block Three
        model.add(TimeDistributed(Conv2D(512, (3, 3), activation="relu", padding="same", trainable=False), name="3timedistconv0"))
        model.add(TimeDistributed(Conv2D(512, (3, 3), activation="relu", padding="same", trainable=False), name="3timedistconv1"))
        model.add(TimeDistributed(Conv2D(512, (3, 3), activation="relu", padding="same", trainable=False), name="3timedistconv2"))
        model.add(TimeDistributed(MaxPooling2D((2,2), strides=(2, 2)), name="3timedistmp"))
        # VGG Block Four
        model.add(TimeDistributed(Conv2D(512, (3, 3), activation="relu", padding="same", trainable=False), name="4timedistconv0"))
        model.add(TimeDistributed(Conv2D(512, (3, 3), activation="relu", padding="same", trainable=False), name="4timedistconv1"))
        model.add(TimeDistributed(Conv2D(512, (3, 3), activation="relu", padding="same", trainable=False), name="4timedistconv2"))
        model.add(TimeDistributed(MaxPooling2D((2,2), strides=(2, 2)), name="4timedistmp"))    
        # Loading weights for VGG16
        model.load_weights("/home/keo7/Downloads/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")

        # Bidirectional LSTM
        bidirnn = Sequential()
        bidirnn.add(TimeDistributed(Flatten(), input_shape=model.output_shape[1:], name="5timedistflatten"))
        bidirnn.add(Bidirectional(LSTM(32, return_sequences=True), name="5bidirlstm0"))
        bidirnn.add(Dropout(0.2, name="5dropout0"))
        bidirnn.add(Bidirectional(LSTM(64, return_sequences=True), name="5bidirlstm1"))
        bidirnn.add(Dropout(0.2, name="5dropout1"))
        bidirnn.add(LSTM(1, activation="softmax", return_sequences=False, name="5bidirlstm2"))

        # Add to model
        for layer in bidirnn.layers:
            model.add(layer)

        adam = Adam(lr = 1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])
        
        return model

    def train_model(self, generator, validation_data, model_fp, epochs=10, patience=5):
        if self.trained != False:
            raise Exception("!! ERROR !! THIS MODEL HAS ALREADY BEEN TRAINED!")


        hi = History()
        mc = ModelCheckpoint(model_fp, monitor="val_loss", verbose=1, save_best_only=True)

        es = EarlyStopping(monitor="val_loss", verbose=1, patience=patience)

        history = self.model.fit_generator(generator, epochs=epochs, callbacks=[hi, mc, es],
                                    validation_data=validation_data, verbose=1)

        self.model.load_weights(model_fp)
        self.history = history
        self.trained = True
        