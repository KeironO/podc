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
    def __init__(self, max_frames, height, width, output_dir, n_channels=3, output_type="categorical"):
        self.max_frames = int(max_frames)
        self.height = int(height)
        self.width = int(width)
        self.output_dir = output_dir
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

        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_last' )(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format='channels_last' )(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format='channels_last' )(x)
        f1 = x
        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format='channels_last' )(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format='channels_last' )(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format='channels_last' )(x)
        f2 = x

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format='channels_last' )(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format='channels_last' )(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format='channels_last' )(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format='channels_last' )(x)
        f3 = x

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format='channels_last' )(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format='channels_last' )(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format='channels_last' )(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format='channels_last' )(x)
        f4 = x

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format='channels_last' )(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format='channels_last' )(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format='channels_last' )(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format='channels_last' )(x)
        f5 = x

        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense( 1000 , activation='softmax', name='predictions')(x)

        vgg  = Model(  img_input , x  )

        levels = [f1 , f2 , f3 , f4 , f5 ]

        o = levels[3]
        
        o = ( ZeroPadding2D( (1,1) , data_format='channels_last' ))(o)
        o = ( Conv2D(512, (3, 3), padding='valid', data_format='channels_last'))(o)
        o = ( BatchNormalization())(o)

        o = ( UpSampling2D( (2,2), data_format='channels_last'))(o)
        o = ( ZeroPadding2D( (1,1), data_format='channels_last'))(o)
        o = ( Conv2D( 256, (3, 3), padding='valid', data_format='channels_last'))(o)
        o = ( BatchNormalization())(o)

        o = ( UpSampling2D((2,2)  , data_format='channels_last' ) )(o)
        o = ( ZeroPadding2D((1,1) , data_format='channels_last' ))(o)
        o = ( Conv2D( 128 , (3, 3), padding='valid' , data_format='channels_last' ))(o)
        o = ( BatchNormalization())(o)

        o = ( UpSampling2D((2,2)  , data_format='channels_last' ))(o)
        o = ( ZeroPadding2D((1,1)  , data_format='channels_last' ))(o)
        o = ( Conv2D( 64 , (3, 3), padding='valid'  , data_format='channels_last' ))(o)
        o = ( BatchNormalization())(o)


        o =  Conv2D( 1 , (3, 3) , padding='same', data_format='channels_last' )( o )
        o_shape = Model(img_input , o ).output_shape
        outputHeight = o_shape[1]
        outputWidth = o_shape[2]

        o = (Reshape((  -1  , outputHeight*outputWidth   )))(o)
        o = (Permute((2, 1)))(o)
        o = (Activation('softmax'))(o)
        model = Model( img_input , o )
        model.outputWidth = outputWidth
        model.outputHeight = outputHeight
        model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])

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