from keras.layers import Dense, Input, LSTM, Flatten, Dropout, \
    MaxPooling2D, Conv2D, BatchNormalization, SpatialDropout2D, concatenate, \
    Reshape, Conv2DTranspose, UpSampling2D, CuDNNLSTM
from keras.models import Model
import numpy as np


def create_tf_model(model_type, dim_in, dim_out, nr=128):
    # Deep Neural Network
    if model_type == 'dnn':
        input_num = Input(shape=(dim_in,))
        # x = Dropout(0.2)(input_num)
        x = Dense(nr*2, activation='relu')(input_num)
        x = Dropout(0.1)(x)
        x = Dense(nr*4, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(nr, activation='relu')(x)

        x = Dense(dim_out, activation='sigmoid')(x)

        model = Model(input_num, x)
        return model