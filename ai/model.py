from keras.layers import Dense, Input, LSTM, Flatten, Dropout, \
    MaxPooling2D, Conv2D, BatchNormalization, SpatialDropout2D, concatenate, \
    Reshape, Conv2DTranspose, UpSampling2D, CuDNNLSTM
from keras.models import Model
import numpy as np


def create_tf_model(model_type, dim_in, dim_out, nr=128):
    # Deep Neural Network
    if model_type == 'dnn':
        input_par = Input(shape=(dim_in[0],))
        input_pop = Input(shape=(dim_in[1],))
        # x = Dropout(0.2)(input_num)
        x1 = Dense(nr*2, activation='relu')(input_pop)
        x1 = Dropout(0.1)(x1)
        # x = Dense(nr*4, activation='relu')(x)
        # x = Dropout(0.1)(x)
        x1 = Dense(nr, activation='relu')(x1)
        # x = Dense(dim_out, activation='relu')(x)
        x2 = Dense(nr, activation='relu')(input_par)
        x = concatenate([x1, x2])
        x = Dense(32, activation='relu')(x)
        y = Dense(dim_out, activation='sigmoid')(x)

        model = Model([input_par, input_pop], y)
        return model