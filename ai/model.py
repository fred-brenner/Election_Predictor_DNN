from keras.layers import Dense, Input, LSTM, Flatten, Dropout, \
    MaxPooling2D, Conv2D, BatchNormalization, SpatialDropout2D, concatenate, \
    Reshape, Conv2DTranspose, UpSampling2D, CuDNNLSTM, Lambda
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
        y = concatenate([x1, x2])
        y = Dense(32, activation='relu')(y)
        y = Dense(dim_out, activation='sigmoid')(y)
        y = Lambda(lambda x: x*100)(y)

        model = Model([input_par, input_pop], y)
        return model

    if model_type == 'lstm':
        input_par = Input(shape=(dim_in[0],))
        input_pop = Input(shape=(dim_in[1][0], dim_in[1][1]))
        x1 = Dense(16, activation='relu')(input_par)

        x2 = CuDNNLSTM(nr, return_sequences=True)(input_pop)
        x2 = CuDNNLSTM(nr, return_sequences=False)(x2)
        x2 = Dropout(0.05)(x2)
        x2 = Dense(16, activation='relu')(x2)

        y = concatenate([x1, x2, input_par])
        y = Dense(32, activation='relu')(y)
        y = Dense(dim_out, activation='sigmoid')(y)
        # y = Lambda(lambda x: x * 100)(y)

        model = Model([input_par, input_pop], y)
        return model