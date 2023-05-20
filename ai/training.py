import numpy as np
from keras.optimizers import adam_v2
import pandas as pd
import joblib

from parameter_estimation.read_data import preprocess_data, standardize_data
from ai.helpers import *
from ai.model import create_tf_model
from parameter_estimation.import_csv import import_csv


# model_name = 'dnn'
model_name = 'lstm'

# limit gpu ram usage
conf = tf.compat.v1.ConfigProto()
conf.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=conf)
tf.compat.v1.keras.backend.set_session(sess)


def train(csv_file_name, neuron_size=128, learning_rate=1e-6):
    # import data
    csv_data = import_csv(csv_file_name)
    ml_in, ml_out = preprocess_data(csv_data, model_name)


    # Check Cuda compatible GPU
    if not test_gpu_tf():
        exit()

    # Set training parameters
    # learning_rate = 8e-6
    n_epochs = 500
    batch_size = 1
    # neuron_size = 128
    # loss = 'mean_squared_error'
    # loss = 'mean_squared_logarithmic_error'
    # loss = 'mean_absolute_percentage_error'
    # loss = 'mean_absolute_error'
    loss = 'binary_crossentropy'

    # setup ML model
    model = create_tf_model(model_name, dim_in=[ml_in[0].shape[1], ml_in[1].shape[1:]],
                            dim_out=ml_out.shape[1], nr=neuron_size)

    adam = adam_v2.Adam(learning_rate=learning_rate, decay=learning_rate / n_epochs)
    # model.compile(loss=loss, optimizer=adam)
    model.compile(loss=loss, optimizer=adam, metrics=['accuracy'])

    # Train autoencoder
    training = model.fit(x=ml_in, y=ml_out, epochs=n_epochs,
                         batch_size=batch_size, shuffle=True, verbose=1)

    # Save model
    inp = input("Save model? (y or n)")
    if inp.lower() == 'y':
        save_path = f'ai/dumps/{model_name}.h5'
        print(f"Saving model at: {save_path}")
        model.save(save_path)

    print("\nFinished Training")


def predict(ml_in=None):
    # load model
    save_path = 'ai/dumps/'
    model, _ = load_keras_model(model_name, save_path)
    if model is None:
        exit()
    y = model.predict(ml_in, verbose=0)
    # result = np.vstack((ml_in[:, 1], y[:, 0])).T
    return y
    # result = np.vstack((ml_in[1], y[:, 0])).T
    # print(result)
    # return result
    # y_df = pd.DataFrame(y, columns=y_col)
    # y = scaler_y.inverse_transform(y_df)
    # y_df = pd.DataFrame(y, columns=y_col)
    # return y_df, y_col


if __name__ == '__main__':
    # csv_file_name = '../0.51-0.49 #11-111.csv'
    # csv_file_name = '../0.51-0.49 #11-211(only odd numbers).csv'
    # csv_file_name = '../0.51-0.49 #11-3011.csv'
    # csv_file_name = '../0.51-0.49 #11-6011 (only odd numbers).csv'
    # csv_file_name = '../0.5105-0.4895 #11-6011 (only odd numbers).csv'
    # csv_file_name = '../0.52-0.48 #11-6011 (only odd numbers).csv',
    # csv_file_name = '../0.505-0.495 #11-10011 (only odd numbers).csv'
    csv_file_name = ['../0.51-0.49 #11-6011 (only odd numbers).csv',
                     '../0.5105-0.4895 #11-6011 (only odd numbers).csv',
                     '../0.52-0.48 #11-6011 (only odd numbers).csv',
                     '../0.505-0.495 #11-10011 (only odd numbers).csv']

    # Training
    nr = 256
    lr = 8e-5
    train(csv_file_name, neuron_size=nr, learning_rate=lr)

    # Prediction
    csv_data = import_csv(csv_file_name)
    ml_in, ml_out = preprocess_data(csv_data, model_name='lstm')
    pred = predict(ml_in)
    sigmoid_threshold = 0.5
    pred = np.where(pred > sigmoid_threshold, 1, 0)
    pred = binary_to_int(pred)
    print(pred)
    # for par_in in [0.51, 0.52, 0.55]:
        # par_in = 0.51
        # print(par_in)
        # pos_in = np.arange(5, 12, 0.05)
        # pos_in = np.round(np.exp(pos_in))
        # ml_in = [np.asarray([par_in]), pos_in]
        # pred = predict(ml_in)
        # threshold = 0.99999
        # target = pred[pred[:, 1] > threshold, 0]
        # print(pred)
        # if len(target) == 0:
        #     print(f"1.0 reached at: not reached")
        #     print(pred)
        # else:
        #     print(f"1.0 reached at: {target[0]}")

