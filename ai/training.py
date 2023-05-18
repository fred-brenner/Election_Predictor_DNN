import numpy as np
from keras.optimizers import adam_v2
import pandas as pd
import joblib

# from parameter_estimation.read_data import get_data, preprocess_data, standardize_data
from ai.helpers import *
from ai.model import create_tf_model
from parameter_estimation.import_csv import import_csv


model_name = 'dnn'

def train(csv_file_name):
    # import data
    csv_data = import_csv(csv_file_name)
    # bev_data_numeric = preprocess_data(bev_data, dataset='org')

    x_input = bev_data_numeric[x_col]
    y_out = bev_data_numeric[y_col]

    x_input, scaler_x = standardize_data(x_input, 'minmax')
    y_out, scaler_y = standardize_data(y_out, 'minmax')
    joblib.dump(scaler_x, 'ai_analyze/dumps/scaler_x.save')
    joblib.dump(scaler_y, 'ai_analyze/dumps/scaler_y.save')

    # Check Cuda compatible GPU
    if not test_gpu_tf():
        exit()

    # Set training parameters
    learning_rate = 1e-4
    n_epochs = 100
    batch_size = 8
    neuron_size = 128
    loss = 'mse'

    # setup ML model
    model = create_tf_model(model_name, dim_in=x_input.shape[1],
                            dim_out=y_out.shape[1], nr=neuron_size)

    adam = adam_v2.Adam(learning_rate=learning_rate, decay=learning_rate / n_epochs)
    model.compile(loss=loss, optimizer=adam, metrics=['accuracy', 'Hinge'])

    # Train autoencoder
    training = model.fit(x=x_input, y=y_out, epochs=n_epochs,
                         batch_size=batch_size, shuffle=True, verbose=1)

    # Save model
    inp = input("Save model? (y or n)")
    if inp.lower() == 'y':
        save_path = f'ai_analyze/dumps/{model_name}.h5'
        print(f"Saving model at: {save_path}")
        model.save(save_path)

    print("\nFinished Training")


def predict(df_imputed=None):
    # load model
    save_path = 'ai_analyze/dumps/'
    model, _ = load_keras_model(model_name, save_path)
    if model is None:
        exit()

    scaler_x = joblib.load('ai_analyze/dumps/scaler_x.save')
    scaler_y = joblib.load('ai_analyze/dumps/scaler_y.save')

    if df_imputed is None:
        while True:
            x = get_ask_data(x_col)
            x = pd.DataFrame(x, columns=x_col)
            x_sc = scaler_x.transform(x)
            y = model.predict(x_sc)

            y_df = pd.DataFrame(y, columns=y_col)
            y = scaler_y.inverse_transform(y_df)
            y_df = pd.DataFrame(y, columns=y_col).astype(int)
            print(f"Output: \n{y_df.head()}")
    else:
        x_sc = scaler_x.transform(df_imputed)
        y = model.predict(x_sc)
        y_df = pd.DataFrame(y, columns=y_col)
        y = scaler_y.inverse_transform(y_df)
        y_df = pd.DataFrame(y, columns=y_col)
        return y_df, y_col


if __name__ == '__main__':
    train()

    predict()
