import numpy as np
from keras.optimizers import adam_v2
import pandas as pd
import joblib

from parameter_estimation.read_data import preprocess_data
from ai.helpers import *
from ai.model import create_tf_model
from parameter_estimation.import_csv import import_csv

model_name = 'dnn'

# limit gpu ram usage
conf = tf.compat.v1.ConfigProto()
conf.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=conf)
tf.compat.v1.keras.backend.set_session(sess)


def train(csv_file_name, learning_rate, n_epochs,
          neuron_size=64, save_model=None, model=None, verb_flag=1):
    # import data
    csv_data = import_csv(csv_file_name, check_size=False)
    ml_in, ml_out = preprocess_data(csv_data)

    if len(ml_out) > 1000:
        batch_size = 32
    elif len(ml_out) > 600:
        batch_size = 24
    elif len(ml_out) > 400:
        batch_size = 16
    elif len(ml_out) > 200:
        batch_size = 12
    elif len(ml_out) > 100:
        batch_size = 8
    elif len(ml_out) > 50:
        batch_size = 6
    else:
        batch_size = 4

    # Check Cuda compatible GPU
    if not test_gpu_tf():
        exit()

    # Set training parameters
    # learning_rate = 8e-5
    # n_epochs = 1000
    # batch_size = 16
    # neuron_size = 80
    # loss = 'mean_squared_error'
    loss = 'mean_squared_logarithmic_error'
    # loss = 'mean_absolute_percentage_error'
    # loss = 'mean_absolute_error'
    # loss = 'cosine_similarity'

    # setup ML model
    if model is None:
        model = create_tf_model(model_name, dim_in=[1, 1],
                                dim_out=ml_out.shape[1], nr=neuron_size)

        adam = adam_v2.Adam(learning_rate=learning_rate, decay=learning_rate / n_epochs)
        model.compile(loss=loss, optimizer=adam)
        # model.compile(loss=loss, optimizer=adam, metrics=['mean_squared_logarithmic_error'])

    # Train autoencoder
    training = model.fit(x=ml_in, y=ml_out, epochs=n_epochs,
                         batch_size=batch_size, shuffle=True, verbose=verb_flag)
    if save_model is not None:
        # Save model
        save_path = f'ai/dumps/{model_name}_{save_model}.h5'
        print(f"Saving model at: {save_path}")
        model.save(save_path)

        print("\nFinished Training")
    return model, training


def predict(ml_in=None):
    # load model
    save_path = 'ai/dumps/'
    model_name_load = 'old'
    model, _ = load_keras_model(model_name_load, save_path)
    if model is None:
        exit()

    # if ml_in is None:
    #     while True:
    #         x = get_ask_data(x_col)
    #         x = pd.DataFrame(x, columns=x_col)
    #         x_sc = scaler_x.transform(x)
    #         y = model.predict(x_sc)
    #
    #         y_df = pd.DataFrame(y, columns=y_col)
    #         y = scaler_y.inverse_transform(y_df)
    #         y_df = pd.DataFrame(y, columns=y_col).astype(int)
    #         print(f"Output: \n{y_df.head()}")
    # else:
    # x_sc = scaler_x.transform(df_imputed)
    y = model.predict(ml_in, verbose=0)
    # result = np.vstack((ml_in[:, 1], y[:, 0])).T
    result = np.vstack((ml_in[1], y[:, 0])).T
    # print(result)
    return result
    # y_df = pd.DataFrame(y, columns=y_col)
    # y = scaler_y.inverse_transform(y_df)
    # y_df = pd.DataFrame(y, columns=y_col)
    # return y_df, y_col


if __name__ == '__main__':
    # csv_file_name = '../0.51-0.49 #11-111.csv'
    # csv_file_name = '../0.51-0.49 #11-211(only odd numbers).csv'
    # csv_file_name = '../0.51-0.49 #11-3011.csv'

    # 2P
    # csv_file_name = '../0.52-0.48 #11-6011 (only odd numbers).csv'              # 5.953
    # csv_file_name = '../0.5105-0.4895 #11-6011 (only odd numbers).csv'          # 21.609
    csv_file_name = '../0.51-0.49 #11-6011 (only odd numbers).csv'              # 23.823
    # csv_file_name = '../0.505-0.495 #11-10011 (only odd numbers).csv'           # 95.309

    # 3P
    # csv_file_name = '../0.35, 0.325, 0.325 #12-6012 (only multiples of 3).csv'  # 11.683
    # csv_file_name = '../0.34, 0.33, 0.33 #12-2001 (only multiples of 3).csv'      # ~45.700
    # csv_file_name = '../0.34, 0.33, 0.33 #12-101 (all values) - borda.csv'

    # 4P
    # csv_file_name = '../0.3, 0.25, 0.25, 0.2 #12-1012 (only multiples of 4).csv'  # unknown (~2.500)
    # csv_file_name = '../0.26, 0.25, 0.25, 0.24 #12-1001 (only multiples of 4).csv'  # unknown (>45k?)
    # csv_file_name = '../0.27, 0.26, 0.26, 0.21 #12-1012 (only multiples of 4).csv'  # unknown

    par_in = csv_file_name.split('/')[1].split('-')[0]
    if len(par_in) > 10:
        par_in = par_in.split(',')[0]
    par_in = float(par_in)

    # Training Hyperparameters
    lr = 7.85e-5
    nr = 128
    n_ep = 600
    n_ep_test = 128
    accuracy_test = 0.0016
    i_tries = 10
    verb_number = 0
    threshold = 0.9999

    # Run Training
    result_runs = []
    loss_runs = []
    for runs in range(5):

        run_flag = False
        for i in range(i_tries):
            # Run first training
            model, training = train(csv_file_name, learning_rate=lr, neuron_size=nr,
                                    n_epochs=n_ep_test, save_model=None,
                                    verb_flag=verb_number)
            # Check accuracy mid-time
            if training.history['loss'][-1] < accuracy_test:
                run_flag = True
                break

        if not run_flag:
            print(f"Training did not converge: {par_in} at iteration {runs}")
        else:
            # Run complete training
            print(f"Took {i + 1} iterations to converge.")
            _, training = train(csv_file_name, learning_rate=lr, n_epochs=n_ep,
                                model=model, save_model=f"{par_in}_{runs}",
                                verb_flag=verb_number)

            # Prediction
            print(par_in)
            pos_in = np.arange(5, 28.01, 0.02)
            pos_in = np.round(np.exp(pos_in))
            ml_in = [np.asarray([par_in] * len(pos_in)), pos_in]
            pred = predict(ml_in)
            target = pred[pred[:, 1] > threshold, 0]
            if len(target) == 0:
                print(f"1.0 reached at: not reached")
                print(pred)
                target = [float('nan')]
            else:
                print(f"1.0 reached at: {target[0]} with training loss: {training.history['loss'][-1]}")

            # Add results to list
            result_runs.append(target[0])
            loss_runs.append(training.history['loss'][-1])

    print(result_runs)
    print(loss_runs)
