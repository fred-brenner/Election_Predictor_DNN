import pandas as pd
import numpy as np
import os

# from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, StandardScaler, MinMaxScaler
import joblib


def preprocess_data(csv_data: pd.DataFrame, model_name, p=1):
    ml_output = None
    ml_input = None

    if model_name == 'dnn':
        ml_output = csv_data.to_numpy()[:, -1]
        ml_output = ml_output.reshape(-1, 1)
        input1 = csv_data.probability_first_option_wins
        input1 = np.asarray(input1).reshape(-1, 1)
        input2 = csv_data['size'].to_numpy().reshape(-1, 1)
        # ml_input = np.hstack((input1, input2))
        ml_input = [input1, input2]

    elif model_name == 'lstm':
        lstm_len = 100
        ml_output = [23823, 21609, 5953, 40000]
        ml_output = np.asarray(ml_output).reshape(-1, 1)
        input1 = csv_data.probability_first_option_wins.unique()
        input1 = np.asarray(input1).reshape(-1, 1)
        _, indices = np.unique(csv_data.probability_first_option_wins, return_index=True)
        indices.sort()
        input2 = []
        for indi in indices:
            input2.append(csv_data.probability.iloc[indi:indi+lstm_len])
        input2 = np.asarray(input2).reshape(-1, lstm_len, 1)
        ml_input = [input1, input2]

    return ml_input, ml_output

def standardize_data(bev_data, scaler_name='minmax', included=False) -> np.array:
    if not included:
        # Standard scale data for clustering
        if scaler_name.lower() == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_name.lower() == 'maxabs':
            scaler = MaxAbsScaler()
        elif scaler_name.lower() == 'standard':
            scaler = StandardScaler()
        else:
            print(f"Could not interpret scaler: {scaler_name}")
            exit()
        bev_data_sc = scaler.fit_transform(bev_data)
        # save scaler
        scaler_filename = f"ai_analyze/dumps/{scaler_name.lower()}_scaler.save"
        joblib.dump(scaler, scaler_filename)
    else:
        scaler = scaler_name
        bev_data_sc = scaler.transform(bev_data)

    return bev_data_sc, scaler


def class_to_category(bev_data_class):
    cat_names = bev_data_class.to_list()

    # # Category names to class and size names
    # class_names = []
    # size_names = []
    # for cn in cat_names:
    #     splitted = cn.split(' ')
    #     if splitted == 1:
    #         size_names.append('')
    #         class_names.append(splitted[0])
    #     elif splitted == 2:
    #         size_names.append(splitted[0])
    #         class_names.append(splitted[1])
    #     else:
    #         print(f"Error: Too many arguments found for car type: {cn}")
    #         exit()

    # Encode string labels to index
    le = LabelEncoder()
    class_idx = le.fit_transform(cat_names)

    return class_idx


if __name__ == '__main__':
    test = get_data()
    print("")
