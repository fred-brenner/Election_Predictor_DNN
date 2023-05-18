import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, StandardScaler, MinMaxScaler
import joblib


def get_data(dataset='org'):
    if dataset == 'org':
        bev = pd.read_csv("dataset/bev_data_org.csv", sep=';', header=0)
    elif dataset == 'FEV':
        bev = pd.read_excel("FEV-data-Excel.xlsx", header=0)
    else:
        exit()
    return bev


def preprocess_data(bev_data, dataset='org'):
    # if keep_class:
    #     class_idx = class_to_category(bev_data.Class)
    #     bev_data.insert(0, 'class_idx', class_idx)
    if dataset == 'FEV':
        rm_col = ['Car full name', 'Make', 'Model', 'Type of brakes', 'Drive type',
                  'Wheelbase [cm]', 'Length [cm]', 'Width [cm]', 'Height [cm]',
                  'Permissable gross weight [kg]', 'Maximum load capacity [kg]',
                  'Number of seats', 'Number of doors', 'Boot capacity (VDA) [l]']

        bev_data_numeric = bev_data.drop(columns=rm_col)
        pln_to_eur = 0.21
        bev_data_numeric = bev_data_numeric.rename(columns={'Minimal price (gross) [PLN]': 'Price [EUR]',
                                                            'Engine power [KM]': 'Engine [HP]',
                                                            'Maximum torque [Nm]': 'Torque [Nm]',
                                                            'Battery capacity [kWh]': 'Capacity [kWh]',
                                                            'Minimal empty weight [kg]': 'Empty mass [kg]',
                                                            'Maximum speed [kph]': 'Max speed [km/h]',
                                                            'Acceleration 0-100 kph [s]': 'Acc 0-100 km/h [s]',
                                                            'Maximum DC charging power [kW]': 'Charge rate: [kW]',
                                                            'mean - Energy consumption [kWh/100 km]': 'Consumption [kWh/100km]'})
        bev_data_numeric['Price [EUR]'] *= pln_to_eur
        bev_data_numeric = bev_data_numeric.astype({'Price [EUR]': int})
        bev_data_numeric = bev_data_numeric.dropna(axis=0, how='any')

    elif dataset == 'org':
        rm_col = ['Model', 'Drive']
        bev_data_numeric = bev_data.drop(columns=rm_col)
        bev_data_numeric = bev_data_numeric.apply(pd.to_numeric, errors='coerce')
        bev_data_numeric = bev_data_numeric.dropna(axis=0, how='any')

    else:
        exit()

    return bev_data_numeric


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
