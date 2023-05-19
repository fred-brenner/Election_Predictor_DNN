import numpy as np
import pandas as pd


def import_csv(csv_fn, check_size=True):
    if type(csv_fn) is not list:
        csv_fn = [csv_fn]

    df_all = None
    for fn in csv_fn:
        df = pd.read_csv(fn, sep=',', header=0)
        if check_size:
            size_mask = df['size'] % 2 == 1
            df = df[size_mask]
        if df_all is None:
            df_all = df
        else:
            df_all = df_all.append(df, ignore_index=True)
    # print(df)
    return df_all


if __name__ == '__main__':
    csv_file_name = '../0.51-0.49 #11-111.csv'
    # csv_file_name = '../0.51-0.49 #11-211(only odd numbers).csv'
    df = import_csv(csv_file_name)
    print(df)
