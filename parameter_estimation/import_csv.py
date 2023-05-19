import numpy as np
import pandas as pd


def import_csv(csv_fn, check_size=True):
    df = pd.read_csv(csv_fn, sep=',', header=0)
    if check_size:
        size_mask = df['size'] % 2 == 1
        df = df[size_mask]
    # print(df)
    return df


if __name__ == '__main__':
    csv_file_name = '../0.51-0.49 #11-111.csv'
    # csv_file_name = '../0.51-0.49 #11-211(only odd numbers).csv'
    df = import_csv(csv_file_name)
    print(df)
