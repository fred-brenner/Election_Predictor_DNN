import numpy as np
import pandas as pd


def import_csv(csv_fn):
    df = pd.read_csv(csv_file_name, sep=',', header=0)
    # print(df)
    return df


if __name__ == '__main__':
    csv_file_name = '../0.51-0.49 #11-111.csv'
    # csv_file_name = '0.51-0.49 #11-211(only odd numbers).csv'
    df = import_csv(csv_file_name)
    print(df)
