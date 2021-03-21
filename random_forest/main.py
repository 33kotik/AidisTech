import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
from itertools import product
import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
def stat_calculation():
    data = pd.DataFrame(pd.read_csv('/home/alex4/Desktop/AidisTech/hotel/data_by_days.csv'))
    df = pd.DataFrame(data.a[14 - 14:14].describe())
    # df.append(data.a[14])
    df2 = pd.DataFrame([[data.a[0], data.a[7], data.a[13], data.a[14]]], columns=["14", '7', '1', '0'])
    # df2=df2.pivot_table(index=df2.columns,columns='w')
    df2.index = ["a"]
    print(df2.index)
    df = df.pivot_table(values="a", columns=df.index)
    stat = pd.concat([df, df2], axis=1)
    print(stat)
    # print(data)
    for i in range(15, 762):
        df = pd.DataFrame(data.a[i - 14:i].describe())
        df = df.pivot_table(values="a", columns=df.index)
        df2 = pd.DataFrame([[data.a[i - 14], data.a[i - 7], data.a[i - 1], data.a[i]]], columns=["14", '7', '1', '0'])
        df2.index = ["a"]
        df = pd.concat([df, df2], axis=1)
        print(df)
        print(i)
        stat = stat.append(df)

    # df=pd.DataFrame(data.a[14 - 14:14].describe())
    print(stat)
    stat.to_csv('/home/alex4/Desktop/AidisTech/random_forest/statistic.csv')
if __name__ == '__main__':
    data = pd.DataFrame(pd.read_csv('/home/alex4/Desktop/AidisTech/hotel/data_by_days.csv'))
    
    # print(df.columns)
    # print(df.pivot_table(values="a",columns=df.index))
    # print_hi('PyCharm')
