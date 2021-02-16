import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import xlrd
import datetime as dt
import time
import statsmodels.api as sm
import calendar
# sm.tsa.stattools.adfuller(milk)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def plots():
    # sns.relplot(data=data, x='begin', y='A+', kind="line")
    # fig = px.scatter_matrix(data, )
    # fig = px.parallel_coordinates(data, color="A+", labels={"species_id": "A+", "sepal_width": "R-", },
    #                               color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2)
    fig = px.line(data, x='begin', y='A+')
    # sns.set_theme(style="ticks")
    # sns.pairplot(data)
    # plt.show()
    fig.show()
    fig = px.line(data, x='begin', y='R-')
    fig.show()

    fig = px.strip(data, x='A+', y='R-')
    fig.show()
def exelread():
    data = pd.DataFrame(pd.read_excel(r'/home/alex4/Desktop/AidisTech/hotel/2019_01.xls', skiprows=8))
    print(data.info)
    data1 = pd.DataFrame(pd.read_excel('/home/alex4/Desktop/AidisTech/hotel/2019_02.xls', skiprows=8))
    data = data.append(data1)
    for i in range(2, 25):
        print(i)
        data1 = pd.DataFrame(
            pd.read_excel(f'/home/alex4/Desktop/AidisTech/hotel/Гостинница Северная, профиль мощности ({i}).xls',
                          skiprows=8))
        data = data.append(data1)
    print(data.info)
    # data.to_csv('/home/alex4/Desktop/AidisTech/hotel/data.csv')
    # data[['date','begin','end','A+','A-','R+','R-']].to_csv('/home/alex4/Desktop/AidisTech/hotel/data.csv')


# csv format

start_time = time.time()
data = pd.DataFrame(pd.read_csv(r'/home/alex4/Desktop/AidisTech/hotel/data.csv',))
print("--- %s seconds ---" % (time.time() - start_time))
data=data[['begin','A+','R-']]
print(sm.tsa.stattools.adfuller(data['A+']))
data=data.head(10000)
print(data.info)
print("--- %s seconds ---" % (time.time() - start_time))
if __name__ == '__main__':
    print(sm.tsa.stattools.adfuller(data['A+']))
    # exelread()
    # plots()
    # date_time=dt.datetime(date=data['date'])
    # print(date_time)
    # print(data)


    # for i in range(10):
    #     a.append(np.random.random(10))
    # a.append(np.random.random(10))
    # print(type(np.random.random(10))
    # print(data)
