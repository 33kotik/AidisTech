import pandas as pd
import plotly.express as px
import seaborn as sns
import numpy as np
import datetime as dt
import statsmodels.api as sm
import calendar
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
from itertools import product


def invboxcox(y, lmbda):
    if lmbda == 0:
        return (np.exp(y))
    else:
        return (np.exp(np.log(lmbda * y + 1) / lmbda))


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

# start_time = time.time()
# data = pd.DataFrame(pd.read_csv(r'/home/alex4/Desktop/AidisTech/hotel/data.csv',',',index_col=['begin'], parse_dates=['begin'], dayfirst=True))
# print("--- %s seconds ---" % (time.time() - start_time))
# data=data[['A+']]
# print(sm.tsa.stattools.adfuller(data['A+']))
# data=data.head(10000)
# print(data.info)
# print("--- %s seconds ---" % (time.time() - start_time))
def convert_to_days():
    data2 = pd.DataFrame(pd.read_csv(r'/home/alex4/Desktop/AidisTech/hotel/a+.csv'))

    # ans = pd.DataFrame()
    data2['begin'] =pd.to_datetime(data2['begin']).dt.date
    value_of_days = []
    time_os_days = []
    line_of_data = data2.shape[0]
    data2.drop(data2.keys()[0],inplace=True,axis=1)
    for i in range(0,int(line_of_data / 48)):
        value_of_days.append(data2.iloc[list(range(48 * i, 48 * (i + 1), 1))]['A+'].sum())
        time_os_days.append(data2['begin'][48 * i])
    new_data = pd.DataFrame(value_of_days)
    new_data.index = time_os_days
    new_data.to_csv('/home/alex4/Desktop/AidisTech/hotel/data_by_days.csv')
    print(new_data)
# def days(data_fr: 'Pandas data type', *column, DAT='Дата'):
#         """ перевод минутного датафрейма в дневной (по дням) """
#
#     if DAT == 'index':
#         days = sorted(pd.to_datetime(list(map(lambda x: x.replace('.', '-') + '-20', {i[:5] for i in data_fr.index})),
#                                      format='%d-%m-%y'))
#     month = data_fr.shape[0]
#     else:
#     month = data_fr[DAT].__len__()
#     days = [list(data_fr['Дата'])[i] for i in range(0, month, 48)]
#     mass_data = pd.DataFrame()
#     for j in column:
#         value_of_days = []
#     for i in range(0, int(month / 48)):
#         value_of_days.append(data_fr[j].iloc[list(range(48 * i, 48 * (i + 1), 1))].sum())
#     mass_data[j] = value_of_days
#     mass_data.index = days
#     return mass_data
    # data2=[d]
    # for i in data:
    #     day = pd.to_datetime((str(i.index))[0:9])
    #     ans[day] += data[i]

    # print(data2.info)
    # return ans


if __name__ == '__main__':
    data = pd.DataFrame(
        pd.read_csv(r'/home/alex4/Desktop/AidisTech/hotel/data_by_days.csv', ',', index_col=['begin'], parse_dates=['begin'],
                    dayfirst=True))
    data=data.head(60)
    # data = data[['A+']]
    # data = data.rename(columns={'begin': 'month', "A+": "a"})
    # convert_to_days()
    # data=data.head(5000)
    print(data.columns)
    print(data.a)
    # data.index=pd.to_datetime(data.index)
    # plt.ylabel('a')
    plt.figure(figsize=(15, 15))
    # sm.tsa.seasonal_decompose(data.a).plot()
    print(sm.tsa.stattools.adfuller(data.a)[1])
    data['a_box'], lmbda = stats.boxcox(data.a)
    # plt.figure(figsize(15, 7))
    # data.a_box.plot()
    plt.ylabel(u'Transformed a')
    print("Оптимальный параметр преобразования Бокса-Кокса: %f" % lmbda)
    print("Критерий Дики-Фуллера: p=%f" % sm.tsa.stattools.adfuller(data.a_box)[1])
    data['a_box_diff'] = data.a - data.a.shift(7)
    # plt.figure(figsize(15, 10))
    # sm.tsa.seasonal_decompose(data.a_box_diff[12:]).plot()
    print("Критерий Дики-Фуллера: p=%f" % sm.tsa.stattools.adfuller(data.a_box_diff[12:])[1])
    # data.a_box_diff.plot()
    # ax = plt.subplot(211)
    # sm.graphics.tsa.plot_acf(data.a_box_diff[13:].values.squeeze(), lags=48, ax=ax)
    # plt.show()
    # ax = plt.subplot(212)
    # sm.graphics.tsa.plot_pacf(data.a_box_diff[13:].values.squeeze(), lags=48, ax=ax)
    # plt.show()
    ps = range(0, 6)
    d = 1
    qs = range(0, 3)
    Ps = range(0, 5)
    D = 1
    Qs = range(0, 2)
    parameters = product(ps, qs, Ps, Qs)
    parameters_list = list(parameters)
    print(len(parameters_list))
    results = []
    best_aic = float("inf")
    warnings.filterwarnings('ignore')
    # for param in parameters_list:
    #     # try except нужен, потому что на некоторых наборах параметров модель не обучается
    #     try:
    #         model = sm.tsa.statespace.SARIMAX(data.a_box, order=(param[0], d, param[1]),
    #                                           seasonal_order=(param[2], D, param[3], 7)).fit(disp=-1)
    #     # выводим параметры, на которых модель не обучается и переходим к следующему набору
    #     except ValueError:
    #         print('wrong parameters:', param)
    #         continue
    #     aic = model.aic
    #     # сохраняем лучшую модель, aic, параметры
    #     if aic < best_aic:
    #         best_model = model
    #         best_aic = aic
    #         best_param = param
    #     results.append([param, model.aic])
    best_model=sm.tsa.statespace.SARIMAX(data.a_box, order=(5, d,2 ),
                                              seasonal_order=(1, D, 0, 7)).fit(disp=-1)
    warnings.filterwarnings('default')
    # result_table = pd.DataFrame(results)
    # result_table.columns = ['parameters', 'aic']
    # print(result_table.sort_values(by='aic', ascending=True).head())
    print(best_model.summary())

    # plt.subplot(211)
    best_model.resid[7:].plot()
    data['model'] = invboxcox(best_model.fittedvalues, lmbda)
    data.a.plot()
    data.model[7:].plot(color='r')
    plt.ylabel('Wine sales')
    plt.show()
    # plt.ylabel(u'Residuals')
    # ax = plt.subplot(212)
    # sm.graphics.tsa.plot_acf(best_model.resid[13:].values.squeeze(), lags=48,)
    # print("Критерий Стьюдента: p=%f" % stats.ttest_1samp(best_model.resid[13:], 0)[1])
    # print("Критерий Дики-Фуллера: p=%f" % sm.tsa.stattools.adfuller(best_model.resid[13:])[1])
    # data.to_csv('/home/alex4/Desktop/AidisTech/hotel/a+.csv')

    # print(sm.tsa.stattools.adfuller(data['A+']))
    # print(data.daily.shift(1))

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
