import pandas as pd
# from scipy import stats
# import statsmodels.api as sm
# import matplotlib.pyplot as plt
# import warnings
# from itertools import product
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import plotly.express as px
from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_percentage_error
# from sklearn.model_selection import cross_val_score
import xgboost as xgb
from catboost import CatBoostRegressor
import tqdm


# Press the green button in the gutter to run the script.
def data_calculation(values, count_describe):
    data = pd.DataFrame(pd.read_csv('/home/alex4/Desktop/AidisTech/hotel/data_by_days.csv'))
    df = pd.DataFrame(data.a[values - count_describe:values - 1].describe())
    # df.append(data.a[14])

    df2 = pd.DataFrame(
        [list(data.a[0:values])], columns=(range(values)))
    # df2=df2.pivot_table(index=df2.columns,columns='w')
    df2.index = ["a"]
    print(df2.index)
    df = df.pivot_table(values="a", columns=df.index)
    stat = pd.concat([df, df2], axis=1)
    print(stat)
    # print(data)
    for i in range(values + 1, 762):
        df = pd.DataFrame(data.a[i - count_describe:i - 1].describe())
        df = df.pivot_table(values="a", columns=df.index)
        df2 = pd.DataFrame([list(data.a[i - values:i])], columns=(range(values)))
        df2.index = ["a"]
        df = pd.concat([df, df2], axis=1)
        print(df)

        # print(i)
        stat = stat.append(df)

    # df=pd.DataFrame(data.a[14 - 14:14].describe())
    stat = stat.drop(["count"], axis=1)
    print(stat)
    stat.to_csv('/home/alex4/Desktop/AidisTech/random_forest/statistic.csv', index=False)


def my_plot():
    data = pd.DataFrame(pd.read_csv('/home/alex4/Desktop/AidisTech/random_forest/statistic.csv'))
    Y = data["13"]
    data = data.drop(["13"], axis=1)
    regr = RandomForestRegressor(max_depth=20, n_estimators=200)
    X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=0.3, shuffle=True)
    # shuffle=True=>5%
    regr.fit(X_train, y_train)
    y_pred = regr.predict(data)
    print(mean_absolute_percentage_error(Y, y_pred))
    print(mean_absolute_percentage_error(Y, y_pred))
    print("размер тестовой выборки= ", y_test.size)
    print("размер обучающей выборки= ", y_train.size)


def test_models(times):
    data = pd.DataFrame(pd.read_csv('statistic.csv'))
    # data=data.sample(frac=1)
    Y = data["59"]
    data = data.drop(["59"], axis=1)
    for i in tqdm.tqdm(range(times)):
        #     print()
        #     print(i)
        #     print()
        # regr = RandomForestRegressor(max_depth=10,n_estimators=100)
        X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=0.3)
        regr = RandomForestRegressor(n_estimators=200, max_depth=5)
        # shuffle=True=>5%
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        print("forest  ", mean_absolute_percentage_error(y_test, y_pred))
        res['forest'] += mean_absolute_percentage_error(y_test, y_pred)
        regr = GradientBoostingRegressor(learning_rate=0.1, n_estimators=200, max_depth=5)
        # shuffle=True=>5%
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        print("GBM  ", mean_absolute_percentage_error(y_test, y_pred))
        res['GBM'] += mean_absolute_percentage_error(y_test, y_pred)
        regr = xgb.XGBRegressor(max_depth=5, learning_rate=0.1)
        # shuffle=True=>5%
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        print("xgb  ", mean_absolute_percentage_error(y_test, y_pred))
        res['xgb'] += mean_absolute_percentage_error(y_test, y_pred)
        regr = CatBoostRegressor(learning_rate=0.1, depth=5, logging_level='Silent')
        # shuffle=True=>5%
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        print("cat  ", mean_absolute_percentage_error(y_test, y_pred))
        res['cat'] += mean_absolute_percentage_error(y_test, y_pred)

        # mean(cross_val_score(regr, X_train, y_train, cv=10))
    print("размер тестовой выборки= ", y_test.size)
    print("размер обучающей выборки= ", y_train.size)


def read_data():
    data = pd.DataFrame(pd.read_csv('statistic.csv'))
    # data=data.sample(frac=1)
    Y = data["29"]
    data = data.drop(["29"], axis=1)


def predict_n_times(data, values, count_describe, n,model):
    for i in range(len(data)-values,len(data)-values+n):
        tmp=pd.DataFrame(data.a[i: i+ values].describe())
        tmp = tmp.drop(["count"], axis=0).T
#         tmp
        tmp.index = ["w"]
        data_for_predict = pd.DataFrame(list(data.a[i:i + values])).T
#         print(data_for_predict)
        data_for_predict.index=["w"]
        data_for_predict=pd.concat([tmp, data_for_predict], axis=1)
        # data_for_predict
        regr.predict(data_for_predict)
        data=data.append(pd.DataFrame(regr.predict(data_for_predict),columns=["a"]),ignore_index=True)
    return(data)
if __name__ == '__main__':
    res = {'forest': 0, 'GBM': 0, 'xgb': 0, 'cat': 0}
    # data_calculation(values=60,count_describe=7)
    # read_data()
    test_models(times=5)

    print(res)
    # rege=xgb.XGBRegressor()

    # data = pd.DataFrame(pd.read_csv('/home/alex4/Desktop/AidisTech/random_forest/statistic.csv'))
    # data = pd.DataFrame(pd.read_csv('statistic.csv'))
    # data=data.sample(frac=1)
    # Y = data["13"]
    # data = data.drop(["13"], axis=1)
    # regr = CatBoostRegressor(learning_rate=0.1, depth=10,logging_level='Silent')
    # regr=  GradientBoostingRegressor(learning_rate=0.2,n_estimators=200,max_depth=2)
    # xgb.plot_importance(regr)
    # print(data['25%', '50%', '75%', 'max', 'mean', 'min', 'std','14', '7', '1'])
    # trein_data = data.head(600)
    # check_data = data
    # .append(data.tail(200))
    # y = data["0"]
    # check_y = check_data["0"]
    # # '25%', '50%', '75%', 'max', 'mean', 'min', 'std','14', '7', '1']
    # x = pd.DataFrame(trein_data.iloc[:, 1:12])
    # check_x = pd.DataFrame(check_data.iloc[:, 1:12])
    # regr = RandomForestRegressor(max_depth=5, random_sta

    # clf = svm.SVC(kernel='linear').fit(X_train, y_train)
    # print(clf)
    # regr.fit(x, y)

    # y_pred = regr.predict(check_x)
    # print(mean_absolute_percentage_error(check_y, y_pred))
    # y_pred=regr.predict(x)
    # print(mean_absolute_percentage_error(y, y_pred))
# data.drop удалаяеет asix=1
# data,dtypes

# fig = px.line(y=[Y ,y_pred])

# fig.show()
# fig = px.line(data, x='begin', y='R-')
# fig.show()
# fig = px.strip(data, x='A+', y='R-')
# fig.show()
