import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import plotly.express as px
from sklearn.model_selection import train_test_split
import xgboost as xgb
from catboost import CatBoostRegressor
import tqdm
import lightgbm as lgb
import time


# Press the green button in the gutter to run the script.
def data_calculation_for_train(data, values, count_describe):
    count_describe = values
    # data = pd.DataFrame(pd.read_csv('/home/alex4/Desktop/AidisTech/hotel/data_by_days.csv'))
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
    for i in range(values + 1, len(data)):
        df = pd.DataFrame(data.a[i - count_describe:i - 1].describe())
        df = df.pivot_table(values="a", columns=df.index)
        df2 = pd.DataFrame([list(data.a[i - values:i])], columns=(range(values)))
        df2.index = ["a"]
        df = pd.concat([df, df2], axis=1)
        print(df)

        print(i)
        stat = stat.append(df)

    # df=pd.DataFrame(data.a[14 - 14:14].describe())
    stat = stat.drop(["count"], axis=1)
    print(stat)
    stat.to_csv('/home/alex4/Desktop/AidisTech/final_project/data_for_train.csv', index=False)


def my_plot_and_error(data,test_data):
    # data = pd.DataFrame(pd.read_csv('/home/alex4/Desktop/AidisTech/final_project/data_for_train.csv'))
    print(mean_absolute_percentage_error(data, test_data))
    # plt.plot(data)
    # sns.plot(test_data)
    # sns.show


def test_models(data, times, values, score, test_part):
    res = {'forest': 0, 'GBM': 0, 'xgb': 0, 'cat': 0, 'lgbm': 0}
    data = pd.DataFrame(pd.read_csv('data_for_train.csv'))

    data=data.head(len(data) - 24)
    # data=data.sample(frac=1)
    Y = data[str(values)]
    data = data.drop([str(values)], axis=1)
    # score=0.05
    models = []
    for i in tqdm.tqdm(range(times)):
        #     print()
        #     print(i)
        #     print()
        # regr = RandomForestRegressor(max_depth=10,n_estimators=100)
        X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=test_part, shuffle=True)
        regr = RandomForestRegressor(n_estimators=100, max_depth=5)
        # shuffle=True=>5%
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        error = mean_absolute_percentage_error(y_test, y_pred)
        print("forest  ", error)
        res['forest'] += error
        if error < score:
            models.append(regr)
        regr = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, max_depth=5)
        # shuffle=True=>5%
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        error = mean_absolute_percentage_error(y_test, y_pred)
        print("GBM  ", error)
        res['GBM'] += error
        if error < score:
            models.append(regr)

        # regr = xgb.XGBRegressor(max_depth=5, learning_rate=0.1)
        # # shuffle=True=>5%
        # regr.fit(X_train, y_train)
        # y_pred = regr.predict(X_test)
        # error = mean_absolute_percentage_error(y_test, y_pred)
        # print("xgb  ", error)
        # res['xgb'] += error
        # if error < score:
            # models.append(regr)

        regr = CatBoostRegressor(learning_rate=0.1, depth=5, logging_level='Silent')
        # shuffle=True=>5%
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        error = mean_absolute_percentage_error(y_test, y_pred)
        print("cat  ", error)
        res['cat'] += error
        if error < score:
            models.append(regr)
        regr = lgb.LGBMRegressor(max_depth=5, learning_rate=0.1)
        # shuffle=True=>5%
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        error = mean_absolute_percentage_error(y_test, y_pred)
        print("lgbm  ", error)
        res['lgbm'] += error
        if error < score:
            models.append(regr)
        # mean(cross_val_score(regr, X_train, y_train, cv=10))
    print("размер тестовой выборки= ", y_test.size)
    print("размер обучающей выборки= ", y_train.size)
    return models


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


def stacking(models, data):
    ans = list()
    for model in models:
        ans.append(float( model.predict(data)))
    return sum(ans) / len(ans)


# def predict_n_times(data, values, count_describe, n, model):
def predict_n_times(data, values, count_describe, n, models):
    data=data.head(len(data)-n)
    result= pd.DataFrame(columns=["begin","a"])
    for i in range(len(data) - values, len(data) - values + n):
        tmp = pd.DataFrame(data.a[i: i + values].describe())
        tmp = tmp.drop(["count"], axis=0).T
        #         tmp
        tmp.index = ["w"]
        data_for_predict = pd.DataFrame(list(data.a[i:i + values])).T
        #         print(data_for_predict)
        data_for_predict.index = ["w"]
        data_for_predict = pd.concat([tmp, data_for_predict], axis=1)
        # data_for_predict
#         model.predict(data_for_predict)
#         data = data.append(pd.DataFrame(model.predict(data_for_predict), columns=["a"]), ignore_index=True)
        data = data.append(pd.DataFrame([stacking (models,data_for_predict)], columns=["a"]), ignore_index=True)
        result=result.append(data.tail(1))

    return result


if __name__ == '__main__':
#     data = pd.DataFrame(pd.read_csv('/home/alex4/Desktop/AidisTech/hotel/data_by_days.csv'))
    start_time = time.time()

    data =pd.DataFrame(pd.read_csv('/home/alex4/Desktop/AidisTech/final_project/check_data.csv'))
    print("data was rеad")
    values=24*1
    n=24
    data_calculation_for_train(data,values,1)

    models=test_models(1, times=7, values=values-1,score=0.015, test_part=0.05)
    y_pred=predict_n_times(data,values-1,n=n,models=models,count_describe=False)
    y_test=data.tail(n)
#     print(y_test.info)
#     print(y_pred.info)
    my_plot_and_error(y_test.a,y_pred.a)
    print("--- %s seconds ---" % (time.time() - start_time))