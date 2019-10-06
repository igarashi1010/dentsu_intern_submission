import pandas as pd
import numpy as np
import datetime

from sklearn.model_selection import KFold
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)


# 時系列性を考える為，dateカラムを処理してdate_intカラム作成

## 文字列をdatetime型へ
d_arr = np.array([])
for date in df["date"]:
    date_str = date.split("T")[0]
    d = datetime.datetime.strptime(date_str,"%Y%m%d")
    d_arr = np.append(d_arr, d)

## 日数のintに変換する
start = d_arr.min()
date_int_arr = np.array([])
for d in d_arr:
    td = d-start
    date_int_arr = np.append(date_int_arr, td.days)
df["date_int"] = date_int_arr

year_arr = np.array([])
month_arr = np.array([])
day_arr = np.array([])

for d in d_arr: 
    year_arr = np.append(year_arr, d.year)
    month_arr = np.append(month_arr, d.month)
    day_arr = np.append(day_arr, d.day)

df["date_int"] = date_int_arr
df["year"] = year_arr
df["month"] = month_arr
df["day"] = day_arr

## 時系列性の可視化
### 連続的な時系列変化
# sns.lmplot(x="date_int", y="price", data=df)

### 月次の時系列性
# sns.lmplot(x="month", y="price", data=df)

### 日次の時系列変化
# sns.lmplot(x="day", y="price", data=df)

### 年次の時系列変化
# sns.lmplot(x="year", y="price", data=df)


X = df.drop(["id", "price", "date"], axis=1).values
y = df["price"].values

# 交差検定
n_fold = 5
regr = SGDRegressor(max_iter=1000)

maes = np.array([])
rmses = np.array([])
r2_scores = np.array([])
kf = KFold(n_splits = n_fold, shuffle=True)
for train_indice, test_indice in kf.split(X, y):

    X_train, y_train = X[train_indice], y[train_indice]
    X_test, y_test = X[test_indice], y[test_indice]

    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)
    
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    maes = np.append(maes, mae)
    rmses = np.append(rmses, rmse)
    r2_scores = np.append(r2_scores, r2)

print("平均絶対誤差: ", np.average(maes) )
print("二乗平均平方根誤差: ", np.average(rmses) )
print("決定係数: ", np.average(r2_scores) )

#平均絶対誤差:  127995.19156634915
#二乗平均平方根誤差:  201100.980994065
#決定係数:  0.6989335744425274


