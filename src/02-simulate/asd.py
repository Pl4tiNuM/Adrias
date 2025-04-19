from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
from sklearn.metrics import r2_score

SCENARIO="5-20-3600-1"

df = pd.read_csv("spark_slowdown.dat")
df = df.dropna()
df = df.loc[df['mode'] == 'remote']

params = { 'max_depth': [3, 5, 6, 10, 15, 20],
           'learning_rate': [0.01, 0.1, 0.2, 0.3],
           'subsample': np.arange(0.5, 1.0, 0.1),
           'colsample_bytree': np.arange(0.4, 1.0, 0.1),
           'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
           'n_estimators': [100, 500, 1000],
           'objective': ['reg:squarederror'],
           'tree_method': ['gpu_hist'],
           'seed': [20]}

for b in df['benchmark'].unique():
    print(b)
    bdf = df.loc[df['benchmark'] == b]
    bdf = bdf.drop(["benchmark","t_start","t_end","mode"],axis=1)
    X = bdf.loc[:, bdf.columns != 'latency']
    y = bdf.loc[:, bdf.columns == 'latency']
    X_trn,X_tst,y_trn,y_tst=train_test_split(X,y,test_size=0.1)
    model = XGBRegressor()
    clf = GridSearchCV(estimator=model, 
                    param_grid=params,
                    scoring='neg_mean_squared_error', 
                    verbose=1)
    
    clf.fit(X_trn,y_trn)

    y_pred = clf.predict(X_tst)
    print(r2_score(y_tst,y_pred))
    exit()

    # print(bdf)
    # print(bdf.shape)
    # print(bdf['hist_tx'].corr(bdf['latency']))
    # print("tx:",bdf['exec_tx'].corr(bdf['latency']))
    # print("rx:",bdf['exec_rx'].corr(bdf['latency']))
    # print("lat:",bdf['exec_lat'].corr(bdf['latency']))



# print(df)