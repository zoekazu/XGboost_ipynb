#!/usr/bin/env python3
# -*-Coding: utf-8 -*-

# %%
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import xgboost as xgb


# %%
df_connected = pd.read_pickle('./pandas_df_connected.pkl')
df_connected.head()


# %%
x = df_connected.iloc[:, 3:6].values
x_org = x.copy()
x.shape

# %%
y = df_connected.loc[:, 'true'].values
y_org = y.copy()
y.shape
# %%
print('True:', np.count_nonzero(y))
print('False:', np.count_nonzero(y == False))
# %%
x_true = x[y == True, :].copy()
x_true
# %%
x_true_10 = np.tile(x_true, (10, 1))
x_true_10
x_true_10.shape
# %%
x = np.append(x, x_true_10, axis=0)
x.shape
# %%
y_true_10 = [True for i in range(x_true_10.shape[0])]
y_true_10
# %%
y = np.append(y, y_true_10)
y.shape
# %%
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2)

params = {'learning_rate': [0.01, 0.05],
          'n_estimators': [500, 1000, 5000],
          'max_depth': [3, 5, 10],
          'min_child_weight': [1],
          'gamma': [0.4],
          'subsample': [0.85],
          'colsample_bytree': [0.75],
          'reg_alpha': [0.001],
          'objective': ['binary:logistic'],
          'scale_pos_weight': [1],
          'n_jobs': [-1]
          }

mod = xgb.XGBClassifier()
cv = GridSearchCV(mod, params)
cv.fit(x_train, y_train)

print(cv.best_params_, cv.best_score_)
# %%
cv_best = xgb.XGBClassifier(**cv.best_params_)
cv_best.fit(x_train, y_train)
# %%
y_train_pred = cv_best.predict(x_train)
y_test_pred = cv_best.predict(x_test)

print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

# %%
for i in range(1, 10, 2):
    train_list = list(df_connected.index[(df_connected['image_No'] != i) &
                                         (df_connected['image_No'] != i+1)])
    test_list = list(df_connected.index[(df_connected['image_No'] == i) |
                                        (df_connected['image_No'] == i+1)])

    x_train_cross = x_org[train_list, :].copy()
    y_train_cross = y_org[train_list].copy()

    x_test_cross = x_org[test_list, :]
    y_test_cross = y_org[test_list]

    data_ratio = np.count_nonzero(y_train_cross == False) // np.count_nonzero(y_train_cross) + 1

    x_cross_true = x_train_cross[y_train_cross == True, :].copy()
    x_cross_true_aug = np.tile(x_true, (data_ratio, 1))

    x_train_cross = np.append(x_train_cross, x_cross_true_aug, axis=0)
    y_cross_true_aug = [True for i in range(x_cross_true_aug.shape[0])]
    y_train_cross = np.append(y_train_cross, y_cross_true_aug)

    cv_cross = xgb.XGBClassifier(**cv.best_params_)
    cv_cross.fit(x_train_cross, y_train_cross)

    y_test_cross_pred = cv_cross.predict(x_test_cross)
    print(confusion_matrix(y_test_cross, y_test_cross_pred))
    print(classification_report(y_test_cross, y_test_cross_pred))


# %%
