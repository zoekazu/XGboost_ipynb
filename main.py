#!/usr/bin/env python3
# -*-Coding: utf-8 -*-

# %%
from sklearn import metrics
from sklearn.svm import SVC
import seaborn as sns
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
df_connected = pd.read_pickle('./pandas_df_connected_pin.pkl')
df_connected.head()


# %%
x = df_connected.loc[:, ('width', 'height', 'area', 'pixle_mean', 'pixel_std',
                         'pixel_var', 'pixel_min',
                         'pixel_max', 'pixel_median',)].values
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
# x_true = x[y == True, :].copy()
# x_true
# # %%
# x_true_10 = np.tile(x_true, (11, 1))
# x_true_10
# x_true_10.shape
# # %%
# x = np.append(x, x_true_10, axis=0)
# x.shape
# # %%
# y_true_10 = [True for i in range(x_true_10.shape[0])]
# y_true_10
# # %%
# y = np.append(y, y_true_10)
# y.shape
# %%
# (x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2)
# %%
x_train = x_org.copy()
y_train = y_org.copy()
# %%
data_ratio_test = np.count_nonzero(y_train == False) // np.count_nonzero(y_train) + 1
data_ratio_test
# %%
x_train_weights = np.where(y_train == True, data_ratio_test, 1)
x_train_weights.shape
# %%
x_train.shape
# %%
dtrain = xgb.DMatrix(x_train, label=y_train)
dtrain
# %%
params = {'learning_rate': [0.005, 0.01],
          'n_estimators': [5000, 10000],
          'max_depth': [5, 10],
          'min_child_weight': [1],
          'gamma': [0.4],
          'subsample': [0.85],
          'colsample_bytree': [0.75],
          'reg_alpha': [0.001],
          'objective': ['binary:logistic'],
          'scale_pos_weight': [1],
          'n_jobs': [-1]
          }
# %%
mod = xgb.XGBClassifier()
cv = GridSearchCV(mod, params, scoring='recall', cv=5)
cv.fit(dtrain)

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
ranking = np.argsort(-cv_best.feature_importances_)
# f, ax = plt.subplot(figsize=(11,9))
sns.barplot(x=cv_best.feature_importances_[ranking], y=df_connected.loc[:, ('width', 'height', 'area', 'pixle_mean', 'pixel_std',
                                                                            'pixel_var', 'pixel_min',
                                                                            'pixel_max', 'pixel_median',)].columns[ranking], orient='h')
# ax.set_xlabel('feature importance')
plt.tight_layout()
plt.show()

# %%
ranking
# %%
x_train_rank = df_connected.loc[:, np.array(['width', 'height', 'area', 'pixle_mean', 'pixel_std',
                                             'pixel_var', 'pixel_min',
                                             'pixel_max', 'pixel_median'])[ranking[:4]]].values
x_train_rank
# %%

y_test_cross_pred_list = []
for i in range(1, 10, 2):
    train_list = list(df_connected.index[(df_connected['image_No'] != i) &
                                         (df_connected['image_No'] != i+1)])
    test_list = list(df_connected.index[(df_connected['image_No'] == i) |
                                        (df_connected['image_No'] == i+1)])

    x_train_cross = x_train_rank[train_list, :].copy()
    y_train_cross = y_org[train_list].copy()

    x_test_cross = x_train_rank[test_list, :]
    y_test_cross = y_org[test_list]

    data_ratio = np.count_nonzero(y_train_cross == False) // np.count_nonzero(y_train_cross) + 1
    x_cross_true = x_train_cross[y_train_cross == True, :].copy()
    x_cross_true_aug = np.tile(x_cross_true, (data_ratio, 1))

    x_train_cross = np.append(x_train_cross, x_cross_true_aug, axis=0)
    y_cross_true_aug = [True for i in range(x_cross_true_aug.shape[0])]
    y_train_cross = np.append(y_train_cross, y_cross_true_aug)

    cv_cross = xgb.XGBClassifier(**cv.best_params_)
    cv_cross.fit(x_train_cross, y_train_cross)

    y_test_cross_pred = cv_cross.predict(x_test_cross)
    print(confusion_matrix(y_test_cross, y_test_cross_pred))
    print(classification_report(y_test_cross, y_test_cross_pred))

    y_test_cross_pred_list.extend(list(y_test_cross_pred))


# %%
y_test_cross_pred_list
# %%
df_connected['xgboost_result'] = y_test_cross_pred_list
df_connected.head()

# %%
df_connected.to_pickle('./pandas_df_connected_xgboost.pkl')


# %%
x_train_svm = x_org.copy()
x_train_svm
# %%
y_train_svm = y_org.copy()
y_train_svm

# %%

svm_params = [{'C': [100, 1000], 'kernel': ['linear']}]

svc = SVC()
svm_cv = GridSearchCV(svc, svm_params, scoring='recall', cv=5)
svm_cv.fit(x_train_svm, y_train_svm)
# %%
print(svm_cv.best_params_, svm_cv.best_score_)
# %%
svm_best = SVC(**svm_cv.best_params_)
svm_best.fit(x_train_svm, y_train_svm)
# %%
y_train_svm_pred = svm_best.predict(x_train_svm)

print(confusion_matrix(y_train_svm, y_train_svm_pred))
print(classification_report(y_train_svm, y_train_svm_pred))
# %%


def f_importances(coef, names, top=-1):
    imp = coef
    imp, names = zip(*sorted(list(zip(imp, names)), reverse=True))

    # Show all features
    if top == -1:
        top = len(names)

    plt.barh(range(top), imp[::-1][0:top], align='center')
    plt.yticks(range(top), names[::-1][0:top])
    plt.show()
    return names


# %%
feature_name = ['width', 'height', 'area', 'pixle_mean', 'pixel_std',
                'pixel_var', 'pixel_min',   'pixel_max', 'pixel_median']
feature_name
# %%
svm_best.coef_
# %%
svm_ranking = list(f_importances(abs(svm_best.coef_[0]), feature_name))

# %%
svm_ranking
# %%
# %%
x_train_svm_rank = df_connected.loc[:, np.array(svm_ranking[:3])].values
x_train_svm_rank
# %%

svm_params = [{'C': [100, 1000], 'kernel': ['linear']}]

svc_rank = SVC()
svm_rank_cv = GridSearchCV(svc_rank, svm_params, scoring='recall', cv=5)
svm_rank_cv.fit(x_train_svm_rank, y_train_svm)
# %%
print(svm_rank_cv.best_params_, svm_rank_cv.best_score_)

# %%
svm_best_rank = SVC(**svm_rank_cv.best_params_)
svm_best_rank.fit(x_train_svm_rank, y_train_svm)

# %%
y_train_svm_rank_pred = svm_best_rank.predict(x_train_svm_rank)

print(confusion_matrix(y_train_svm, y_train_svm_rank_pred))
print(classification_report(y_train_svm, y_train_svm_rank_pred))

# %%
