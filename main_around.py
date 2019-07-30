#!/usr/bin/env python3
# -*-Coding: utf-8 -*-

# %%
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense
from keras.models import Sequential
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
df_connected = pd.read_pickle('./pandas_df_connected_ignore_pin.pkl')
df_connected.head()


# %%
x = df_connected.loc[:, ('width', 'height', 'area',
                         'pixle_mean', 'pixel_std',
                         'pixel_var', 'pixel_min',
                         'pixel_max', 'pixel_median',
                         'pixle_mean_ar3', 'pixel_std_ar3',
                         'pixel_var_ar3', 'pixel_min_ar3',
                         'pixel_max_ar3', 'pixel_median_ar3',
                         'pixle_mean_ar5', 'pixel_std_ar5',
                         'pixel_var_ar5', 'pixel_min_ar5',
                         'pixel_max_ar5', 'pixel_median_ar5')].values
x_org = x.copy()
x.shape

# %%
y = df_connected.loc[:, 'true'].values
y_org = y.copy()
y.shape
# %%
data_ratio = np.count_nonzero(y == False) // np.count_nonzero(y) + 1
data_ratio
# %%
x_true = x[y == True, :].copy()
x_true
# %%
x_true_10 = np.tile(x_true, (data_ratio, 1))
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
x_train = x.copy()
y_train = y.copy()
# %%
params = {'learning_rate': [0.0005, 0.001],
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
cv.fit(x_train, y_train)

print(cv.best_params_, cv.best_score_)
# %%
cv_best = xgb.XGBClassifier(**cv.best_params_)
cv_best.fit(x_train, y_train)
# %%
y_org_pred = cv_best.predict(x_org)

print(confusion_matrix(y_org, y_org_pred))
print(classification_report(y_org, y_org_pred))

# %%
ranking = np.argsort(-cv_best.feature_importances_)
sns.barplot(
    x=cv_best.feature_importances_[ranking],
    y=df_connected.loc
    [:,
     ('width', 'height', 'area',
                         'pixle_mean', 'pixel_std',
                         'pixel_var', 'pixel_min',
                         'pixel_max', 'pixel_median',
                         'pixle_mean_ar3', 'pixel_std_ar3',
                         'pixel_var_ar3', 'pixel_min_ar3',
                         'pixel_max_ar3', 'pixel_median_ar3',
                         'pixle_mean_ar5', 'pixel_std_ar5',
                         'pixel_var_ar5', 'pixel_min_ar5',
                         'pixel_max_ar5', 'pixel_median_ar5')].columns[ranking],
    orient='h')
plt.tight_layout()
# plt.show()
plt.savefig('xgboost.png')

# %%
ranking
# %%
x_train_rank = df_connected.loc[:, np.array(['width', 'height', 'area',
                         'pixle_mean', 'pixel_std',
                         'pixel_var', 'pixel_min',
                         'pixel_max', 'pixel_median',
                         'pixle_mean_ar3', 'pixel_std_ar3',
                         'pixel_var_ar3', 'pixel_min_ar3',
                         'pixel_max_ar3', 'pixel_median_ar3',
                         'pixle_mean_ar5', 'pixel_std_ar5',
                         'pixel_var_ar5', 'pixel_min_ar5',
                         'pixel_max_ar5', 'pixel_median_ar5'])[ranking[:4]]].values
x_train_rank
# %%

y_test_cross_pred_list = []
for i in range(1, 10, 2):
    train_list = list(df_connected.index[(df_connected['image_No'] != i) &
                                         (df_connected['image_No'] != i+1)])
    test_list = list(df_connected.index[(df_connected['image_No'] == i) |
                                        (df_connected['image_No'] == i+1)])

    x_train_cross = x_train[train_list, :].copy()
    y_train_cross = y_train[train_list].copy()

    x_test_cross = x_train[test_list, :]
    y_test_cross = y_train[test_list]

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
df_connected.to_pickle('./pandas_df_connected_ignore_xgboost.pkl')


# %%
x_train_svm = x_org.copy()
x_train_svm.shape
# %%
y_train_svm = y_org.copy()
y_train_svm.shape

# %%
data_ratio_svm = np.count_nonzero(y_train_svm == False) // np.count_nonzero(y_train_svm) + 1
data_ratio_svm
# %%
x_train_svm_weights = np.where(y_train_svm, data_ratio_svm, 1)
x_train_svm_weights
# %%

svm_params = [{'C': [100, 1000], 'kernel': ['linear']}]

svc = SVC()
svm_cv = GridSearchCV(svc, svm_params, scoring='recall', cv=5)
svm_cv.fit(x_train_svm, y_train_svm, sample_weight=x_train_svm_weights)
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
    plt.savefig('svm_importance.png')
    return names


# %%
feature_name = ['width', 'height', 'area',
                         'pixle_mean', 'pixel_std',
                         'pixel_var', 'pixel_min',
                         'pixel_max', 'pixel_median',
                         'pixle_mean_ar3', 'pixel_std_ar3',
                         'pixel_var_ar3', 'pixel_min_ar3',
                         'pixel_max_ar3', 'pixel_median_ar3',
                         'pixle_mean_ar5', 'pixel_std_ar5',
                         'pixel_var_ar5', 'pixel_min_ar5',
                         'pixel_max_ar5', 'pixel_median_ar5']
feature_name
# %%
svm_best.coef_
# %%
svm_ranking = list(f_importances(abs(svm_best.coef_[0]), feature_name))


# %%
# x_train_svm_rank = df_connected.loc[:, np.array(svm_ranking[:3])].values
# x_train_svm_rank
# # %%

# svm_params = [{'C': [100, 1000], 'kernel': ['linear']}]

# svc_rank = SVC()
# svm_rank_cv = GridSearchCV(svc_rank, svm_params, scoring='recall', cv=5)
# svm_rank_cv.fit(x_train_svm_rank, y_train_svm, sample_weight=x_train_svm_weights)
# # %%
# print(svm_rank_cv.best_params_, svm_rank_cv.best_score_)

# # %%
# svm_best_rank = SVC(**svm_rank_cv.best_params_)
# svm_best_rank.fit(x_train_svm_rank, y_train_svm)

# # %%
# y_train_svm_rank_pred = svm_best_rank.predict(x_train_svm_rank)

# print(confusion_matrix(y_train_svm, y_train_svm_rank_pred))
# print(classification_report(y_train_svm, y_train_svm_rank_pred))


# %%

y_test_cross_pred_list = []
for i in range(1, 10, 2):
    train_list = list(df_connected.index[(df_connected['image_No'] != i) &
                                         (df_connected['image_No'] != i+1)])
    test_list = list(df_connected.index[(df_connected['image_No'] == i) |
                                        (df_connected['image_No'] == i+1)])

    x_train_cross = x_train[train_list, :].copy()
    y_train_cross = y_train[train_list].copy()

    x_test_cross = x_train[test_list, :]
    y_test_cross = y_train[test_list]

    data_ratio = np.count_nonzero(y_train_cross == False) // np.count_nonzero(y_train_cross) + 1
    x_train_svm_weights = np.where(y_train_cross, data_ratio, 1)

    svm_cross = SVC(**svm_cv.best_params_)
    svm_cross.fit(x_train_cross, y_train_cross, sample_weight=x_train_svm_weights)

    y_test_cross_pred = svm_cross.predict(x_test_cross)
    print(confusion_matrix(y_test_cross, y_test_cross_pred))
    print(classification_report(y_test_cross, y_test_cross_pred))

    y_test_cross_pred_list.extend(list(y_test_cross_pred))


# %%
y_test_cross_pred_list
# %%
df_connected['svm_result'] = y_test_cross_pred_list
df_connected.head()

# %%
df_connected[df_connected['true'] == True]

# %%
df_connected.to_pickle('./pandas_df_connected_ignore_analsys.pkl')
# %%
# %%
x_train_nn = x_org.copy()
x_train_nn.shape
# %%
y_train_nn = y_org.copy()
y_train_nn.shape
# %%
data_ratio_nn = np.count_nonzero(y_train_nn == False) // np.count_nonzero(y_train_nn) + 1
data_ratio_nn
# %%
x_train_nn_weights = np.where(y_train_nn, data_ratio_nn, 1)
x_train_nn_weights
# %%


def create_model(optimizer='adam'):
    model = Sequential()
    model.add(
        Dense(
            21,
            input_dim=21,
            kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model


# %%
model = KerasClassifier(build_fn=create_model, verbose=0)
optimizer = ['SGD', 'Adam']
batch_size = [10, 30, 50]
epochs = [10, 50, 100, 1000]
param_grid = dict(optimizer=optimizer, batch_size=batch_size, epochs=epochs)
nn_cv = GridSearchCV(estimator=model, param_grid=param_grid,  scoring='recall', n_jobs=-1, cv=5)
nn_cv.fit(x_train_nn, y_train_nn, sample_weight=x_train_nn_weights)
# %%
print(nn_cv.best_params_, nn_cv.best_score_)
# %%
nn_best = KerasClassifier(build_fn=create_model, verbose=0, **nn_cv.best_params_)
nn_best.fit(x_train_nn, y_train_nn, sample_weight=x_train_nn_weights)
# %%
y_train_nn_pred = nn_best.predict(x_train_nn)

print(confusion_matrix(y_train_nn, y_train_nn_pred))
print(classification_report(y_train_nn, y_train_nn_pred))

# %%
