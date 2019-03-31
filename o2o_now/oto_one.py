# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import csv
import os, sys, pickle
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve

if __name__ == '__main__':

    # C参数keep_default_na=False 将NaN -> null
    # dfoff = pd.read_csv("C:\\Users\\Administrator\\Desktop\\tianchi\\new_o2o\\ccf_offline_stage1_train.csv",keep_default_na=False)
    dfoff = pd.read_csv("off_oto_add_feature_and_labels.csv", keep_default_na=False)
    #dftest = pd.read_csv("C:\\Users\\Administrator\\Desktop\\tianchi\\new_o2o\\ccf_offline_stage1_test_revised.csv",
    #                     keep_default_na=False)
    #dfon = pd.read_csv("C:\\Users\\Administrator\\Desktop\\tianchi\\new_o2o\\ccf_online_stage1_train.csv",keep_default_na=False)



    # data split  训练集，测试集划分
    df =dfoff[dfoff['label'] != -1].copy()
    # train = df[(df['Date_received'] < '20160530')].copy()
    # valid = df[(df['Date_received'] >= '20160530') & (df['Date_received'] <= '20160615')].copy()
    train, valid = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=100)
    print(train['label'].value_counts())
    print(valid['label'].value_counts())

    # 选取训练特征：
    # original_feature = ['discount_rate','discount_type','discount_man', 'discount_jian','distance', 'weekday', 'weekday_type'] + weekdaycols
    original_feature = ['User_id','Merchant_id','discount_rate', 'discount_type', 'discount_man', 'discount_jian', 'distance']
    print(len(original_feature), original_feature)
    predictors = original_feature
    print(predictors)


    # # 用线性模型 SGDClassifier
    # # model_1
    # def check_model(data, predictors):
    #     classifier = lambda: SGDClassifier(
    #         loss='log',
    #         penalty='elasticnet',
    #         fit_intercept=True,
    #         max_iter=100,
    #         shuffle=True,
    #         n_jobs=1,
    #         class_weight=None)
    #     model = Pipeline(steps=[
    #         ('ss', StandardScaler()),
    #         ('en', classifier())
    #     ])
    #     # ImportError: [joblib] Attempting to do parallel computing without protecting your import on a system that does not support forking. To use parallel-computing in a script, you must protect your main loop using "if __name__ == '__main__'". Please see the joblib documentation on Parallel for more information
    #     parameters = {
    #         'en__alpha': [0.001, 0.01, 0.1],
    #         'en__l1_ratio': [0.001, 0.01, 0.1]
    #     }
    #     folder = StratifiedKFold(n_splits=3, shuffle=True)
    #     grid_search = GridSearchCV(
    #         model,
    #         parameters,
    #         cv=folder,
    #         n_jobs=-1,
    #         verbose=1)
    #     grid_search = grid_search.fit(data[predictors],
    #                                   data['label'])
    #     return grid_search

    #保存模型
    import oto_model as oto_model
    if not os.path.isfile('1_model.pkl'):
        model = oto_model.check_model(train, predictors)
        print(model.best_score_)
        print(model.best_params_)
        with open('1_model.pkl', 'wb') as f:
            pickle.dump(model, f)
    else:
        with open('1_model.pkl', 'rb') as f:
            model = pickle.load(f)

    # 预测以及结果评价
    y_valid_pred = model.predict_proba(valid[predictors])
    valid1 = valid.copy()
    valid1['pred_prob'] = y_valid_pred[:, 1]
    valid1.head(2)

    # avgAUC calculation AUG评估算法
    vg = valid1.groupby(['Coupon_id'])
    aucs = []
    for i in vg:
        tmpdf = i[1]
        if len(tmpdf['label'].unique()) != 2:
            continue
        fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
        aucs.append(auc(fpr, tpr))
    print(np.average(aucs))

    # # test prediction for submission
    # y_test_pred = model.predict_proba(dftest[predictors])
    # dftest1 = dftest[['User_id', 'Coupon_id', 'Date_received']].copy()
    # dftest1['label'] = y_test_pred[:, 1]
    # dftest1.to_csv('submit1.csv', index=False, header=False)
    # dftest1.head()







