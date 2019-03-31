import numpy as np
import pandas as pd
import csv
import os, sys, pickle
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve


# df['Discount_rate'] = df['Discount_rate'].astype('float')
# #添加类别标识
# for i in xrange(1000):
#     if (pd.isnull(df.iloc[i, 2]) and not pd.isnull(df.iloc[i, 6])):
#         df.iat[i, 6] = -1
#     if (not (pd.isnull(df.iloc[i, 2])) and not (pd.isnull(df.iloc[i, 6]))):
#         df.iat[i, 6] = 1
#     if (not (pd.isnull(df.iloc[i, 2])) and pd.isnull(df.iloc[i, 6])):
#         df.iat[i, 6] = 0
#     if (pd.isnull(df.iloc[i, 2]) and pd.isnull(df.iloc[i, 6])):
#         df.iat[i, 6] = -2
#     #折扣数值化
#     if(not pd.isnull(df.iloc[i,3])):
#         k = df.ix[i, 3].split(':')
#         if(len(k)==2):
#             k1=float(k[0])
#             k2=float(k[1])
#             df.ix[i,3]=1-k2/k1
#
# df=df.fillna(0)
# #df[['Discount_rate']]=df[['Discount_rate']].astype(float)
# df['Discount_rate'] =pd.to_numeric(df['Discount_rate'], errors = ' coerce')
# print df.dtypes
# print df.head(20)


# print df.head(10)
# train_data=[]
# train_lable=[]
#  test_data=[]
# test_lable=[]
# train_data=df.iloc[:1500000,:5]
# train_lable=df.iloc[:1500000,6]
# test_data=df.iloc[1500000:,:5]
# test_lable=df.iloc[1500000:,6]
# #print train_lable[:10]

# result是结果列表
# csvName是存放结果的csv文件名


def saveResult(result, csvName):
    with open(csvName, 'wb') as myFile:
        myWriter = csv.writer(myFile)
        for i in result:
            tmp = []
            tmp.append(i)
            myWriter.writerow(tmp)


# 调用scikit的knn算法包
from sklearn.neighbors import KNeighborsClassifier


def knnClassify(trainData, trainLabel, testData):
    knnClf = KNeighborsClassifier()  # default:k = 5,defined by yourself:KNeighborsClassifier(n_neighbors=10)
    knnClf.fit(trainData, np.array(trainLabel))
    testLabel = knnClf.predict(testData)
    # testlable_proba= knnClf.predict_proba(trainLabel)
    saveResult(testLabel, 'sklearn_knn_Result.csv')
    return testLabel


# 调用scikit的SVM算法包
from sklearn import svm


def svcClassify(trainData, trainLabel, testData):
    svcClf = svm.SVC(
        C=5.0)  # default:C=1.0,kernel = 'rbf'. you can try kernel:‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    svcClf.fit(trainData, np.ravel(trainLabel))
    testLabel = svcClf.predict(testData)
    saveResult(testLabel, 'sklearn_SVC_C=5.0_Result.csv')
    return testLabel


# 调用scikit的朴素贝叶斯算法包,GaussianNB和MultinomialNB
from sklearn.naive_bayes import GaussianNB  # nb for 高斯分布的数据


def GaussianNBClassify(trainData, trainLabel, testData):
    nbClf = GaussianNB()
    nbClf.fit(trainData, np.ravel(trainLabel))
    testLabel = nbClf.predict(testData)
    saveResult(testLabel, 'sklearn_GaussianNB_Result.csv')
    return testLabel


from sklearn.naive_bayes import MultinomialNB  # nb for 多项式分布的数据


def MultinomialNBClassify(trainData, trainLabel, testData):
    nbClf = MultinomialNB(
        alpha=0.1)  # default alpha=1.0,Setting alpha = 1 is called Laplace smoothing, while alpha < 1 is called Lidstone smoothing.
    nbClf.fit(trainData, np.ravel(trainLabel))
    testLabel = nbClf.predict(testData)
    saveResult(testLabel, 'sklearn_MultinomialNB_alpha=0.1_Result.csv')
    return testLabel


def o2o_start():
    df = pd.read_csv("C:\\Users\\Administrator\\Desktop\\tianchi\\new_o2o\\ccf_offline_stage1_train.csv")
    m, n = df.shape
    # print df.head(50)
    # print df.dtypes
    # print m, n
    for i in range(m):
        if (pd.isnull(df.iloc[i, 2]) and not pd.isnull(df.iloc[i, 6])):
            df.iat[i, 6] = -1
        if (not (pd.isnull(df.iloc[i, 2])) and not (pd.isnull(df.iloc[i, 6]))):
            df.iat[i, 6] = 1
        if (not (pd.isnull(df.iloc[i, 2])) and pd.isnull(df.iloc[i, 6])):
            df.iat[i, 6] = 0
        if (pd.isnull(df.iloc[i, 2]) and pd.isnull(df.iloc[i, 6])):
            df.iat[i, 6] = -2
        if (not pd.isnull(df.iloc[i, 3])):
            k = df.ix[i, 3].split(':')
            if (len(k) == 2):
                k1 = float(k[0])
                k2 = float(k[1])
                df.ix[i, 3] = k2 / k1
            else:
                df.ix[i, 3] = 0
    df = df.fillna(0)
    df['Discount_rate'] = pd.to_numeric(df['Discount_rate'], errors=' ignore')
    df.head(30)
    train_data = []
    train_lable = []
    test_data = []
    test_lable = []
    # 训练集
    train_data = df.iloc[:1500000, :5]
    train_lable = df.iloc[:1500000, 6]
    # 测试机
    test_data = df.iloc[1500000:, :5]
    test_lable = df.iloc[1500000:, 6]
    # 使用不同算法
    # result1 = knnClassify(train_data, train_lable, test_data)
    result1 = svcClassify(train_data, train_lable, test_data)
    # result1= GaussianNBClassify(trainData, trainLabel, testData)
    # result4 = MultinomialNBClassify(trainData, trainLabel, testData)
    # 将结果与跟给定的knn_benchmark对比,以result1为例
    test_data_array = np.array(test_data)
    m, n = test_data_array.shape
    # different = 0  # result1中与benchmark不同的label个数，初始化为0
    # for i in xrange(m):
    #     if result1[i] != resultGiven[0, i]:
    #         different += 1
    # print 'result1:%d'%different
    different = 0
    for i in range(m):
        if result1[i] != test_lable[0, i]:
            different += 1
    print('result1:%d' % different)

# o2o_start()