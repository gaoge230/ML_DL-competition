from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,GridSearchCV

#import xgboost as xgb
#import lightgbm as lgb
# 用线性模型 SGDClassifier
# model_1
def check_model(data, predictors):
    classifier = lambda: SGDClassifier(
        loss='log',
        penalty='elasticnet',
        fit_intercept=True,
        max_iter=100,
        shuffle=True,
        n_jobs=1,
        class_weight=None)
    model = Pipeline(steps=[
        ('ss', StandardScaler()),
        ('en', classifier())
    ])
    # ImportError: [joblib] Attempting to do parallel computing without protecting your import on a system that does not support forking. To use parallel-computing in a script, you must protect your main loop using "if __name__ == '__main__'". Please see the joblib documentation on Parallel for more information
    parameters = {
        'en__alpha': [0.001, 0.01, 0.1],
        'en__l1_ratio': [0.001, 0.01, 0.1]
    }
    folder = StratifiedKFold(n_splits=3, shuffle=True)
    grid_search = GridSearchCV(
        model,
        parameters,
        cv=folder,
        scoring = 'roc_auc',
        n_jobs=-1,
        verbose=1)
    grid_search = grid_search.fit(data[predictors],
                                  data['label'])
    return grid_search

#import lightgbm as lgb
#model2
# def check_model_lgb(data, predictors):
# model = lgb.LGBMClassifier(
#                     learning_rate = 0.01,
#                     boosting_type = 'gbdt',
#                     objective = 'binary',
#                     metric = 'logloss',
#                     max_depth = 5,
#                     sub_feature = 0.7,
#                     num_leaves = 3,
#                     colsample_bytree = 0.7,
#                     n_estimators = 5000,
#                     early_stop = 50,
#                     verbose = -1)
# model.fit(trainSub[predictors], trainSub['label'])
