import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import oto_feature_3
import warnings
import gc

def prepare(dataset):
    #数据预处理阶段
    '''
    ①折扣处理：
        判断折扣是‘满减’（如10:1）还是‘折扣率’如（0.9），新增一列‘is_manjian’表示该信息
        将‘满减’转换为折扣率形式（如10:1转换为0.9），新增一列‘discount_rate’表示该信息
        得到‘满减’的最低消费（如折扣10:1的最低消费为10），新增一列‘min_cost_of_manjian’表示该信息
        扩展：
        得到‘满减’的减多少（如折扣10:1的减为1），新增一列‘reduce_manjian’表示该信息
    ②距离处理：
        将空距离填充为-1（区别于距离0,1,2,3,4,5,6,7,8,9,10）
        判断是否为空距离，新增一列‘null_distance’表示该信息
    ③时间处理：
        将‘Date_received’列中int 或 float类型的元素转换为datetime类型，新增一列‘date_received’表示该信息
        将‘Date’列中int 或 float类型的元素转换为datetime类型，新增一列‘date’表示该信息

    :param dataset:
    :return:data 预处理后的DataFrame类型的数据
    '''

    data = dataset.copy()
    # Discount_rate是否为满减
    data['is_manjian'] = data['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)
    # 满减全部转化为折扣率
    data['discount_rate'] = data['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else(float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))
    # 满减最低消费
    data['min_cost_of_manjian'] = data['Discount_rate'].map(
        lambda x: -1 if ':' not in str(x) else int(str(x).split(':')[0]))
    # 满减减多少
    data['reduce_manjian'] = data['Discount_rate'].map(
        lambda x: -1 if ':' not in str(x) else int(str(x).split(':')[1]))
    # 距离空值填充为：-1
    data['Distance'].fillna(-1, inplace=True)
    # 判断是否为空距离
    data['null_distance'] = data['Distance'].map(lambda x: 1 if x == -1 else 0)
    # 时间转换
    data['date_received'] = pd.to_datetime(data['Date_received'], format='%Y%m%d')
    if 'Date' in data.columns.tolist():
        data['date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
    return data
def get_label(dataset):
    '''领取优惠券后15天内使用的样本为1，否则为0
    Args :
        dataset:DateFrame类型的数据集off_train,包含属性'User_id','Merchant_id','Coupon_id','Discount_rate',
            'Distance','Data_received','Date'
    return:
        打标后的DataFrame型的数据集
    '''
    #数据源
    data = dataset.copy()
    #打标： 领劵后15天内消费为1，否则为0
    data['label'] = list(map(lambda x , y : 1 if x!='null' and y!='null' and (x-y).total_seconds()/(60*60*24) <=15 else 0 , data['date'] , data['date_received']))
    return data
def get_simple_feature(label_field):
    '''提取的5个特征，作为初学实例
    1.‘simple_User_id_receive_cnt’: 用户领劵数；
    2.‘simple_User_id_Coupon_id_received_cnt’:用户领取特定优惠券数;
    3.‘simple_User_id_Date_received_receive_cnt’:用户当天领券数;
    4.‘simple_User_id_Coupon_id_Data_received_cnt’:用户当天领取特定优惠券数；
    5.‘simple_User_id_Coupon_id_Data_received_repeat_receive’:用户是否在同一天重置领取了特定优惠券
    6.离散化星期
    :param label_field: DataFrame类型的数据集
    :return:
            feature：提取完特征后的DataFrame类型的数据集
    '''
    data = label_field.copy()
    # 将Coupon_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素为float
    # data['Coupon_id'] = data['Coupon_id'].map(int)  出现cannot convert float NaN to
    data['Coupon_id'] = data['Coupon_id'].map(lambda x: 0 if pd.isna(x) else int(x))
    # 同理
    data['Date_received'] = data['Date_received'].map(lambda x: 0 if pd.isna(x) else int(x))
    #data['Date'] = data['Date'].map(lambda x: 0 if pd.isna(x) else int(x))
    # data['Distance'] = data['Distance'].map(int)
    # data['Date_received'] = data['Date_received'].map(int)
    # 方便特征提取
    data['cnt'] = 1

    feature = data.copy()
    # 用户领劵数
    keys = ['User_id']  # 主键
    prefixs = 'simple_' + '_'.join(keys) + '_'  # 特征名前缀，由'simple'和主键组成
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)  # 透视图，以keys为键，’cnt‘为值，使用len统计出现的次数
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    # pivot_table后keys会成为index，统计出的特征列会以values即’cnt‘命名，将其改名为特证名前缀+特征意义，并将index还原
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户领取特定优惠券数
    keys = ['User_id', 'Coupon_id']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户当天领券数
    keys = ['User_id', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户当天领取特定优惠券数
    keys = ['User_id', 'Coupon_id', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户是否在同一天重复领取了特定优惠券
    keys = ['User_id', 'Coupon_id', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=lambda x: 1 if len(x) > 1 else 0)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_receive'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 删除辅助体特征的cnt
    feature.drop(['cnt'], axis=1, inplace=True)

    # 离散化星期
    feature['week'] = feature['date_received'].map(lambda x: x.weekday())  # 星期几
    feature['is_weekend'] = feature['week'].map(lambda x: 1 if x == 5 or x == 6 else 0)  # 判断领券日是否为休息日
    feature = pd.concat([feature, pd.get_dummies(feature['week'], prefix='week')], axis=1)
    feature.index = range(len(feature))  # 重置index

    #离散化距离(prefix 表示以此指定字符为前缀   )
    feature = pd.concat([feature, pd.get_dummies(feature['Distance'], prefix='distance_')], axis=1)
    feature.index = range(len(feature))  # 重置index

    return feature

def model_xgb(train,test):
    '''xgboost模型
        调用xgboost模型训练预测

    train: 训练集（包含‘label’列），DataFrame类型的数据集
    test: 测试集或验证集（不包含‘label’列），DataFrame类型的数据集

        result：预测结果，包含属性‘User_id’,'Coupon_id','Date_received','prob',其中'prob'表示预测为1的概率，DataFrame类型的数据集
        feat_importance :特征重要性，包含属性'feature_name','importance',其中’feature_naem‘表示特证名，importance 表示特征重要性
    '''
    #xgb参数
    # params = {'booster': 'gbtree',
    #           'objective': 'binary:logistic',
    #           'eval_metric': 'auc',
    #           'silent': 1,
    #           'eta': 0.01,
    #           'max_depth': 5,
    #           'min_child_weight': 1,
    #           'gamma': 0,
    #           'lambda': 1,
    #           'colsample_bylevel': 0.7,
    #           'colsample_bytree': 0.7,
    #           'subsample': 0.9,
    #           'scale_pos_weight': 1}
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'gamma': 0.1,
              'min_child_weight': 1.1,
              'max_depth': 5,
              'lambda': 10,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'colsample_bylevel': 0.7,
              'eta': 0.01,
              'tree_method': 'gpu_hist',
              'n_gpus': '-1',
              'seed': 0,
              'nthread': 'cpu_jobs',
              'predictor': 'gpu_predictor'
              }

    dtrain = xgb.DMatrix(train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1),label=train['label'])  # 训练集
    dtest = xgb.DMatrix(test.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1))  # 测试集特征（没有label）
    # 训练
    watchlist = [(dtrain, 'train')]
    #model = xgb.train(params, dtrain, num_boost_round=5167, evals=watchlist)  # 迭代次数5167次
    model = xgb.train(params, dtrain, num_boost_round=50, evals=watchlist)
    # 预测
    predict = model.predict(dtest)
    #处理结果
    predict = pd.DataFrame(predict , columns=['prob'])
    result = pd.concat([test[['User_id','Coupon_id','Date_received']],predict],axis=1)
    #特征重要性
    feat_importance = pd.DataFrame(columns=['feature_name','importance'])
    feat_importance['feature_name'] = model.get_score().keys()
    feat_importance['importance'] = model.get_score().values()
    feat_importance.sort_values(['importance'],ascending= False,inplace=True)
    #返回
    return result,feat_importance

def off_evaluate(validate , off_result):
    '''线下验证
    1.评测指标为AUC,但不是直接计算AUC,是对每个‘Coupon_id’单独计算核销预测的AUC值，再对所有优惠券的AUC值求平均作为最终得评价标准
    2.注意计算AUC时标签的真实值必须为二值，所以应先过滤掉全被核销的‘Coupon_id’(该‘Coupon_id’标签的真实值均为1)和全没被核销的‘Coupon_id’（该‘Coupon_id’标签的真实值均为0）
    :param validate:验证集，DataFrame类型的数据集
    :param off_result:验证集的预测结果，DataFrame类型的数据集
    :return:
        auc:线下验证的AUC,float类型
    '''
    evaluate_date =pd.concat([validate[['Coupon_id','label']],off_result[['prob']]],axis = 1)
    aucs = 0
    lens = 0
    for name,group in evaluate_date.groupby('Coupon_id'):
        if len(set(list(group['label'])))==1:   #过滤掉标签真实值全1和全0的Coupon_id值
            continue
        aucs += roc_auc_score(group['label'],group['prob'])
        lens += 1
    auc = aucs/lens
    return auc


if __name__ == '__main__':
    #读数据集
    off_train = pd.read_csv('ccf_offline_stage1_train.csv', header=0, keep_default_na=False)
    off_test = pd.read_csv('ccf_offline_stage1_test_revised.csv', header=0, keep_default_na=False)
    #是否是第一次领劵
    first_date_received = oto_feature_3.get_all_feature(off_train, off_test)
    first_date_received['Coupon_id'] = first_date_received['Coupon_id'].map(lambda x: 0 if x == 'null' else int(x))
    off_test['Coupon_id'] = off_test['Coupon_id'].map(lambda x: 0 if x == 'null' else int(x))
    off_test = pd.merge(off_test, first_date_received, on='Coupon_id', how='left')
    off_test['is_first_date_received'] = list(
        map(lambda x, y: 1 if x != 'null' and y != 'null' and int(x) == int(y) else 0, off_test['Date_received'],
            off_test['first_date_received']))
    off_train['Coupon_id'] = off_train['Coupon_id'].map(lambda x: 0 if x == 'null' else int(x))
    off_train = pd.merge(off_train, first_date_received, on='Coupon_id', how='left')
    off_train['is_first_date_received'] = list(
        map(lambda x, y: 1 if x != 'null' and y != 'null' and int(x) == int(y) else 0, off_train['Date_received'],
            off_train['first_date_received']))
    off_train.drop(['first_date_received'], axis=1, inplace=True)
    off_test.drop(['first_date_received'], axis=1, inplace=True)



    #划分区间
    # 交叉训练集一：收到券的日期大于4月14日和小于5月14日
    dataset1 = off_train[(off_train.Date_received >= '201604014') & (off_train.Date_received <= '20160514')]
    # 交叉训练集一特征：线下数据中领券和用券日期大于1月1日和小于4月13日
    feature1 = off_train[(off_train.Date >= '20160101') & (off_train.Date <= '20160413') | (
            (off_train.Date == 'null') & (off_train.Date_received >= '20160101') & (
            off_train.Date_received <= '20160413'))]

    # 交叉训练集二：收到券的日期大于5月15日和小于6月15日
    dataset2 = off_train[(off_train.Date_received >= '20160515') & (off_train.Date_received <= '20160615')]
    # 交叉训练集二特征：线下数据中领券和用券日期大于2月1日和小于5月14日
    feature2 = off_train[(off_train.Date >= '20160201') & (off_train.Date <= '20160514') | (
            (off_train.Date == 'null') & (off_train.Date_received >= '20160201') & (
            off_train.Date_received <= '20160514'))]

    # 测试集
    dataset3 = off_test
    dataset3['Coupon_id'] = dataset3['Coupon_id'].astype('object')
    dataset3['Date_received'] = dataset3['Date_received'].astype('object')
    # 测试集特征 :线下数据中领券和用券日期大于3月15日和小于6月30日的
    feature3 = off_train[((off_train.Date >= '20160315') & (off_train.Date <= '20160630')) | (
            (off_train.Date == 'null') & (off_train.Date_received >= '20160315') & (
            off_train.Date_received <= '20160630'))]

    dataset_1 =oto_feature_3.get_processData(feature1,dataset1)
    dataset_2 = oto_feature_3.get_processData(feature2,dataset2)
    dataset_3 = oto_feature_3.get_processData(feature3,dataset3)
    dataset_1.drop_duplicates(inplace=True)
    dataset_2.drop_duplicates(inplace=True)
    dataset_12 = pd.concat([dataset_1, dataset_2], axis=0)
    dataset_12=get_label(dataset_12)
    dataset_12['discount_rate'] = dataset_12['discount_rate'].astype(float)
    dataset_12['Distance'] = dataset_12['Distance'].astype(int)
    dataset_3['discount_rate'] = dataset_3['discount_rate'].astype(float)
    dataset_3['Distance'] = dataset_3['Distance'].astype(int)
    off_result, off_feat_importance = model_xgb(dataset_12.drop(['Discount_rate','Date','date_received','date','day_gap_before','day_gap_after'],axis=1), dataset_3.drop(['Discount_rate','date_received','day_gap_before','day_gap_after'],axis=1))

    # # 训练集历史区间、中间区间、标签区间。
    # train_history_field = off_train[
    #     off_train['date_received'].isin(pd.date_range('2016/3/2', periods=60))]  # [20160302,20160501]
    # # train_middle_field = off_train[off_train['date'].isin(pd.date_range('2016/5/1', periods=15))]  # [20160501,20160516]
    # # train_label_field = off_train[
    # #     off_train['date_received'].isin(pd.date_range('2016/5/16', periods=31))]  # [20160516,20160616]
    # # # 验证集历史区间、中间区间、标签区间、
    # validate_history_field = off_train[
    #     off_train['date_received'].isin(pd.date_range('2016/1/16', periods=60))]  # [20160116,20160316]
    # # validate_middle_field = off_train[
    # #     off_train['date'].isin(pd.date_range('2016/3/16', periods=15))]  # [20160316,20160331]
    # # validate_label_field = off_train[
    # #     off_train['date_received'].isin(pd.date_range('2016/3/31', periods=31))]  # [20160616,20160701]
    # # # 测试集历史区间、中间区间、标签区间、
    # test_history_field = off_train[
    #     off_train['date_received'].isin(pd.date_range('2016/4/17', periods=60))]  # [20160417,20160616]
    # # test_middle_field = off_train[off_train['date'].isin(pd.date_range('2016/6/16', periods=15))]  # [20160616,20160701]
    # # test_label_field = off_test.copy()  # [20160701,20160701]
    # #构造训练集、验证集、测试集
    # train,validate,test= oto_feature_3.get_feature(train_history_field,validate_history_field,test_history_field)

    # train = train.drop(['Discount_rate', 'Distance', 'Date', 'date_received', 'date'],axis=1)
    # validate = validate.drop(['Discount_rate', 'Distance', 'Date', 'date_received', 'date'], axis=1)
    # test = test.drop(['Discount_rate', 'Distance', 'Date', 'date_received', 'date'], axis=1)

    #保存训练集、验证集、测试集
    # train.to_csv(r'trian.csv',index=False)
    # validate.to_csv(r'validate.csv', index=False)
    # test.to_csv(r'test.csv', index=False)
    #线下验证
    # off_result,off_feat_importance = model_xgb(train,validate.drop(['label'],axis = 1))
    # auc = off_evaluate(validate,off_result)
    # print('线下验证auc = {}'.format(auc))

    #需要分区间，时间序列，？？？
    # off_test = pd.read_csv('ccf_offline_stage1_test_revised.csv')
    # off_test = prepare(off_test)
    # big_train =pd.concat([train,validate,test],axis=0)
    # off_test_1 = oto_feature_3.get_simple_feature(off_test).drop(['Discount_rate','Distance','date_received'],axis=1)
    # result,feat_importance =model_xgb(big_train,off_test_1)
    # # #最终结果
    # # #result.to_csv(r'base.csv',index=False,header=None)