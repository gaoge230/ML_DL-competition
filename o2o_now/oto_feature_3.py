import json
import urllib.request
import pandas as pd
import numpy as np
from datetime import date
import datetime as dt

def get_all_feature(train,test):
    unite = pd.concat([train,test])
    t = unite[['Coupon_id', 'Date_received']].copy()
    t.Date_received = t.Date_received.astype('str')
    t = t.groupby(['Coupon_id'])['Date_received'].agg(lambda x: ':'.join(x)).reset_index()
    t.rename(columns={'Date_received': 'date_receives_all'}, inplace=True)
    t['first_date_received'] = t.date_receives_all.apply(get_first_data_received)
    t.drop(['date_receives_all'], axis=1, inplace=True)
    return t
def get_first_data_received(dates):
    dates = dates.split(':')
    dates_int = []
    if len(dates) > 0:
        for d in dates:
            if d == 'null':
                continue
            else:
                dates_int.append(int(d))
    if len(dates_int) == 0:
        return -1
    else:
        return str(min(dates_int))
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
    data['discount_rate'] = data['Discount_rate'].map(lambda x: x if ':' not in str(x) else(float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))
    # 满减最低消费
    data['min_cost_of_manjian'] = data['Discount_rate'].map(
        lambda x: -1 if ':' not in str(x) else int(str(x).split(':')[0]))
    # 满减减多少
    data['reduce_manjian'] = data['Discount_rate'].map(
        lambda x: -1 if ':' not in str(x) else int(str(x).split(':')[1]))
    # 距离空值填充为：-1
    data['Distance']=data['Distance'].replace('null' ,-1)
    # 判断是否为空距离
    data['null_distance'] = data['Distance'].map(lambda x: 1 if x == -1 else 0)
    data['Coupon_id'] = data['Coupon_id'].map(lambda x : 0 if x =='null' else int(x))
    data['discount_rate'] = data['discount_rate'].map(lambda x : 1.0 if x =='null' else float(x))
    data['Distance'] = data['Distance'].astype(int)
    print(data.info())
    # 时间转换
    data['date_received'] = data['Date_received'].map(lambda x: x if x == 'null' else pd.to_datetime(x, format='%Y%m%d'))
    if 'Date' in data.columns.tolist():
        data['date'] =data['Date'].map(lambda x: x if x == 'null' else pd.to_datetime(x, format='%Y%m%d'))
    return data
def in_holiday(row):
    holiday = set([20160101,20160102,20160103,20160101,20160109,20160110,20160116,20160117,20160123,20160124,20160130,
                   20160131,20160207,20160208,20160209,20160210,20160211,20160212,20160213,20160220,20160221,20160227,20160228,
                   20160305,20160306,20160312,20160313,20160319,20160320,20160326,20160327,
                   20160402,20160403,20160404,20160409,20160410,20160416,20160417,20160423,20160424,20160430,
                   20160501,20160502,20160507,20160508,20160514,20160515,20160521,20160522,20160528,20160529,
                   20160604,20160605,20160609,20160610,20160611,20160618,20160619,20160625,20160626,
                   20160702,20160703,20160709,20160710,20160716,20160717,20160723,20160724,20160730,20160731])
    if row in holiday:
        return 1
    else:
        return 0
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
def get_processData(feature,dataset):
    dataset =prepare(dataset)
    feature = prepare(feature)
    feature = get_label(feature)
    # 关于数据的特征
    dataset = get_dataset_feature(dataset)
    #关于以往历史的特征
    u_feat = get_user_feature(feature)
    print('user_feature finish')
    w_feat = get_merchant_feature(feature)
    print('merchant_feature finish')
    c_feat = get_coupon_feature(feature)
    print('coupon_feature finish')
    #结合
    dataset = pd.merge(dataset, u_feat, on='User_id', how='left')
    dataset.fillna(0, downcast='infer', inplace=True)
    dataset = pd.merge(dataset, w_feat, on='Merchant_id', how='left')
    dataset.fillna(0, downcast='infer', inplace=True)
    dataset = pd.merge(dataset, c_feat, on='Coupon_id', how='left')
    dataset.fillna(0, downcast='infer', inplace=True)

    return dataset
def is_firstlastone(x):
    if x==0:
        return 1
    elif x>0:
        return 0
    else:
        return -1
def get_day_gap_before(s):
    date_received,dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    if len(dates)>0:
        for d in dates:
            #将时间差转化为天数
            this_gap = (dt.date(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))-dt.date(int(d[0:4]),int(d[4:6]),int(d[6:8]))).days
            if this_gap>0:
                gaps.append(this_gap)
    if len(gaps)==0:
        return -1
    else:
        return min(gaps)
def get_day_gap_after(s):
    date_received,dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (dt.datetime(int(d[0:4]),int(d[4:6]),int(d[6:8]))-dt.datetime(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))).days
        if this_gap>0:
            gaps.append(this_gap)
    if len(gaps)==0:
        return -1
    else:
        return min(gaps)
def get_gap_sum(dates):
    dates = dates.split(':')
    dates_int = []
    gap=0
    if len(dates) > 0:
        for d in dates:
            if d == 'null':
                continue
            else:
                dates_int.append(dt.date(int(d[0:4]),int(d[4:6]),int(d[6:8])))

    if len(dates_int) == 0:
        return gap
    else:
        s_dates_int = sorted(dates_int)
        for i in range(len(s_dates_int)):
            if i <= len(s_dates_int)-2:
                gap=gap+int((s_dates_int[i+1]-s_dates_int[i]).days)
        return gap

def get_dataset_feature(dataset):
    data = dataset.copy()
    # 将Coupon_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素为float
    # 同理
    data['week'] = data['Date_received'].map(lambda x: pd.to_datetime(x, format='%Y%m%d').weekday()) # 星期几
    data['Date_received'] = data['Date_received'].map(lambda x: 0 if x=='null' else int(x))
    # 方便特征提取
    data['cnt'] = 1

    feature = data.copy()
    # # 用户领劵数
    # keys = ['User_id']  # 主键
    # prefixs = 'simple_' + '_'.join(keys) + '_'  # 特征名前缀，由'simple'和主键组成
    # pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)  # 透视图，以keys为键，’cnt‘为值，使用len统计出现的次数
    # pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    # # pivot_table后keys会成为index，统计出的特征列会以values即’cnt‘命名，将其改名为特证名前缀+特征意义，并将index还原
    # feature = pd.merge(feature, pivot, on=keys, how='left')
    print(feature.info())
    #最大优惠券接收日期
    keys = ['User_id', 'Coupon_id']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='Date_received', aggfunc=max)
    pivot = pd.DataFrame(pivot).rename(columns={'Date_received': 'max_date_received'}).reset_index()
    print(pivot.info())
    feature = pd.merge(feature, pivot, on=keys, how='left')
    # 最小优惠券接收日期
    keys = ['User_id', 'Coupon_id']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='Date_received', aggfunc=min)
    pivot = pd.DataFrame(pivot).rename(columns={'Date_received': 'min_date_received'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')
    # 这个优惠券最近接受时间
    feature['this_month_user_receive_same_coupon_lastone'] = feature.max_date_received - feature.Date_received.astype(int)
    # 这个优惠券最远接受时间
    feature['this_month_user_receive_same_coupon_firstone'] = feature.Date_received.astype(int) - feature.min_date_received
    feature.this_month_user_receive_same_coupon_lastone = feature.this_month_user_receive_same_coupon_lastone.apply(
        is_firstlastone)
    feature.this_month_user_receive_same_coupon_firstone = feature.this_month_user_receive_same_coupon_lastone.apply(
        is_firstlastone)
    feature.drop(['max_date_received'], axis=1, inplace=True)
    feature.drop(['min_date_received'], axis=1, inplace=True)

    keys = ['User_id', 'Coupon_id']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    t=data[['User_id','Coupon_id','Date_received']].copy()
    t.Date_received = t.Date_received.astype('str')
    t = t.groupby(['User_id', 'Coupon_id'])['Date_received'].agg(lambda x: ':'.join(x)).reset_index()
    #pivot = pd.pivot_table(data, index=keys, values='Date_received', aggfunc=lambda x:':'.join(x))
    t.rename(columns={'Date_received': 'dates'}, inplace=True)
    feature = pd.merge(feature, t, on=['User_id', 'Coupon_id'], how='left')

    feature['date_received_date'] = feature.Date_received.astype('str') + '-' + feature.dates
    feature['day_gap_before'] = feature.date_received_date.apply(get_day_gap_before)
    feature['day_gap_after'] = feature.date_received_date.apply(get_day_gap_after)
    feature['gap_sum'] = feature.dates.apply(get_gap_sum)
    feature['gap_mean']=list(map(lambda x,y : x/len(y.split(':')),feature['gap_sum'],feature['dates']))

    feature.drop(['date_received_date'], axis=1, inplace=True)
    feature.drop(['dates'], axis=1, inplace=True)


    # # 用户领取特定优惠券数(unit_two 中有)
    # keys = ['User_id', 'Coupon_id']
    # prefixs = 'simple_' + '_'.join(keys) + '_'
    # pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    # pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    # feature = pd.merge(feature, pivot, on=keys, how='left')
    # # 用户当天领券数(unit_two 中有)
    # keys = ['User_id', 'Date_received']
    # prefixs = 'simple_' + '_'.join(keys) + '_'
    # pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    # pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    # feature = pd.merge(feature, pivot, on=keys, how='left')
    # 用户当天领取特定优惠券数(unit_three 中有)
    # keys = ['User_id', 'Coupon_id', 'Date_received']
    # prefixs = 'simple_' + '_'.join(keys) + '_'
    # pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    # pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    # feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户是否在同一天重复领取了特定优惠券
    keys = ['User_id', 'Coupon_id', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=lambda x: 1 if len(x) > 1 else 0)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_receive'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')
    # 用户是否在同一天重复领取了特定商户
    keys = ['User_id', 'Merchant_id', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=lambda x: 1 if len(x) > 1 else 0)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_receive'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')


    # # 删除辅助体特征的cnt
    feature.drop(['cnt'], axis=1, inplace=True)
    # 是否为节假日
    feature['is_holiday'] = feature['Date_received'].apply(in_holiday)
    # 离散化星期
    feature['is_weekend'] = feature['week'].map(lambda x: 1 if x == 5 or x == 6 else 0)  # 判断领券日是否为休息日
    feature = pd.concat([feature, pd.get_dummies(feature['week'], prefix='week')], axis=1)
    feature.index = range(len(feature))  # 重置index
    # 离散化距离(prefix 表示以此指定字符为前缀   )
    feature = pd.concat([feature, pd.get_dummies(feature['Distance'], prefix='distance_')], axis=1)
    feature.index = range(len(feature))  # 重置index
    #
    # feature = get_unite_feature_one(feature)
    # feature=get_unite_feature_two(feature)
    # feature=get_unite_feature_three(feature)
    # feature =get_unite_feature_four(feature)
    # feature = get_unite_feature_five(feature)
    return feature
def get_user_feature(dataset):
    data = dataset.copy()
    # 将Coupon_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素为float
    # data['Coupon_id'] = data['Coupon_id'].map(int)  出现cannot convert float NaN to
    #data['Coupon_id'] = data['Coupon_id'].map(lambda x: 0 if x=='null' else int(x))
    # data['Date_received'] = data['Date_received'].map(int)
    # 同理
    data['Date_received'] = data['Date_received'].map(lambda x: 0 if x == 'null' else int(x))
    data['Date'] = data['Date'].map(lambda x: 0 if x=='null' else int(x))
    data['Distance'] = data['Distance'].map(lambda x: 0 if x == 'null' else int(x))
    # 方便特征提取
    data['cnt'] = 1
    #主键
    keys=['User_id']
    #前缀
    prefixs ='field_'+''.join(keys)+'_'
    #返回特征数据集
    u_feat =dataset[keys].drop_duplicates(keep='first')

    # 有优惠券，购买商品条数75382
    #
    # 无优惠券，购买商品条数701602
    #
    # 有优惠券，不购买商品条数977900
    #
    # 无优惠券，不购买商品条数0

    #用户总记录次数
    pivot = pd.pivot_table(data,index=keys,values='cnt',aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'record_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    #缺失值填充为0，最好加上参数downcast='infer',不然可能会改变DataFrame某些列中元素的类型
    u_feat.fillna(0,downcast='infer',inplace=True)
    # 用户总消费次数
    pivot = pd.pivot_table(
        data[data['Date'].map(lambda x: str(x) != '0')],
        index=keys, values='cnt',aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'consume_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户核销数
    pivot = pd.pivot_table(
        data[data['Date'].map(lambda x: str(x) != '0') & data['Date_received'].map(lambda x: str(x) != '0')],
        index=keys, values='cnt',aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_and_consume_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户未核销数,即Date_received不为空，Date为空即未核销的样本  与#用户未消费次数一样
    pivot = pd.pivot_table(
        data[data['Date'].map(lambda x: str(x) == '0')],
        index=keys, values='cnt',aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_and_not_consume_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    #用户无优惠券但购物了
    pivot = pd.pivot_table(
        data[data['Date'].map(lambda x: str(x) != '0')&data['Date_received'].map(lambda x: str(x) == '0')],
        index=keys, values='cnt',aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'not_receive_and_consume_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    #用户收到优惠券数
    pivot = pd.pivot_table(
        data[data['Date_received'].map(lambda x: str(x) != '0')],
        index=keys, values='cnt',aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    #用户核销率
    # u_feat[prefixs+'receive_and_consumer_rate']=list(map(lambda x,y : x/y if y != 0 else 0,u_feat[prefixs+'receive_and_consume_cnt'],u_feat[prefixs+'received_cnt']))
    # #用户消费率
    # u_feat[prefixs + 'consumer_rate'] = list(
    #     map(lambda x,y: x / y if y != 0 else 0, u_feat[prefixs + 'consume_cnt'],
    #         u_feat[prefixs + 'record_cnt']))
    # #用户领券率
    # u_feat[prefixs + 'receive_rate'] = list(
    #     map(lambda x, y: x / y if y != 0 else 0, u_feat[prefixs + 'received_cnt'],
    #         u_feat[prefixs + 'record_cnt']))

    #用户记录来自多少个不同商家
    #以keys为键，‘Merchant_id’为值，使用len统计去重后的商家出现的次数
    pivot = pd.pivot_table(data, index=keys, values='Merchant_id',aggfunc=lambda x : len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'Merchant_id':prefixs+'differ_merchant_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    #用户收到的优惠券来自的商家有多少个
    pivot = pd.pivot_table(data[data['Date_received'].map(lambda x: str(x) != '0')], index=keys, values='Merchant_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'Merchant_id': prefixs + 'receive_differ_merchant_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    #用户消费的优惠券来自的商家有多少个
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != '0')&data['Date_received'].map(lambda x: str(x) != '0')], index=keys, values='Merchant_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'Merchant_id': prefixs + 'receive_and_consume_differ_merchant_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    #用户消费的商家有多少个
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != '0')], index=keys, values='Merchant_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'Merchant_id': prefixs + 'consume_differ_merchant_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    # 用户未收到的优惠券来自的商家有多少个
    pivot = pd.pivot_table(data[data['Date_received'].map(lambda x: str(x) == '0')], index=keys, values='Merchant_id',
                           aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'Merchant_id': prefixs + 'not_receive_differ_merchant_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户的不同优惠券的个数
    pivot = pd.pivot_table(data, index=keys, values='Coupon_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'Coupon_id': prefixs + 'differ_Coupon_id_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    # 用户收到不同优惠券的个数
    pivot = pd.pivot_table(data[data['Date_received'].map(lambda x: str(x) != '0')], index=keys, values='Coupon_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'Coupon_id': prefixs + 'receive_differ_Coupon_id_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    # 用户未收到不同优惠券的个数
    pivot = pd.pivot_table(data[data['Date_received'].map(lambda x: str(x) == '0')], index=keys, values='Coupon_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'Coupon_id': prefixs + 'not_receive_differ_Coupon_id_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    #用户使用优惠券消费
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != '0')&data['Date_received'].map(lambda x: str(x) != '0')], index=keys, values='Coupon_id',
                           aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(
        columns={'Coupon_id': prefixs + 'receive_consume_differ_Coupon_id_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    # 用户消费(总)
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != '0')], index=keys, values='Coupon_id',
                           aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(
        columns={'Coupon_id': prefixs + 'consume_differ_Coupon_id_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    #用户收到未消费的优惠券
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) == '0')&data['Date_received'].map(lambda x: str(x) != '0')], index=keys, values='Coupon_id',
                           aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(
        columns={'Coupon_id': prefixs + 'receive_not_consume_differ_Coupon_id_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)


    # 用户距离
    pivot = pd.pivot_table(
        data,
        index=keys, values='Distance',
        aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(
        columns={'Distance': prefixs + 'distance_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    #用户消费的距离
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != '0')&data['Date_received'].map(lambda x: str(x) != '0')], index=keys, values='Distance',
                           aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(
        columns={'Distance': prefixs + 'consume_distance_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    # 用户未消费的距离
    pivot = pd.pivot_table(
        data[data['Date'].map(lambda x: str(x) == '0')],
        index=keys, values='Distance',
        aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(
        columns={'Distance': prefixs + 'not_consume_distance_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    # 用户收到的的距离
    pivot = pd.pivot_table(
        data[data['Date_received'].map(lambda x: str(x) != '0')],
        index=keys, values='Distance',
        aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(
        columns={'Distance': prefixs + 'receive_distance_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    # 用户未收到的的距离
    pivot = pd.pivot_table(
        data[data['Date_received'].map(lambda x: str(x) == '0')],
        index=keys, values='Distance',
        aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(
        columns={'Distance': prefixs + 'not_receive_distance_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    # 用户收到未消费的距离
    pivot = pd.pivot_table(
        data[data['Date_received'].map(lambda x: str(x) != '0')&data['Date'].map(lambda x: str(x) == '0')],
        index=keys, values='Distance',
        aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(
        columns={'Distance': prefixs + 'receive_not_consume_distance_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(
        data,
        index=keys, values='Distance',
        aggfunc=max)
    pivot = pd.DataFrame(pivot).rename(
        columns={'Distance': prefixs + 'Distance_max'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(
        data,
        index=keys, values='Distance',
        aggfunc=min)
    pivot = pd.DataFrame(pivot).rename(
        columns={'Distance': prefixs + 'Distance_min'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(
        data,
        index=keys, values='Distance',
        aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(
        columns={'Distance': prefixs + 'Distance_mean'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(
        data,
        index=keys, values='Distance',
        aggfunc=np.median)
    pivot = pd.DataFrame(pivot).rename(
        columns={'Distance': prefixs + 'Distance_median'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)


    #用户不同折扣的个数（总）
    pivot = pd.pivot_table(
        data,
        index=keys, values='discount_rate',
        aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(
        columns={'discount_rate': prefixs + 'diff_discount_rate_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    #用户消费的不同折扣的种类
    pivot = pd.pivot_table(
        data[data['Date'].map(lambda x: str(x) != '0')],
        index=keys, values='discount_rate',
        aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(
        columns={'discount_rate': prefixs + 'consume_diff_discount_rate_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    #用户使用优惠券的不同折扣的种类
    pivot = pd.pivot_table(
        data[data['Date_received'].map(lambda x: str(x) != '0') & data['Date'].map(lambda x: str(x) != '0')],
        index=keys, values='discount_rate',
        aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(
        columns={'discount_rate': prefixs + 'receive_consume_diff_discount_rate_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    # 用户收到的不同折扣的种类
    pivot = pd.pivot_table(
        data[data['Date_received'].map(lambda x: str(x) != '0')],
        index=keys, values='discount_rate',
        aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(
        columns={'discount_rate': prefixs + 'receive_consume_diff_discount_rate_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    # 用户收到未使用的不同折扣的种类
    pivot = pd.pivot_table(
        data[data['Date_received'].map(lambda x: str(x) != '0') & data['Date'].map(lambda x: str(x) == '0')],
        index=keys, values='discount_rate',
        aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(
        columns={'discount_rate': prefixs + 'receive_not_consume_diff_discount_rate_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    # 用户未收到的不同折扣的种类
    pivot = pd.pivot_table(
        data[data['Date_received'].map(lambda x: str(x) == '0')],
        index=keys, values='discount_rate',
        aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(
        columns={'discount_rate': prefixs + 'not_receive_diff_discount_rate_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    #用户折扣的（最大、最小、均值、中位数）
    pivot = pd.pivot_table(
        data,
        index=keys, values='discount_rate',
        aggfunc=min)
    pivot = pd.DataFrame(pivot).rename(
        columns={'discount_rate': prefixs + 'discount_rate_min'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(
        data,
        index=keys, values='discount_rate',
        aggfunc=max)
    pivot = pd.DataFrame(pivot).rename(
        columns={'discount_rate': prefixs + 'discount_rate_max'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(
        data,
        index=keys, values='discount_rate',
        aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(
        columns={'discount_rate': prefixs + 'discount_rate_mean'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(
        data,
        index=keys, values='discount_rate',
        aggfunc=np.median)
    pivot = pd.DataFrame(pivot).rename(
        columns={'discount_rate': prefixs + 'discount_rate_median'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    #用户满减的满的（最大、最小、均值、中位数)
    pivot = pd.pivot_table(
        data,
        index=keys, values='min_cost_of_manjian',
        aggfunc=min)
    pivot = pd.DataFrame(pivot).rename(
        columns={'min_cost_of_manjian': prefixs + 'min_cost_of_manjian_min'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(
        data,
        index=keys, values='min_cost_of_manjian',
        aggfunc=max)
    pivot = pd.DataFrame(pivot).rename(
        columns={'min_cost_of_manjian': prefixs + 'min_cost_of_manjian_max'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(
        data,
        index=keys, values='min_cost_of_manjian',
        aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(
        columns={'min_cost_of_manjian': prefixs + 'min_cost_of_manjian_mean'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(
        data,
        index=keys, values='min_cost_of_manjian',
        aggfunc=np.median)
    pivot = pd.DataFrame(pivot).rename(
        columns={'min_cost_of_manjian': prefixs + 'min_cost_of_manjian_median'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    #用户满减的减的（最大、最小、均值、中位数）
    pivot = pd.pivot_table(
        data,
        index=keys, values='reduce_manjian',
        aggfunc=min)
    pivot = pd.DataFrame(pivot).rename(
        columns={'reduce_manjian': prefixs + 'reduce_manjian_mmin'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(
        data,
        index=keys, values='reduce_manjian',
        aggfunc=max)
    pivot = pd.DataFrame(pivot).rename(
        columns={'reduce_manjian': prefixs + 'reduce_manjian_mmax'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(
        data,
        index=keys, values='reduce_manjian',
        aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(
        columns={'reduce_manjian': prefixs + 'reduce_manjian_mmean'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(
        data,
        index=keys, values='reduce_manjian',
        aggfunc=np.median)
    pivot = pd.DataFrame(pivot).rename(
        columns={'reduce_manjian': prefixs + 'reduce_manjian_median'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    #用户

    #用户常去的商家（分类）
    pivot = pd.pivot_table(data, index=keys, values='Merchant_id', aggfunc=lambda x: 2 if len(x)>=6 else 1 if len(x)>=4 else 0)
    pivot = pd.DataFrame(pivot).rename(columns={'Merchant_id': prefixs + 'repair_merchant_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    #用户常使用优惠券的商家（分类）
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'nan')&data['Date_received'].map(lambda x: str(x) != 'nan')], index=keys, values='Merchant_id',
                           aggfunc=lambda x: 2 if len(x) >= 6 else 1 if len(x) >= 4 else 0)
    pivot = pd.DataFrame(pivot).rename(columns={'Merchant_id': prefixs + 'repair_consume_merchant_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    #用户常领取优惠券的商家（分类）
    pivot = pd.pivot_table(
        data[data['Date_received'].map(lambda x: str(x) != 'nan')],
        index=keys, values='Merchant_id',
        aggfunc=lambda x: 2 if len(x) >= 6 else 1 if len(x) >= 4 else 0)
    pivot = pd.DataFrame(pivot).rename(columns={'Merchant_id': prefixs + 'repair_receive_merchant_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')

    # 客户优惠券被消费的时间间隔
    tmp = data[data['label'] == 1]
    tmp['gap'] = (tmp['date'] - tmp['date_received']).map(lambda x: x.total_seconds() / (60 * 60 * 24))
    pivot = pd.pivot_table(tmp, index=keys, values='gap', aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={'gap': prefixs + 'mean_time_gap_15'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(tmp, index=keys, values='gap', aggfunc=max)
    pivot = pd.DataFrame(pivot).rename(columns={'gap': prefixs + 'max_time_gap_15'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(tmp, index=keys, values='gap', aggfunc=min)
    pivot = pd.DataFrame(pivot).rename(columns={'gap': prefixs + 'min_time_gap_15'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(tmp, index=keys, values='gap', aggfunc=np.median)
    pivot = pd.DataFrame(pivot).rename(columns={'gap': prefixs + 'median_time_gap_15'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # #用户15天内核销的商家的数目
    # pivot = pd.pivot_table(data[data['label']==1],index=keys,values='Merchant_id',aggfunc=lambda x:len(set(x)))
    # pivot = pd.DataFrame(pivot).rename(columns={'Merchant_id':prefixs+'receive_and_consume_differ_merchant_cnt_15'}).reset_index()
    # feature = pd.merge(feature, pivot, on=keys, how='left')
    return u_feat
def get_merchant_feature(dataset):
    data = dataset.copy()
    # 同理
    data['Date_received'] = data['Date_received'].map(lambda x: 0 if x == 'null' else int(x))
    data['Date'] = data['Date'].map(lambda x: 0 if x == 'null' else int(x))
    # 方便特征提取
    data['cnt'] = 1
    keys = ['Merchant_id']
    prefixs = 'feature_'+''.join(keys)+''
    m_feat = dataset[keys].drop_duplicates(keep='first')

    #商家的优惠券被领取的次数
    pivot = pd.pivot_table(data[data['Date_received'].map(lambda x : str(x) != '0')],index=keys,values='cnt',aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'received_cnt'}).reset_index()
    m_feat =pd.merge(m_feat,pivot,on=keys,how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    #商家被记录的次数
    pivot = pd.pivot_table(data, index=keys, values='cnt',
                           aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'record_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    #商家被被消费的次数
    pivot = pd.pivot_table(data[data['Date'].map(lambda x : str(x) != '0')], index=keys, values='cnt',
                           aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'consume_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    # 商家被使用优惠券消费的次数
    pivot = pd.pivot_table(data[data['Date_received'].map(lambda x: str(x) != '0')&data['Date'].map(lambda x: str(x) != '0')], index=keys, values='cnt',
                           aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_and_consume_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    # 商家被没有使用优惠券的消费的次数
    pivot = pd.pivot_table(
        data[data['Date_received'].map(lambda x: str(x) == '0')&data['Date'].map(lambda x: str(x) != '0')],
        index=keys, values='cnt',
        aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'notreceived_and_consume_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    # 商家未被消费的次数
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) == '0')], index=keys, values='cnt',
                           aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'notconsume_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    # # #扩展一些比例
    # #商户核销率
    # m_feat[prefixs + 'received_and_consume_cnt'] = list(
    #     map(lambda x, y: x / y if y != 0 else 0, m_feat[prefixs + 'received_and_consume_cnt'],
    #         m_feat[prefixs + 'received_cnt']))
    # # 商户消费率
    # m_feat[prefixs + 'consumer_rate'] = list(
    #     map(lambda x, y: x / y if y != 0 else 0, m_feat[prefixs + 'consume_cnt'],
    #         m_feat[prefixs + 'record_cnt']))
    # # 用户领券率
    # m_feat[prefixs + 'receive_rate'] = list(
    #     map(lambda x, y: x / y if y != 0 else 0, m_feat[prefixs + 'received_cnt'],
    #         m_feat[prefixs + 'record_cnt']))

    #商家优惠券被多少个不同的客户领取
    pivot = pd.pivot_table(data[data['Date_received'].map(lambda x: str(x) != '0')], index=keys, values='User_id',
                           aggfunc=lambda x : len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'User_id': prefixs + 'received_differ_User_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    #商家被多少个不同用户记录
    pivot = pd.pivot_table(data, index=keys, values='User_id',
                           aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'User_id': prefixs + 'differ_User_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    #商家优惠券被多少个不同的客户消费
    pivot = pd.pivot_table(data[data['Date_received'].map(lambda x: str(x) != '0')&data['Date'].map(lambda x: str(x) != '0')], index=keys, values='User_id',
                           aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'User_id': prefixs + 'received_and_consume_differ_User_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    # 商家优惠券未被多少个不同的客户消费
    pivot = pd.pivot_table(
        data[data['Date_received'].map(lambda x: str(x) != '0')&data['Date'].map(lambda x: str(x) == '0')],
        index=keys, values='User_id',
        aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'User_id': prefixs + 'received_and_notconsume_differ_User_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)

    #商家提供的不同优惠券数
    pivot = pd.pivot_table(data, index=keys, values='Coupon_id',
                           aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'Coupon_id': prefixs + 'differ_coupon_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    #商户的距离（最大、最小、均值、中位数）
    pivot = pd.pivot_table(data, index=keys, values='Distance',
                           aggfunc=min)
    pivot = pd.DataFrame(pivot).rename(columns={'Distance': prefixs + 'Distance_min'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(data, index=keys, values='Distance',
                           aggfunc=max)
    pivot = pd.DataFrame(pivot).rename(columns={'Distance': prefixs + 'Distance_max'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(data, index=keys, values='Distance',
                           aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={'Distance': prefixs + 'Distance_mean'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(data, index=keys, values='Distance',
                           aggfunc=np.median)
    pivot = pd.DataFrame(pivot).rename(columns={'Distance': prefixs + 'Distance_median'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    #商户的折扣（最大、最小、均值、中位数（扣率唯一化））
    pivot = pd.pivot_table(
        data,
        index=keys, values='discount_rate',
        aggfunc=min)
    pivot = pd.DataFrame(pivot).rename(
        columns={'discount_rate': prefixs + 'discount_rate_min'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(
        data,
        index=keys, values='discount_rate',
        aggfunc=max)
    pivot = pd.DataFrame(pivot).rename(
        columns={'discount_rate': prefixs + 'discount_rate_max'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(
        data,
        index=keys, values='discount_rate',
        aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(
        columns={'discount_rate': prefixs + 'discount_rate_mean'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(
        data,
        index=keys, values='discount_rate',
        aggfunc=np.median)
    pivot = pd.DataFrame(pivot).rename(
        columns={'discount_rate': prefixs + 'discount_rate_median'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    # 商户的满（）
    pivot = pd.pivot_table(
        data,
        index=keys, values='min_cost_of_manjian',
        aggfunc=min)
    pivot = pd.DataFrame(pivot).rename(
        columns={'min_cost_of_manjian': prefixs + 'min_cost_of_manjian_min'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(
        data,
        index=keys, values='min_cost_of_manjian',
        aggfunc=max)
    pivot = pd.DataFrame(pivot).rename(
        columns={'min_cost_of_manjian': prefixs + 'min_cost_of_manjian_max'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(
        data,
        index=keys, values='min_cost_of_manjian',
        aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(
        columns={'min_cost_of_manjian': prefixs + 'min_cost_of_manjian_mean'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(
        data,
        index=keys, values='min_cost_of_manjian',
        aggfunc=np.median)
    pivot = pd.DataFrame(pivot).rename(
        columns={'min_cost_of_manjian': prefixs + 'min_cost_of_manjian_median'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    # 商户的减（）
    pivot = pd.pivot_table(
        data,
        index=keys, values='reduce_manjian',
        aggfunc=min)
    pivot = pd.DataFrame(pivot).rename(
        columns={'reduce_manjian': prefixs + 'reduce_manjian_min'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(
        data,
        index=keys, values='reduce_manjian',
        aggfunc=max)
    pivot = pd.DataFrame(pivot).rename(
        columns={'reduce_manjian': prefixs + 'reduce_manjian_max'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(
        data,
        index=keys, values='reduce_manjian',
        aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(
        columns={'reduce_manjian': prefixs + 'reduce_manjian_mean'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(
        data,
        index=keys, values='reduce_manjian',
        aggfunc=np.median)
    pivot = pd.DataFrame(pivot).rename(
        columns={'reduce_manjian': prefixs + 'reduce_manjian_median'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)

    #商户优惠券被消费的时间间隔
    tmp = data[data['label'] == 1]
    tmp['gap'] = (tmp['date'] - tmp['date_received']).map(lambda x: x.total_seconds() / (60 * 60 * 24))
    pivot = pd.pivot_table(tmp, index=keys, values='gap', aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={'gap': prefixs + 'marchant_mean_time_gap_15'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(tmp, index=keys, values='gap', aggfunc=max)
    pivot = pd.DataFrame(pivot).rename(columns={'gap': prefixs + 'max_time_gap_15'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(tmp, index=keys, values='gap', aggfunc=min)
    pivot = pd.DataFrame(pivot).rename(columns={'gap': prefixs + 'min_time_gap_15'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    pivot = pd.pivot_table(tmp, index=keys, values='gap', aggfunc=np.median)
    pivot = pd.DataFrame(pivot).rename(columns={'gap': prefixs + 'median_time_gap_15'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)

    return m_feat
def get_coupon_feature(dataset):
    data = dataset.copy()
    # 同理
    data['Date_received'] = data['Date_received'].map(lambda x: 0 if x == 'null' else int(x))
    data['Date'] = data['Date'].map(lambda x: 0 if x == 'null' else int(x))
    # 方便特征提取
    data['cnt'] = 1
    keys = ['Coupon_id']
    prefixs = 'feature_' + '_'.join(keys) + '_'
    c_feat = dataset[keys].drop_duplicates(keep='first')
    #优惠券被领取的次数
    pivot = pd.pivot_table(data,index=keys,values='cnt',aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'received_cnt'}).reset_index()
    c_feat = pd.merge(c_feat,pivot,on=keys,how='left')
    c_feat.fillna(0, downcast='infer', inplace=True)

    #优惠券15天内被核销的次数
    pivot = pd.pivot_table(data[data['Date_received'].map(lambda x: str(x) != '0')&data['Date'].map(lambda x: str(x) != '0')], index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_and_consume_cnt'}).reset_index()
    c_feat = pd.merge(c_feat, pivot, on=keys, how='left')
    c_feat.fillna(0, downcast='infer', inplace=True)

    # #优惠券15天内的被核销率
    c_feat[prefixs + 'received_and_consume_rate']=list(map(lambda x,y:x/y if y!=0 else 0,c_feat[prefixs + 'received_and_consume_cnt'],c_feat[prefixs+'received_cnt']))

    #优惠券15天被核销的平均时间间隔
    tmp=data[data['label']==1]
    tmp['gap']=(tmp['date']-tmp['date_received']).map(lambda x:x.total_seconds()/(60*60*24))
    pivot =pd.pivot_table(tmp,index=keys,values='gap',aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={'gap':prefixs+'consume_mean_time_gap_15'}).reset_index()
    c_feat = pd.merge(c_feat, pivot, on=keys, how='left')
    c_feat.fillna(0, downcast='infer', inplace=True)

    #满减优惠券最低消费的中位数
    pivot = pd.pivot_table(
        data[data['is_manjian']==1],
        index=keys, values='min_cost_of_manjian', aggfunc=np.median)
    pivot = pd.DataFrame(pivot).rename(columns={'min_cost_of_manjian': prefixs + 'median_of_min_cost_of_manjian'}).reset_index()
    c_feat = pd.merge(c_feat, pivot, on=keys, how='left')
    c_feat.fillna(0, downcast='infer', inplace=True)

    #优惠券距离（）



    return c_feat
def get_unite_feature_one(dataset):
    data = dataset.copy()
    data['cnt'] = 1
    feature = data.copy()
    # 用户领劵数
    keys = ['User_id']  # 主键
    prefixs = 'unite_' + '_'.join(keys) + '_'  # 特征名前缀，由'simple'和主键组成
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)  # 透视图，以keys为键，’cnt‘为值，使用len统计出现的次数
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    # pivot_table后keys会成为index，统计出的特征列会以values即’cnt‘命名，将其改名为特证名前缀+特征意义，并将index还原
    feature = pd.merge(feature, pivot, on=keys, how='left')
    # 商户被领券数
    keys = ['Merchant_id']  # 主键
    prefixs = 'unite_' + '_'.join(keys) + '_'  # 特征名前缀，由'simple'和主键组成
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)  # 透视图，以keys为键，’cnt‘为值，使用len统计出现的次数
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    # pivot_table后keys会成为index，统计出的特征列会以values即’cnt‘命名，将其改名为特证名前缀+特征意义，并将index还原
    feature = pd.merge(feature, pivot, on=keys, how='left')
    # 优惠券被领数
    keys = ['Coupon_id']  # 主键
    prefixs = 'unite_' + '_'.join(keys) + '_'  # 特征名前缀，由'simple'和主键组成
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)  # 透视图，以keys为键，’cnt‘为值，使用len统计出现的次数
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    # pivot_table后keys会成为index，统计出的特征列会以values即’cnt‘命名，将其改名为特证名前缀+特征意义，并将index还原
    feature = pd.merge(feature, pivot, on=keys, how='left')
    # 距离被领数
    keys = ['Distance']  # 主键
    prefixs = 'unite_' + '_'.join(keys) + '_'  # 特征名前缀，由'simple'和主键组成
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)  # 透视图，以keys为键，’cnt‘为值，使用len统计出现的次数
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    # pivot_table后keys会成为index，统计出的特征列会以values即’cnt‘命名，将其改名为特证名前缀+特征意义，并将index还原
    feature = pd.merge(feature, pivot, on=keys, how='left')
    # 折扣率被领数
    keys = ['discount_rate']  # 主键
    prefixs = 'unite_' + '_'.join(keys) + '_'  # 特征名前缀，由'simple'和主键组成
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)  # 透视图，以keys为键，’cnt‘为值，使用len统计出现的次数
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    # pivot_table后keys会成为index，统计出的特征列会以values即’cnt‘命名，将其改名为特证名前缀+特征意义，并将index还原
    feature = pd.merge(feature, pivot, on=keys, how='left')
    # 日期被领数
    keys = ['Date_received']  # 主键
    prefixs = 'unite_' + '_'.join(keys) + '_'  # 特征名前缀，由'simple'和主键组成
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)  # 透视图，以keys为键，’cnt‘为值，使用len统计出现的次数
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    # pivot_table后keys会成为index，统计出的特征列会以values即’cnt‘命名，将其改名为特证名前缀+特征意义，并将index还原
    feature = pd.merge(feature, pivot, on=keys, how='left')
    feature.drop(['cnt'], axis=1, inplace=True)
    return feature
def get_unite_feature_two(dataset):
    data = dataset.copy()
    data['cnt'] = 1
    feature = data.copy()

    # 用户领取特定优惠券数
    keys = ['User_id', 'Coupon_id']
    prefixs = 'unite_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户访问特定商户的次数
    keys = ['User_id', 'Merchant_id']
    prefixs = 'unite_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户访问特定折扣率的次数
    keys = ['User_id', 'discount_rate']
    prefixs = 'unite_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户访问特定距离的次数
    keys = ['User_id', 'Distance']
    prefixs = 'unite_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户访问特定日期的次数
    keys = ['User_id', 'Date_received']
    prefixs = 'unite_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')


    #商户的特定优惠券被领取的次数
    keys = ['Merchant_id', 'Coupon_id']
    prefixs = 'unite_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 商户的特定折扣率被领取的次数
    keys = ['Merchant_id', 'discount_rate']
    prefixs = 'unite_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 商户的特定距离被领取的次数
    keys = ['Merchant_id', 'Distance']
    prefixs = 'unite_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 商户的特定日期被领取的次数
    keys = ['Merchant_id', 'Date_received']
    prefixs = 'unite_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')


    # Coupon_id的折扣率的次数
    keys = ['Coupon_id', 'discount_rate']
    prefixs = 'unite_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # Coupon_id的Distance的次数
    keys = ['Coupon_id', 'Distance']
    prefixs = 'unite_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # Coupon_id的Date_received的次数
    keys = ['Coupon_id', 'Date_received']
    prefixs = 'unite_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')


    # discount_rate的Distance的次数
    keys = ['discount_rate', 'Distance']
    prefixs = 'unite_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # discount_rate的Date_received的次数
    keys = ['discount_rate', 'Date_received']
    prefixs = 'unite_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')
    feature.drop(['cnt'], axis=1, inplace=True)
    return feature
def get_unite_feature_three(dataset):
    data = dataset.copy()
    data['cnt'] = 1
    feature = data.copy()

    # 用户当天领取特定商户券数
    keys = ['User_id', 'Merchant_id', 'Date_received']
    prefixs = 'unite_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户领取特定商户特定优惠券数
    keys = ['User_id', 'Merchant_id', 'Coupon_id']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户领取特定商户特定优惠券数
    keys = ['User_id', 'Merchant_id', 'discount_rate']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户领取特定商户特定距离
    keys = ['User_id', 'Merchant_id', 'Distance']
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

    # 用户当领取特定优惠券数特定折扣率
    keys = ['User_id', 'Coupon_id', 'discount_rate']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户当领取特定优惠券数的距离
    keys = ['User_id', 'Coupon_id', 'Distance']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')


    # 用户、折扣率、距离
    keys = ['User_id', 'discount_rate', 'Distance']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户、折扣率、日期
    keys = ['User_id', 'discount_rate', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户、距离、日期
    keys = ['User_id', 'Distance', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')
    feature.drop(['cnt'], axis=1, inplace=True)
    return feature
def get_unite_feature_four(dataset):
    data = dataset.copy()
    data['cnt'] = 1
    feature = data.copy()
    # 用户、商户、券、折扣
    keys = ['User_id', 'Merchant_id', 'Coupon_id','discount_rate']
    prefixs = 'unite_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户、商户、券、距离
    keys = ['User_id', 'Merchant_id', 'Coupon_id', 'Distance']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户、商户、券、日期
    keys = ['User_id', 'Merchant_id', 'Coupon_id', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')


    # 用户、商户、折扣、距离
    keys = ['User_id', 'Merchant_id', 'discount_rate', 'Distance']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户、商户、折扣、日期
    keys = ['User_id', 'Merchant_id', 'discount_rate', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')


    # 用户、商户、距离、日期
    keys = ['User_id', 'Merchant_id', 'Distance', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')


    # 用户、卷、折扣、距离
    keys = ['User_id', 'Coupon_id', 'discount_rate', 'Distance']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户、卷、折扣、日期
    keys = ['User_id', 'Coupon_id', 'discount_rate', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户、卷、距离、日期
    keys = ['User_id', 'Coupon_id', 'Distance', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户、折扣、距离、日期
    keys = ['User_id', 'discount_rate', 'Distance', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')


    # 商户、卷、折扣、距离
    keys = ['Merchant_id', 'Coupon_id', 'discount_rate', 'Distance']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 商户、卷、折扣、日期
    keys = ['Merchant_id', 'Coupon_id', 'discount_rate', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 商户、卷、距离、日期
    keys = ['Merchant_id', 'Coupon_id', 'Distance', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 商户、折扣、距离、日期
    keys = ['Merchant_id', 'discount_rate', 'Distance', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 卷、折扣、距离、日期
    keys = ['Coupon_id', 'discount_rate', 'Distance', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')
    feature.drop(['cnt'], axis=1, inplace=True)
    return feature
def get_unite_feature_five(dataset):
    data = dataset.copy()
    data['cnt'] = 1
    feature = data.copy()
    # 用户、商户、券、折扣、距离
    keys = ['User_id', 'Merchant_id', 'Coupon_id', 'discount_rate','Distance']
    prefixs = 'unite_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户、商户、券、折扣、日期
    keys = ['User_id', 'Merchant_id', 'Coupon_id', 'discount_rate', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户、商户、折扣、距离、日期
    keys = ['User_id', 'Merchant_id', 'discount_rate', 'Distance', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户、卷、折扣、距离、日期
    keys = ['User_id', 'Coupon_id', 'discount_rate', 'Distance', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 商户、卷、折扣、距离、日期
    keys = ['Merchant_id', 'Coupon_id', 'discount_rate', 'Distance', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')
    feature.drop(['cnt'], axis=1, inplace=True)
    return feature


# 读数据集
off_train = pd.read_csv('ccf_offline_stage1_train.csv', header=0, keep_default_na=False)
# 划分区间
off_test = pd.read_csv('ccf_offline_stage1_test_revised.csv', header=0, keep_default_na=False)

first_date_received = get_all_feature(off_train,off_test)
first_date_received['Coupon_id'] = first_date_received['Coupon_id'].map(lambda x : 0 if x =='null' else int(x))
off_test['Coupon_id'] = off_test['Coupon_id'].map(lambda x : 0 if x =='null' else int(x))
off_test=pd.merge(off_test, first_date_received, on='Coupon_id', how='left')
off_test['is_first_date_received']=list(map(lambda x,y: 1 if x != 'null' and y != 'null' and int(x)==int(y) else 0,off_test['Date_received'],off_test['first_date_received']))
off_train['Coupon_id'] = off_train['Coupon_id'].map(lambda x : 0 if x =='null' else int(x))
off_train=pd.merge(off_train, first_date_received, on='Coupon_id', how='left')
off_train['is_first_date_received']=list(map(lambda x,y: 1 if x != 'null' and y != 'null' and int(x)==int(y) else 0,off_train['Date_received'],off_train['first_date_received']))
off_train.drop(['first_date_received'], axis=1, inplace=True)
off_test.drop(['first_date_received'], axis=1, inplace=True)

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
# 测试集特征 :线下数据中领券和用券日期大于3月15日和小于6月30日的
feature3 = off_train[((off_train.Date >= '20160315') & (off_train.Date <= '20160630')) | (
        (off_train.Date == 'null') & (off_train.Date_received >= '20160315') & (
        off_train.Date_received <= '20160630'))]
# dataset3['Coupon_id'] = dataset3['Coupon_id'].astype('object')
# dataset3['Date_received'] = dataset3['Date_received'].astype('object')3=
#print(dataset3.info())
#dataset_3 = get_processData(feature1,dataset1)
dataset3=prepare(dataset3)
dataset3=get_dataset_feature(dataset3)
