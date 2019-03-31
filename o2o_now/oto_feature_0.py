import json
import urllib.request
import pandas as pd
import numpy as np

#判断是否未节假日(不适合)
def is_holiday_demo(row):
    #date = "20170530"
    server_url = "http://www.easybots.cn/api/holiday.php?d="
    vop_url_request = urllib.request.Request(server_url + row)
    vop_response = urllib.request.urlopen(vop_url_request)
    vop_data = json.loads(vop_response.read())
    #print(vop_data)
    # if vop_data[row] == '0':
    #     print("this day is weekday")
    # elif vop_data[row] == '1':
    #     print('This day is weekend')
    # elif vop_data[row] == '2':
    #     print('This day is holiday')
    # else:
    #     print('Error')
    if vop_data[row] == '0':
        return 0
    else:
        return 1

def is_holiday(dataset):
    dataset['is_holiday']=dataset['Date_received'].apply(in_holiday)
    return dataset

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

def get_feature(trainset,validateset,testset):
    #feature = trainset.copy()
    # # 将Coupon_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素为float
    # # data['Coupon_id'] = data['Coupon_id'].map(int)  出现cannot convert float NaN to
    # feature['Coupon_id'] = feature['Coupon_id'].map(lambda x: 0 if pd.isna(x) else int(x))
    # # 同理
    # feature['Date_received'] = feature['Date_received'].map(lambda x: 0 if pd.isna(x) else int(x))
    # # data['Date'] = data['Date'].map(lambda x: 0 if pd.isna(x) else int(x))
    # # data['Distance'] = data['Distance'].map(int)
    # # data['Date_received'] = data['Date_received'].map(int)

    u_feat = get_user_feature(trainset)
    print('user_feature finish')
    w_feat = get_merchant_feature(trainset)
    print('merchant_feature finish')
    c_feat = get_coupon_feature(trainset)
    print('coupon_feature finish')

    train = pd.merge(trainset, u_feat, on='User_id', how='left')
    train.fillna(0, downcast='infer', inplace=True)
    train = pd.merge(train, w_feat, on='Merchant_id', how='left')
    train.fillna(0, downcast='infer', inplace=True)
    train = pd.merge(train, c_feat, on='Coupon_id', how='left')
    train.fillna(0, downcast='infer', inplace=True)

    validate = pd.merge(validateset, u_feat, on='User_id', how='left')
    validate.fillna(0, downcast='infer', inplace=True)
    validate = pd.merge(validate, w_feat, on='Merchant_id', how='left')
    validate.fillna(0, downcast='infer', inplace=True)
    validate = pd.merge(validate, c_feat, on='Coupon_id', how='left')
    validate.fillna(0, downcast='infer', inplace=True)

    test = pd.merge(testset, u_feat, on='User_id', how='left')
    test.fillna(0, downcast='infer', inplace=True)
    test = pd.merge(test, w_feat, on='Merchant_id', how='left')
    test.fillna(0, downcast='infer', inplace=True)
    test = pd.merge(test, c_feat, on='Coupon_id', how='left')
    test.fillna(0, downcast='infer', inplace=True)

    train =get_simple_feature(train)
    validate =get_simple_feature(validate)
    test =get_simple_feature(test)
    return train,validate,test,u_feat,w_feat,c_feat
def get_feature_all(trainset):
    #feature = trainset.copy()
    # # 将Coupon_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素为float
    # # data['Coupon_id'] = data['Coupon_id'].map(int)  出现cannot convert float NaN to
    # feature['Coupon_id'] = feature['Coupon_id'].map(lambda x: 0 if pd.isna(x) else int(x))
    # # 同理
    # feature['Date_received'] = feature['Date_received'].map(lambda x: 0 if pd.isna(x) else int(x))
    # # data['Date'] = data['Date'].map(lambda x: 0 if pd.isna(x) else int(x))
    # # data['Distance'] = data['Distance'].map(int)
    # # data['Date_received'] = data['Date_received'].map(int)

    u_feat = get_user_feature(trainset)
    print('user_feature finish')
    w_feat = get_merchant_feature(trainset)
    print('merchant_feature finish')
    c_feat = get_coupon_feature(trainset)
    print('coupon_feature finish')


    return u_feat,w_feat,c_feat

def get_simple_feature(dataset):
    '''提取的5个特征，作为初学实例
        1.‘simple_User_id_receive_cnt’: 用户领劵数；(删除)
        2.‘simple_User_id_Coupon_id_received_cnt’:用户领取特定优惠券数;
        3.‘simple_User_id_Date_received_receive_cnt’:用户当天领券数;
        4.‘simple_User_id_Coupon_id_Data_received_cnt’:用户当天领取特定优惠券数；
        5.‘simple_User_id_Coupon_id_Data_received_repeat_receive’:用户是否在同一天重置领取了特定优惠券
        6.离散化星期
        :param label_field: DataFrame类型的数据集
        :return:
                feature：提取完特征后的DataFrame类型的数据集
        '''
    # 将Coupon_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素为float
    # data['Coupon_id'] = data['Coupon_id'].map(int)  出现cannot convert float NaN to
    # data['Date'] = data['Date'].map(lambda x: 0 if pd.isna(x) else int(x))
    # data['Distance'] = data['Distance'].map(int)
    # data['Date_received'] = data['Date_received'].map(int)
    data=dataset.copy()
    data['Coupon_id'] = data['Coupon_id'].map(lambda x: 0 if pd.isna(x) else int(x))
    #获得星期
    data['week'] = data['Date_received'].map(lambda x: pd.to_datetime(x, format='%Y%m%d').weekday())  # 星期几
    #获得月份
    dataset['day_of_month'] = data.Date_received.astype('str').apply(lambda x: int(x[6:8]))
    # 同理
    data['Date_received'] = data['Date_received'].map(lambda x: 0 if pd.isna(x) else int(x))
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

    # 用户领取特定优惠券数
    keys = ['User_id', 'Coupon_id']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')
    feature.fillna(0, downcast='infer', inplace=True)


    # 用户当天领券数
    keys = ['User_id', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')
    feature.fillna(0, downcast='infer', inplace=True)


    # 用户当天领取特定优惠券数
    keys = ['User_id', 'Coupon_id', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')
    feature.fillna(0, downcast='infer', inplace=True)

    # 用户是否在同一天重复领取了特定优惠券
    keys = ['User_id', 'Coupon_id', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=lambda x: 1 if len(x) > 1 else 0)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_receive'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')
    feature.fillna(0, downcast='infer', inplace=True)
    # 删除辅助体特征的cnt
    feature.drop(['cnt'], axis=1, inplace=True)

    # 是否为节假日
    feature['is_holiday']=dataset['Date_received'].apply(in_holiday)
    # 离散化星期
    feature['is_weekend'] = feature['week'].map(lambda x: 1 if x == 5 or x == 6 else 0)  # 判断领券日是否为休息日
    feature = pd.concat([feature, pd.get_dummies(feature['week'], prefix='week')], axis=1)
    feature.index = range(len(feature))  # 重置index
    # 离散化距离(prefix 表示以此指定字符为前缀   )feature.index = range(len(feature))  # 重置index
    feature = pd.concat([feature, pd.get_dummies(feature['Distance'], prefix='distance_')], axis=1)
    feature.index = range(len(feature))  # 重置index
    return feature

def get_user_feature(dataset):
    data = dataset.copy()
    # 将Coupon_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素为float
    # data['Coupon_id'] = data['Coupon_id'].map(int)  出现cannot convert float NaN to
    data['Coupon_id'] = data['Coupon_id'].map(lambda x: 0 if pd.isna(x) else int(x))
    # 同理
    data['Date_received'] = data['Date_received'].map(lambda x: 0 if pd.isna(x) else int(x))
    data['Date'] = data['Date'].map(lambda x: 0 if pd.isna(x) else int(x))
    # data['Distance'] = data['Distance'].map(int)
    # data['Date_received'] = data['Date_received'].map(int)
    # 方便特征提取
    data['cnt'] = 1

    #主键
    keys=['User_id']
    #前缀
    prefixs ='field_'+'_'.join(keys)+'_'
    #返回特征数据集
    u_feat =data[keys].drop_duplicates(keep='first')

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
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != '0')],index=keys, values='cnt',aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'consume_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户核销数
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != '0') & data['Date_received'].map(lambda x: str(x) != '0')],index=keys, values='cnt',aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_and_consume_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户未核销数,即Date_received不为空，Date为空即未核销的样本  与#用户未消费次数一样
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) == '0')],index=keys, values='cnt',aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_and_not_consume_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    #用户无优惠券但购物了
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != '0')&data['Date_received'].map(lambda x: str(x) == '0')],index=keys, values='cnt',aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'not_receive_and_consume_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    #用户收到优惠券数
    pivot = pd.pivot_table(data[data['Date_received'].map(lambda x: str(x) != '0')],index=keys, values='cnt',aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # #用户核销率
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



    # #用户常去的商家（分类）
    # pivot = pd.pivot_table(data, index=keys, values='Merchant_id', aggfunc=lambda x: 2 if len(x)>=6 else 1 if len(x)>=4 else 0)
    # pivot = pd.DataFrame(pivot).rename(columns={'Merchant_id': prefixs + 'repair_merchant_cnt'}).reset_index()
    # feature = pd.merge(feature, pivot, on=keys, how='left')
    # #用户常使用优惠券的商家（分类）
    # pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'nan')&data['Date_received'].map(lambda x: str(x) != 'nan')], index=keys, values='Merchant_id',
    #                        aggfunc=lambda x: 2 if len(x) >= 6 else 1 if len(x) >= 4 else 0)
    # pivot = pd.DataFrame(pivot).rename(columns={'Merchant_id': prefixs + 'repair_consume_merchant_cnt'}).reset_index()
    # feature = pd.merge(feature, pivot, on=keys, how='left')
    # #用户常领取优惠券的商家（分类）
    # pivot = pd.pivot_table(
    #     data[data['Date_received'].map(lambda x: str(x) != 'nan')],
    #     index=keys, values='Merchant_id',
    #     aggfunc=lambda x: 2 if len(x) >= 6 else 1 if len(x) >= 4 else 0)
    # pivot = pd.DataFrame(pivot).rename(columns={'Merchant_id': prefixs + 'repair_receive_merchant_cnt'}).reset_index()
    # feature = pd.merge(feature, pivot, on=keys, how='left')
    #

    # #用户15天内核销的商家的数目
    # pivot = pd.pivot_table(data[data['label']==1],index=keys,values='Merchant_id',aggfunc=lambda x:len(set(x)))
    # pivot = pd.DataFrame(pivot).rename(columns={'Merchant_id':prefixs+'receive_and_consume_differ_merchant_cnt_15'}).reset_index()
    # feature = pd.merge(feature, pivot, on=keys, how='left')

    return u_feat

def get_merchant_feature(dataset):
    data = dataset.copy()
    # 将Coupon_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素为float
    # data['Coupon_id'] = data['Coupon_id'].map(int)  出现cannot convert float NaN to
    data['Coupon_id'] = data['Coupon_id'].map(lambda x: 0 if pd.isna(x) else int(x))
    # 同理
    data['Date_received'] = data['Date_received'].map(lambda x: 0 if pd.isna(x) else int(x))
    data['Date'] = data['Date'].map(lambda x: 0 if pd.isna(x) else int(x))
    # data['Distance'] = data['Distance'].map(int)
    # data['Date_received'] = data['Date_received'].map(int)
    # 方便特征提取
    data['cnt'] = 1
    keys = ['Merchant_id']
    prefixs = 'feature_'+'_'.join(keys)+'_'
    m_feat = data[keys].drop_duplicates(keep='first')

    #商家的优惠券被领取的次数
    pivot = pd.pivot_table(data[data['Date_received'].map(lambda x : str(x) != '0')],index=keys,values='cnt',aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'received_cnt'}).reset_index()
    m_feat =pd.merge(m_feat,pivot,on=keys,how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    #商家被记录的次数
    pivot = pd.pivot_table(data, index=keys, values='cnt',aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'record_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    #商家被被消费的次数
    pivot = pd.pivot_table(data[data['Date'].map(lambda x : str(x) != '0')], index=keys, values='cnt',aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'consume_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    # 商家被使用优惠券消费的次数
    pivot = pd.pivot_table(data[data['Date_received'].map(lambda x: str(x) != '0')&data['Date'].map(lambda x: str(x) != '0')], index=keys, values='cnt',aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_and_consume_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    # 商家被没有使用优惠券的消费的次数
    pivot = pd.pivot_table(data[data['Date_received'].map(lambda x: str(x) == '0')&data['Date'].map(lambda x: str(x) != '0')],index=keys, values='cnt',aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'not_received_and_consume_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    # 商家未被消费的次数
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) == '0')], index=keys, values='cnt',aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'not_consume_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    # #扩展一些比例
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
    pivot = pd.pivot_table(data[data['Date_received'].map(lambda x: str(x) != '0')], index=keys, values='User_id',aggfunc=lambda x : len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'User_id': prefixs + 'received_differ_User_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    #商家被多少个不同用户记录
    pivot = pd.pivot_table(data, index=keys, values='User_id',aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'User_id': prefixs + 'differ_User_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    #商家优惠券被多少个不同的客户消费
    pivot = pd.pivot_table(data[data['Date_received'].map(lambda x: str(x) != '0')&data['Date'].map(lambda x: str(x) != '0')], index=keys, values='User_id',aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'User_id': prefixs + 'received_and_consume_differ_User_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    # 商家优惠券未被多少个不同的客户消费
    pivot = pd.pivot_table(data[data['Date_received'].map(lambda x: str(x) != '0')&data['Date'].map(lambda x: str(x) == '0')],index=keys, values='User_id',aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'User_id': prefixs + 'received_and_not_consume_differ_User_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)

    #商家提供的不同优惠券数
    pivot = pd.pivot_table(data, index=keys, values='Coupon_id',aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'Coupon_id': prefixs + 'differ_coupon_cnt'}).reset_index()
    m_feat = pd.merge(m_feat, pivot, on=keys, how='left')
    m_feat.fillna(0, downcast='infer', inplace=True)
    return m_feat

def get_coupon_feature(dataset):
    data = dataset.copy()
    # 将Coupon_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素为float
    # data['Coupon_id'] = data['Coupon_id'].map(int)  出现cannot convert float NaN to
    data['Coupon_id'] = data['Coupon_id'].map(lambda x: 0 if pd.isna(x) else int(x))
    # 同理
    data['Date_received'] = data['Date_received'].map(lambda x: 0 if pd.isna(x) else int(x))
    data['Date'] = data['Date'].map(lambda x: 0 if pd.isna(x) else int(x))
    # data['Distance'] = data['Distance'].map(int)
    # data['Date_received'] = data['Date_received'].map(int)
    # 方便特征提取
    data['cnt'] = 1
    keys = ['Coupon_id']
    prefixs = 'feature_' + '_'.join(keys) + '_'
    c_feat = data[keys].drop_duplicates(keep='first')
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
    # c_feat[prefixs + 'received_and_consume_rate']=list(map(lambda x,y:x/y if y!=0 else 0,c_feat[prefixs + 'received_and_consume_cnt'],c_feat[prefixs+'received_cnt']))

    #优惠券15天被核销的平均时间间隔
    tmp=data[data['label']==1]
    tmp['gap']=(tmp['date']-tmp['date_received']).map(lambda x : x.total_seconds()/(60*60*24))
    pivot =pd.pivot_table(tmp,index=keys,values='gap',aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={'gap':prefixs+'consume_mean_time_gap_15'}).reset_index()
    c_feat = pd.merge(c_feat, pivot, on=keys, how='left')
    c_feat.fillna(0, downcast='infer', inplace=True)

    #满减优惠券最低消费的中位数
    pivot = pd.pivot_table(data[data['is_manjian']==1],index=keys, values='min_cost_of_manjian', aggfunc=np.median)
    pivot = pd.DataFrame(pivot).rename(columns={'min_cost_of_manjian': prefixs + 'median_of_min_cost_of_manjian'}).reset_index()
    c_feat = pd.merge(c_feat, pivot, on=keys, how='left')
    c_feat.fillna(0, downcast='infer', inplace=True)

    return c_feat


def get_processData(feature,dataset):
    # 关于以往历史的特征
    u_feat = get_user_feature(feature)
    print('user_feature finish')
    w_feat = get_merchant_feature(feature)
    print('merchant_feature finish')
    c_feat = get_coupon_feature(feature)
    print('coupon_feature finish')
    # 结合
    dataset = pd.merge(dataset, u_feat, on='User_id', how='left')
    dataset.fillna(0, downcast='infer', inplace=True)
    dataset = pd.merge(dataset, w_feat, on='Merchant_id', how='left')
    dataset.fillna(0, downcast='infer', inplace=True)
    dataset = pd.merge(dataset, c_feat, on='Coupon_id', how='left')
    dataset.fillna(0, downcast='infer', inplace=True)
    # 关于数据的特征
    dataset = get_simple_feature(dataset)
    return dataset
