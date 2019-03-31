import numpy as np
import pandas as pd

# convert Discount_rate and Distance 折扣类型是”满减“类型还是”打折“类型
def getDiscountType(row):
    if row == 'null':
        return 'null'
    elif ':' in row:
        return 1
    else:
        return 0

# 转换折扣率
def convertRate(row):
    """Convert discount to rate"""
    if row == 'null':
        return 1.0
    elif ':' in row:
        rows = row.split(':')
        return 1.0 - float(rows[1]) / float(rows[0])
    else:
        return float(row)

# 获得折扣率满多少
def getDiscountMan(row):
    if ':' in row:
        rows = row.split(':')
        return int(rows[0])
    else:
        return 0

# 获得折扣率减多少
def getDiscountJian(row):
    if ':' in row:
        rows = row.split(':')
        return int(rows[1])
    else:
        return 0


#增加特征类型（提取特征）
def processData(df):
    # convert discunt_rate
    df['discount_rate'] = df['Discount_rate'].apply(convertRate)
    df['discount_man'] = df['Discount_rate'].apply(getDiscountMan)
    df['discount_jian'] = df['Discount_rate'].apply(getDiscountJian)
    df['discount_type'] = df['Discount_rate'].apply(getDiscountType)
    print (df['discount_rate'].unique())
    # convert distance
    df['distance'] = df['Distance'].replace('null', -1).astype(int)
    print (df['distance'].unique())
    return df

# 数据标注  至于传入的函数具体是对每一行还是每一列进行操作，取决于apply传入的axis参数，默认axis=0，表示对每一列进行操作，axis=1，表示对每一行进行操作。
def label(row):
    if row['Date_received'] == 'null':
        return -1
    if row['Date'] != 'null':
        td = pd.to_datetime(row['Date'], format='%Y%m%d') - pd.to_datetime(row['Date_received'], format='%Y%m%d')
        if td <= pd.Timedelta(15, 'D'):
            return 1
    return 0





# C参数keep_default_na=False 将NaN -> null
dfoff = pd.read_csv("C:\\Users\\Administrator\\Desktop\\tianchi\\new_o2o\\ccf_offline_stage1_train.csv",keep_default_na=False)
#dfoff = pd.read_csv("off_oto_add_feature_and_labels.csv", keep_default_na=False)
dftest = pd.read_csv("C:\\Users\\Administrator\\Desktop\\tianchi\\new_o2o\\ccf_offline_stage1_test_revised.csv",                     keep_default_na=False)
#dfon = pd.read_csv("C:\\Users\\Administrator\\Desktop\\tianchi\\new_o2o\\ccf_online_stage1_train.csv",keep_default_na=False)


dfoff = processData(dfoff)
dftest = processData(dftest)
print (dfoff.head(10))
dfoff['label'] = dfoff.apply(label, axis=1)
#至于传入的函数具体是对每一行还是每一列进行操作，取决于apply传入的axis参数，默认axis=0，表示对每一列进行操作，axis=1，表示对每一行进行操作。
print(dfoff['label'].value_counts())
dfoff.to_csv('off_oto_add_feature_and_labels.csv',index=False,header=True)




# print '有优惠券，购买商品条数', dfoff[(dfoff['Date_received'] != 'null') & (dfoff['Date'] != 'null')].shape[0]
# print '无优惠券，购买商品条数', dfoff[(dfoff['Date_received'] == 'null') & (dfoff['Date'] != 'null')].shape[0]
# print '有优惠券，不购买商品条数', dfoff[(dfoff['Date_received'] != 'null') & (dfoff['Date'] == 'null')].shape[0]
# print '无优惠券，不购买商品条数', dfoff[(dfoff['Date_received'] == 'null') & (dfoff['Date'] == 'null')].shape[0]
# # 在测试集中出现的用户但训练集没有出现
# print '1. User_id in training set but not in test set', set(dftest['User_id']) - set(dfoff['User_id'])
# # 在测试集中出现的商户但训练集没有出现
# print '2. Merchant_id in training set but not in test set', set(dftest['Merchant_id']) - set(dfoff['Merchant_id'])

# 值域
# print 'Discount_rate 值域:',dfoff['Discount_rate'].unique()
# print 'Distance 值域:', dfoff['Distance'].unique()
# #获取收到优惠券日期范围 和  消费日期的范围
# date_received = dfoff['Date_received'].unique()
# date_received = sorted(date_received[date_received != 'null'])
# date_buy = dfoff['Date'].unique()
# date_buy = sorted(date_buy[date_buy != 'null'])
# #date_buy = sorted(dfoff[dfoff['Date'] != 'null']['Date'])
# print '优惠券收到日期从', date_received[0], '到', date_received[-1]
# print '消费日期从', date_buy[0], '到', date_buy[-1]







