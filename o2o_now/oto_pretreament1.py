import pandas as pd
import numpy as np

# pd.set_option('display.max_columns', None)
# #显示所有行
# pd.set_option('display.max_rows', None)

#读入数据集
offline = pd.read_csv('ccf_offline_stage1_train.csv')

#———————————————————————————————————————————————————————————————————————
#           ------------数据预处理阶段①:缺失值处理--------------
#查看缺失值
print(offline.isnull().any())
#查询缺失值比例
print(offline.isnull().sum()/len(offline))

#距离空值填充为：-1
offline['Distance'].fillna(-1,inplace=True)
#判断是否为空距离
offline['null_distance'] = offline['Distance'].map(lambda x : 1 if x == -1 else 0 )
#          --------------数据预处理阶段② ：异常数据监测与剔除----------

#          --------------数据预处理阶段③ ：非数值型数据样本的转化-------
#Discount_rate是否为满减
offline['is_manjian'] = offline['Discount_rate'].map(lambda x : 1 if ':' in str(x) else 0)
#满减全部转化为折扣率
offline['discount_rate'] = offline['Discount_rate'].map(lambda x : float(x) if ':' not in str(x) else
    (float(str(x).split(':')[0])-float(str(x).split(':')[1]))/float(str(x).split(':')[0]))
#满减最低消费
offline['min_cost_of_manjian'] = offline['Discount_rate'].map(lambda x : -1 if ':' not in str(x) else int(str(x).split(':')[0]))
#满减减多少
offline['reduce_manjian'] = offline['Discount_rate'].map(lambda x : -1 if ':' not in str(x) else int(str(x).split(':')[1]))

#时间转换
offline['date_received'] = pd.to_datetime(offline['Date_received'],format = '%Y%m%d')
offline['date'] = pd.to_datetime(offline['Date'],format = '%Y%m%d')

#          --------------数据预处理阶段④： 对某些属性进行预先处理以方便后面进行特征提取

#------------------------------------------------------------------------------------------------------------------
#          --------------数据划分-----------------------
#划分函数
def get_dataset(history_field,middle_field,label_field):
    #特征工程
    history_feat = hf.get_history_field_feature(label_field,history_field)#历史区间特征
    middle_feat = mf.get_middle_feature(label_field,middle_field) #中间权健特征
    label_feat = lf.get_label_field_feature(label_field) # 标签区间特征
    #构造数据集
    share_chracters = list(set(history_feat.columns.tolist())&set(middle_feat.columns.tolist())&
                           set(label_feat.columns.tolist())) #共有属性，包括id和一些基础特征，为每个特征块的交集
    dataset = pd.concat([history_feat,middle_feat.drop(share_chracters,axis=1)],axis = 1)#这里使用concat连接而不是merge
            #因为几个特征块的样本顺序一致，index一致，但需要注意在连接两个特征时要删去其中一个特征块的共有属性
    dataset = pd.concat([dataset , label_feat.drop(share_chracters,axis=1)],axis =1)
    #删除无用属性并将label置于最后一列
    if 'Date' in dataset.columns.tolist():  #表示训练集和验证集
        dataset.drop(['Merchant_id','Discount_rate','Date','date_received','date'],axis = 1 ,inplace = True)
        label =dataset['label'].tolist()
        dataset.drop(['label'],axis = 1, inplace =True)
        dataset['label']=label
    else: #表示测试集
        dataset.drop(['Merchant_id','Discount_rate','Date','date_received'],axis = 1 ,inplace = True)
    #修正数据类型
    dataset['User_id'] = dataset['User_id'].map(int)
    dataset['Coupon_id'] = dataset['Coupon_id'].map(int)
    dataset['Date_received'] = dataset['Date_received'].map(int)
    dataset['Distance'] = dataset['Distance'].map(int)
    if 'label' in dataset.columns.tolist():
        dataset['label'] = dataset['lable'].map(int)
    #返回
    return dataset
#划分区间
off_train = pd.DataFrame
off_test = pd.DataFrame
#训练集历史区间、中间区间、标签区间。
train_history_field = off_train[off_train['date_received'].isin(pd.date_range('2016/3/2',periods = 60))]  #[20160302,20160501]
train_middle_field = off_train[off_train['date'].isin(pd.date_range('2016/5/1',periods = 15))] #[20160501,20160516]
train_label_field = off_train[off_train['date_received'].isin(pd.date_range('2016/5/16',periods = 31))] #[20160516,20160616]
#验证集历史区间、中间区间、标签区间、
validate_history_field = off_train[off_train['date_received'].isin(pd.date_range('2016/1/16',periods = 60))]  #[20160116,20160316]
validate_middle_field = off_train[off_train['date'].isin(pd.date_range('2016/3/16',periods = 15))] #[20160316,20160331]
validate_label_field = off_train[off_train['date_received'].isin(pd.date_range('2016/3/31',periods = 31))] #[20160616,20160701]
#测试集历史区间、中间区间、标签区间、
test_history_field = off_train[off_train['date_received'].isin(pd.date_range('2016/4/17',periods = 60))]  #[20160417,20160616]
test_middle_field = off_train[off_train['date'].isin(pd.date_range('2016/6/16',periods = 15))] #[20160616,20160701]
test_label_field = off_test.copy() #[20160701,20160701]

#构造训练集、验证集、测试集
print('构造训练集')
train = get_dataset(train_history_field,train_middle_field,train_label_field)
print('构造验证集')
validate = get_dataset(validate_history_field,validate_middle_field,validate_label_field)
print('构造测试集')
test = get_dataset(test_history_field,test_middle_field,test_label_field)

#          --------------数据打标-----------------------
def get_label(dataset):
    '''领取优惠券后15天内使用的样本为1，否则为0
    Args :
        dataset:DateFrame类型的数据集off_train,包含属性'User_id','Merchant_id','Coupon_id','Discount_rate',
            'Distance','Data_received','Date'
    return:
        打标后的DataFrame类型的数据集
    '''
    #数据源
    data = dataset.copy()
    #打标： 领劵后15天内消费为1，否则为0
    data['label'] = list(map(lambda x , y : 1 if (x-y).total_seconds()/(60*60*24) <=15 else 0 , data['date'] , data['date_received']))
    return data


#———————————————————————————————————————————————————————————————————————
#                              特征工程（简单的处理）(也就是基础特征)
#获得打了标签的数据
data =get_label(offline).copy()
#将Coupon_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素为float
#data['Coupon_id'] = data['Coupon_id'].map(int)  出现cannot convert float NaN to
data['Coupon_id'] = data['Coupon_id'].map(lambda x : 0 if pd.isna(x) else int(x))
#同理
data['Date_received'] = data['Date_received'].map(lambda x : 0 if pd.isna(x) else int(x))
#data['Distance'] = data['Distance'].map(int)
#data['Date_received'] = data['Date_received'].map(int)
#方便特征提取
data['cnt'] = 1

feature = data.copy()

#用户领劵数
keys = ['User_id'] #主键
prefixs = 'simple_'+'_'.join(keys)+'_'  #特征名前缀，由'simple'和主键组成
pivot = pd.pivot(data,index = keys , values= 'cnt' , aggfunc =len) #透视图，以keys为键，’cnt‘为值，使用len统计出现的次数
pivot = pd.DataFrame(pivot).rename(columns = {'cnt':prefixs + 'receive_cnt'}).reset_index()
#pivot_table后keys会成为index，统计出的特征列会以values即’cnt‘命名，将其改名为特证名前缀+特征意义，并将index还原
feature = pd.merge(feature,pivot, on=keys ,how ='left') #将id列与特征列左连接

#用户领取特定优惠券数
keys = ['User_id','Coupon_id']
prefixs = 'simple_'+'_'.join(keys)+'_'
pivot = pd.pivot(data,index= keys,values='cnt',aggfunc =len)
pivot =pd.DataFrame(pivot).rename(columns = {'cnt':prefixs + 'receive_cnt'}).reset_index()
feature = pd.merge(feature,pivot, on=keys ,how ='left')

#用户当天领券数
keys = ['User_id','Date_received']
prefixs = 'simple_'+'_'.join(keys)+'_'
pivot = pd.pivot(data,index= keys,values='cnt',aggfunc =len)
pivot =pd.DataFrame(pivot).rename(columns = {'cnt':prefixs + 'receive_cnt'}).reset_index()
feature = pd.merge(feature,pivot, on=keys ,how ='left')

#用户当天领取特定优惠券数
keys = ['User_id','Coupon_id','Date_received']
prefixs = 'simple_'+'_'.join(keys)+'_'
pivot = pd.pivot(data,index= keys,values='cnt',aggfunc =len)
pivot =pd.DataFrame(pivot).rename(columns = {'cnt':prefixs + 'receive_cnt'}).reset_index()
feature = pd.merge(feature,pivot, on=keys ,how ='left')

#用户是否在同一天重复领取了特定优惠券
keys = ['User_id','Coupon_id','Date_received']
prefixs = 'simple_'+'_'.join(keys)+'_'
pivot = pd.pivot(data,index= keys,values='cnt',aggfunc = lambda x : 1 if len(x)>1 else 0)
pivot =pd.DataFrame(pivot).rename(columns = {'cnt':prefixs + 'receive_receive'}).reset_index()
feature = pd.merge(feature,pivot, on=keys ,how ='left')

#删除辅助体特征的cnt
feature.drop(['cnt'],axis=1,inplace = True)

#离散化星期
feature['week'] = feature['date_received'].map(lambda x : x.weekday()) #星期几
feature['is_weekend'] = feature['week'].map(lambda x : 1 if x==5 or x== 6 else 0)#判断领券日是否为休息日
feature = pd.concat([feature,pd.get_dummies(feature['week'],prefix='week')],axis = 1)
feature.index = range(len(feature)) #重置index


import xgboost as xgb
#xgb参数
params = {'booster':'gbtree',
          'objective':'binary:logistic',
          'eval_metric':'auc',
          'silent':'auc',
          'eta':0.01,
          'max_depth':5,
          'min_child_weight':1,
          'gamma':0,
          'lambda':1,
          'colsample_bylevel':0.7,
          'colsample_bytree':0.7,
          'subsample':0.9,
          'scale_pos_weight':1}
#数据集
train=pd.DataFrame
dtest=pd.DataFrame
dtrain = xgb.DMatrix(train.drop(['User_id','Coupon_id','Date_received','label'],axis=1),label= train['label'])#训练集
dtest = xgb.DMatrix(dtest.drop(['User_id','Coupon_id','Date_received','label'],axis=1),label= train['label'])#测试集
#训练
watchlist = [(dtrain,'train')]
model = xgb.train(params,dtrain,num_boost_round = 5167, evals = watchlist) #迭代次数5167次
#预测
predict = model.predict(dtest)