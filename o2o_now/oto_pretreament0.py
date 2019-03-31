from datetime import date
import pandas as pd
import numpy as np

#新建关于星期的特征
def getWeekday(row):
    if row == 'null':
        return row
    else:
        return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1


read = pd.read_csv('./off_oto_add_feature_and_labels.csv', keep_default_na=False)
read['weekday'] = read['Date_received'].astype(str).apply(getWeekday)
read['weekday_type'] = read['weekday'].apply(lambda x : 1 if x in [6,7] else 0 )
#星期几的特征
weekdaycols = ['weekday_' + str(i) for i in range(1,8)]
print(weekdaycols)
tmpdf = pd.get_dummies(read['weekday'].replace('null', np.nan))
tmpdf.columns = weekdaycols
read[weekdaycols] = tmpdf
read.to_csv('off_oto_add_feature_and_labels2.csv',index=False,header=True)




# dfoff['weekday'] = dfoff['Date_received'].astype(str).apply(getWeekday)
# dftest['weekday'] = dftest['Date_received'].astype(str).apply(getWeekday)
#
# # weekday_type :  周六和周日为1，其他为0
# dfoff['weekday_type'] = dfoff['weekday'].apply(lambda x : 1 if x in [6,7] else 0 )
# dftest['weekday_type'] = dftest['weekday'].apply(lambda x : 1 if x in [6,7] else 0 )

# # change weekday to one-hot encoding
# weekdaycols = ['weekday_' + str(i) for i in range(1,8)]
# print(weekdaycols)
#
# tmpdf = pd.get_dummies(dfoff['weekday'].replace('null', np.nan))
# tmpdf.columns = weekdaycols
# dfoff[weekdaycols] = tmpdf
#
# tmpdf = pd.get_dummies(dftest['weekday'].replace('null', np.nan))
# tmpdf.columns = weekdaycols
# dftest[weekdaycols] = tmpdf