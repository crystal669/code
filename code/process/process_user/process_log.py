import numpy as np
import pandas as pd

import os

read_train_path='F:\\DeepLearning\\DataSet\\mydataset\\raw\\kddcup15\\kddcup15\\prediction_log\\train_log.csv'
read_test_path='F:\\DeepLearning\\DataSet\\mydataset\\raw\\kddcup15\\kddcup15\\prediction_log\\test_log.csv'
save_path='F:\\DeepLearning\\DataSet\\mydataset\\process\\kddcup\\enroll_user_id.csv'

train_data=pd.read_csv(read_train_path,usecols=['enroll_id', 'username'])
test_data=pd.read_csv(read_test_path,usecols=['enroll_id', 'username'])

#若ignore_index=False,会出现一列索引列
data=pd.concat((train_data,test_data),axis=0,ignore_index=True)
#修改列名
data.rename(columns={'username':'user_id'})

if not os.path.exists(save_path):
    os.makedirs(save_path)

print(data.shape)
#保存文件
data.to_csv(save_path)