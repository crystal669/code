import numpy as np
import pandas as pd
import os

read_path1='F:\\DeepLearning\\DataSet\\mydataset\\process\\kddcup\\user_info\\convert_user_info.csv'
read_path2='F:\\DeepLearning\\DataSet\\mydataset\\process\\kddcup\\user_info\\extract_enroll_userid.csv'

save_dir='F:\\DeepLearning\\DataSet\\mydataset\\process\\kddcup\\user_info'

data1=pd.read_csv(read_path1)
data2=pd.read_csv(read_path2)

data=pd.merge(data1,data2,on='user_id')
#(data.shape)
#keep=last:保留最后值
data.drop_duplicates(['user_id','enroll_id'],keep='last',inplace=True,ignore_index=True)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 保存文件
data.to_csv(os.path.join(save_dir, 'user_log.csv'), index=False)