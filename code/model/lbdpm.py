#!/usr/bin/env python
# coding: utf-8

# In[1]:


#use allinfo---user+course+enroll
#attention after lstm
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer,RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import accuracy_score

import keras
from sklearn.utils import class_weight
import tensorflow.compat.v1 as tf
from tensorflow.keras.utils import to_categorical
from keras.models import Model
from keras.models import Input
from keras.layers import Dense,LSTM,Dropout,Flatten,Conv1D,MaxPool1D,AlphaDropout
from keras.layers import Permute,Reshape,RepeatVector,Multiply,Lambda,merge,BatchNormalization
from tensorflow.keras.layers import Attention
from keras import backend as K
from keras import regularizers
from keras.initializers import he_normal,lecun_normal,RandomNormal,glorot_normal,glorot_uniform

from keras.layers import Embedding
from keras.layers import concatenate
#from sklearn.utils import shuffle
import time
import os
import matplotlib.pyplot as plt
import re

enroll_num = 900#选课记录数50
#####参数
global time_step,n_pred
time_step = 80# 用100步预测未来-----time_steps
n_pred = 1  # 预测未来1步

#n_features = 9# 特征数
#对部分特征进行one_hot编码
#特征数从官方信息以及预处理文件中得到

#n_stu_features=1*3+1*8+1
global n_onehot_gender,n_onehot_edu,n_age,n_onehot_type,n_onehot_category,n_onehot_action,n_time,n_time_span
n_onehot_gender=3
n_onehot_edu=1
n_age=1
#n_course_features=1*3+1*19
n_onehot_type=3
n_onehot_category=19
#n_enrol_features=1*7+2
n_onehot_action=5
#time、timespan向量化
n_time=5
n_timespan=1
#action_feature_agument
#action1_sum,action1_prob,action1_timespan_sum,action1_timespan_avg
# 特征数(on-hot)
n_enroll_features = n_onehot_action+n_time+n_timespan
n_static_features=n_onehot_gender+n_onehot_edu+n_age+n_onehot_type+n_onehot_category
n_action_statistic_features=75
###网络参数
#embedding
static_emb_size=100

#lstm
lstm1_cell =128
lstm2_cell =128
lstm2_cell = 64

#action_statistic--hidden
#cnn
cnn_filter=128
kernel_sizes=1
#hidden
static_hidden1_cell=128
static_hidden2_cell=64
#
#特征融合--dense
hidden1_cell=128#和lstm_cell一样，要拼接
hidden2_cell=128

#attn
SINGLE_ATTENTION_VECTOR=False
#out
n_output_reg_cell = n_timespan  # 回归
n_output_clas_cell = n_onehot_action  # 分类(onehot),使用onehot编码分类树为编码长度

#training
epochs = 25#30
batch_size =32

#lr=0.001


# In[ ]:





# In[2]:


#提前终止
from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='val_out_clas_accuracy',patience=4,mode='max')

#loss出现nan停止训练
from keras.callbacks import TerminateOnNaN
nan_stoping=TerminateOnNaN()


# In[3]:


# Define our custom loss function
#regression
def huber(y_true, y_pred, delta=1):
    #return tf.reduce_mean(tf.square(true-pred)) with tf.Session() as sess:
    loss = tf.where(tf.abs(y_true-y_pred) < delta , 0.5*((y_true-y_pred)**2), delta*tf.abs(y_true - y_pred) - 0.5*(delta**2))
    #reduce_sum()中就是按照求和的方式对矩阵降维
    #reduce_mean()就是按照某个维度求平均值
    # K.sum(loss)
    return tf.reduce_sum(loss)
#classifoication
def focal_loss_softmax(y_true, y_pred,gamma=2.0,epsilon=1e-6):
    #多分类中alpha没用
    #alpha =0.25
    y_true=tf.convert_to_tensor(y_true,tf.float32)
    y_pred=tf.convert_to_tensor(y_pred,tf.float32)
    
    model_out=tf.add(y_pred,epsilon)
    ce=tf.multiply(y_true,-tf.log(model_out))
    weight=tf.multiply(y_true,tf.pow(tf.subtract(1.,model_out),gamma))
    fl=tf.multiply(1.,tf.multiply(weight,ce))
    reduce_fl=tf.reduce_max(fl,axis=1)
    return tf.reduce_mean(reduce_fl)
    
def focal_loss(y_true, y_pred,gamma=2.0,epsilon=1e-8):
   
    alpha =tf.constant([[0.15],[0.35],[0.45],[0.05],[0.2]],dtype=tf.float32)
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+epsilon))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0+epsilon))


# In[4]:


get_ipython().system('')


# In[5]:


a =tf.Variable([0.15,0.35,0.45,0.05,0.2])

print(a)


# In[6]:


#define optimizer
#前5个epoch学习率保持不变，5个之后学习率按照指数进行衰减
def lr_decay(epoch):
    lr=0.001
    if epoch < 4:
        return lr
    elif epoch<9: 
        return lr/10
    elif epoch<15:
        return lr/100
    else:
        return lr/1000
        #return lr*tf.math.exp(0.1*(epochs-epoch))

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_decay)


# In[7]:


#存储最佳权重
from keras.callbacks import ModelCheckpoint
checkpoint_filepath='best_weight_model_test_600_batch32_ts80-900.hdf5'
print(checkpoint_filepath)
weight_callback = ModelCheckpoint( #创建回调
    filepath=checkpoint_filepath, #告诉回调要存储权重的filepath在哪
    save_weights_only=True, #只保留权重(更有效)，而不是整个模型
    monitor='val_out_reg_loss', #度量
    mode='min', #找出使度量最大化的模型权重
    save_best_only=True #只保留最佳模型的权重(更有效)，而不是所有的权重
)


# In[ ]:





# In[8]:


#对时间步的attention
def attention_time_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)#行列互换，求矩阵inputs的转置
    a = Reshape((input_dim, time_step))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(time_step, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_time_vec')(a)
    time_attention_out= Multiply()([inputs, a_probs])                                        
    return time_attention_out


# In[9]:


#对特征（某维）的attention
def attention_static_block(inputs):
    input_dim = int(inputs.shape[2])
    features=int(inputs.shape[1])
    
    a=Flatten()(inputs)
    
    a=Dense(features*input_dim,activation='softmax')(a)
    
    a=Reshape((features,input_dim,))(a)

    a=Lambda(lambda x:K.sum(x,axis=2),name='attention')(a)
    a=RepeatVector(input_dim)(a)
    
    a_probs=Permute((2,1),name="attention_vec_dim")(a)
    static_attention_out=Multiply()([inputs,a_probs])
    return static_attention_out


# In[10]:


def get_timespan_statistics(data,time_step):
    cols=['action_navigate','navigate_sum','navigate_timespan_avg',
     'action_video','video_sum','video_timespan_avg',
     'action_courseware','courseware_sum','courseware_timespan_avg',
     'action_forum','forum_sum','forum_timespan_avg',
     'action_problem','problem_sum','problem_timespan_avg',]
    action_stainfo=pd.DataFrame(columns=cols)
    action_list=list(0.0 for i in range(15))
    action_name=['navigate','video','courseware','forum','problem']
    
    for i,row in data.iterrows():
        if i<time_step:
            continue
        action_list=list(0.0 for i in range(15))
        act_num=0
        for j in range(len(action_list)):
            if j%3==0:
                action_list[j]=act_num
                act_num+=1
        
        for k in range(i-time_step,i):
            index=action_name.index(data.loc[k,'action'])
            action_list[index*3+1]+=1
            action_list[index*3+2]+=data.loc[k,'time_span']
        
        #计算均值
        for m in range(len(action_list)):
            if(m%3==2 and action_list[m-1]!=0):
                action_list[m]=action_list[m]/action_list[m-1]
                action_list[m]=float(int(action_list[m]))
                
        action_stainfo.loc[len(action_stainfo)]=action_list
    action_sta_value=action_stainfo.values

    for i in range(len(action_list)):
        if(i%3==0):
            onehot_action_encoder= to_categorical(action_sta_value[:,i], n_onehot_action)
            sum_vector=np.array(onehot_action_encoder)
            avg_timespan_vector=np.array(onehot_action_encoder)
            for j in range(0,len(action_sta_value)):
                one_index=list(onehot_action_encoder[j]).index(1)
                sum_vector[j,one_index]=action_sta_value[j,1]
                avg_timespan_vector[j,one_index]=action_sta_value[j,2]
        elif i%3==1:
            if i==1:
                sta_values= np.concatenate((onehot_action_encoder, sum_vector), axis=1)
                sta_values= np.concatenate((sta_values,avg_timespan_vector), axis=1)
            else:
                sta_values=np.concatenate((sta_values, onehot_action_encoder), axis=1)
                sta_values= np.concatenate((sta_values, sum_vector), axis=1)
                sta_values= np.concatenate((sta_values,avg_timespan_vector), axis=1)
    print('sta_shape',sta_values.shape)            
    return sta_values


# In[11]:


a=pd.Series([1,2,3,0,4])
b=a.mask(a>3)
b=b.mask(b==0)
print(b)


# In[12]:


def get_mu_3sigma_outliers(s_List,cof=3):
    mu=np.std(s_List)
    var=np.var(s_List)
    out=mu+cof*var
    if out<100:
        outliers=s_List.mask(s_List<out)
    else:
        outliers=s_List.mask(s_List<100)
    outliers.dropna(axis=0,how='any',inplace = True)
    outliers = outliers.reset_index(drop=True)
    
    return outliers,out


# In[13]:


def get_filling(s_List,out):
    #可能包含异常值
    #series 类型
    if out<100:
        filling_nums=s_List.mask(s_List>=out)
    else:
        filling_nums=s_List.mask(s_List>=100)
    filling_nums=filling_nums.mask(filling_nums==0)
    filling_nums.dropna(axis=0,how='any',inplace = True)
    filling_nums = filling_nums.reset_index(drop=True)
    
    return filling_nums


# In[14]:


def get_act_outlier_filling(data):
    #异常值处理：
    #异常值可能大于100
    #大于100的必定为异常值
    #0也是异常值
    action_dict={'navigate':[],'video':[],'courseware':[],'forum':[],'problem':[]}

    for i, row in data.iterrows():
        action=data.loc[i,'action']
        time_timespan=data.loc[i,'time_span']
        action_dict[action].append(time_timespan)
    #action_data=pd.DataFrame.from_dict(action_dict,orient='index')
    #action_data=action_data.fillna(1.0)
    for key,value in  action_dict.items():
        if(len(value)>0):
            #注意是pandas
            value_sorted = sorted(value)
            series_value_sorted=pd.Series(value_sorted)
            series_value_sorted.fillna(0.0)
            outliers,out=get_mu_3sigma_outliers(series_value_sorted)
            outliers_list=outliers.tolist()
            outliers_list.append(0.0)
            
            #填充：
            series_value=pd.Series(value)
            series_value_sorted.fillna(0.0)
            filling_value=get_filling(series_value,out)
            filling_list=filling_value.tolist()
            
            #[[异常值列表],[填充值列表（<100）]
            action_dict[key]=[]
            action_dict[key].append(value)
            action_dict[key].append(outliers_list)
            action_dict[key].append(filling_list)
            
            #action_dict[key].append(filling_list_mu_3sigma)
    return action_dict


# In[ ]:





# In[15]:


# In[2]:
def judge_user(en_id,stulog_data):
    flag=0
    course_id=""
    stulog_data['enroll_id'] = stulog_data['enroll_id'].astype('str')
    if en_id in stulog_data['enroll_id'].values:
        #tolist返回的使列表
        index=stulog_data[stulog_data.enroll_id==en_id].index.tolist()[0]
        course_id=stulog_data.loc[index].values[5]
        flag=1
    return flag,course_id

def judge_course(cour_id,course_data):
    flag=0
    if cour_id in course_data['course_id'].values:
        flag=1
    return flag

#判断是否存在学生信息
def data_preprocess(data):
    #inplace = True 才会保存修改后的
    #data.dropna(axis=0,how='any',inplace = True)
    # 处理时间
    for i, row in data.iterrows():
        hour = int(row.start_time[11:13])
        data.loc[i,'start_time'] = hour
    #对时间进行log处理
    #print(data['time_span'])
    #data['time_span']=np.log2(data['time_span']) 
    #print(data['time_span'][:len(data)])
    
    act_out_fill_dict=get_act_outlier_filling(data)   

    for i, row in data.iterrows():
        act_list=act_out_fill_dict[row['action']][0]
        act_out_list=act_out_fill_dict[row['action']][1]
        act_fill_list=act_out_fill_dict[row['action']][2]
        #stu_fill_mu_3sigma_list=stu_out_fill_dict[row['action']][2]
        if data.loc[i,'time_span'] in act_out_list:
            ind=act_list.index(data.loc[i,'time_span'])
            flag=0
            #从中间向左边找
            for j in range(ind,-1,-1):
                if act_list[j] in act_fill_list:
                    data.loc[i,'time_span']=act_list[j]
                    flag=1
                    break
            if flag==0:
                for j in range(ind,len(act_list),1):
                    if act_list[j] in act_fill_list:
                        data.loc[i,'time_span']=act_list[j]
                        flag=1
                        break
            if flag==0:
                data.loc[i,'time_span']=1.0
    data = data.reset_index(drop=True)
    #for i, row in data.iterrows():
        #if data.loc[i,'time_span']<1.0 and data.loc[i,'time_span']>0.0 :
            #data.loc[i,'time_span']=1.0
        #data.loc[i,'time_span']=float(int(data.loc[i,'time_span']))
        
    data_value = data.values
    # 对action进行整数编码
    action_name=['navigate','video','courseware','forum','problem']
    for i in range(len(data_value)):
        data_value[i,0]=int(action_name.index(data_value[i,0]))
    # 对 action进行one-hot编码
    onehot_action_encoder = to_categorical(data_value[:,0], n_onehot_action)
    #action：[0,1,0,0,0],time_span:[0,20,0,0,0]
    #data_values = np.concatenate((onehot_encoder, data_value[:, 1:]), axis=1)
    
    #不能直接写=号
    starttime_vector=np.array(onehot_action_encoder)
    for i in range(0,len(data_value)):
        one_index=list(onehot_action_encoder[i]).index(1)
        starttime_vector[i,one_index]=data_value[i,1]
    data_values=np.concatenate((onehot_action_encoder,starttime_vector), axis=1)
    timespan_value=data_value[:,-1]
    timespan_value=timespan_value.reshape(len(timespan_value),1)
    data_values=np.concatenate((data_values,timespan_value), axis=1)

    return data_values,data # 返回张量

#add gender,education,age
def add_user_info(en_id,stulog_data,enroll_len):
    user=np.zeros(3)
    #inplace = True 才会保存修改后的
    stulog_data.fillna(0,inplace = True)
    course_id=""
    stulog_data['enroll_id'] = stulog_data['enroll_id'].astype('str')
    if en_id in stulog_data['enroll_id'].values:
        #tolist返回的使列表
        index=stulog_data[stulog_data.enroll_id==en_id].index.tolist()[0]
        user=stulog_data.loc[index].values[1:4]
        course_id=stulog_data.loc[index].values[5]
        #print('user:',user)

    # 对 gender进行one-hot编码
    onehot_gender = to_categorical(user[0], n_onehot_gender)
    #对 education进行one-hot编码
    #onehot_edu = to_categorical(user[1], n_onehot_edu)
    
    edus = [0,"Bachelor's", "High", "Master's", "Primary", "Middle", "Associate", "Doctorate"]
    if(user[1]!=0):
        user[1]=edus[user[1]]
    if(user[1]=="Associate"):
        user[1]="Bachelor's" 
    new_edus=["Primary","Middle", "High","Bachelor's", "Master's", "Doctorate"]
    if(user[1]!=0):
        user[1]=int(new_edus.index(user[1])+1)
    
    #拼接
    onehot_user=np.append(onehot_gender,user[1])
    onehot_user=np.append(onehot_user,user[2])
    print(onehot_user)
    #onehot_user=onehot_user.reshape(1,len(onehot_user))
    
    #重复复制n行
    user_info=np.tile(onehot_user,enroll_len)
    user_info=user_info.reshape(enroll_len,len(onehot_user))
    #不写赋值，直接user_info.reshape(enroll_len,3)会报错
    
    #print(len(onehot_user))
    #返回张量
    return  user_info,course_id

#加入课程信息
#可以加入课程大类
def add_course_info(cour_id,course_data,enroll_len):
    course=np.zeros(2)
    course_data.fillna(0,inplace = True)
    if cour_id in course_data['course_id'].values:
        #tolist返回的使列表
        index=course_data[course_data.course_id==cour_id].index.tolist()[0]
        course=course_data.loc[index].values[1:3]
    # 对 type进行one-hot编码
    onehot_type = to_categorical(course[0], n_onehot_type)
    #对 category进行one-hot编码
    onehot_cat = to_categorical(course[1], n_onehot_category)
    
    #拼接
    onehot_course=np.append(onehot_type,onehot_cat)
    #onehot_course=onehot_course.reshape(1,len(onehot_course))
    
    #重复复制n行
    course_info=np.tile(onehot_course,enroll_len)
    #不写赋值，直接user_info.reshape(enroll_len,3)会报错
    course_info=course_info.reshape(enroll_len,len(onehot_course))
    
    #返回张量
    return course_info

# 构造nstep->n_pred的监督型数据,n_steo组数据预测n_pred组
# 隐式添加时序信息
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_dcol = 1 if type(data) is list else data.shape[1]  # data列数
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # 输入序列（t-n,...,t-1）
    # 将三组输入数据依次向下移动n,n-1,..,1行，将数据加入cols列表
    # var1----var7表示action(one-hot)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_dcol)]
    # 预测序列（t，t+1,t+n）
    # 将一组数据加入cols列表
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_dcol)]
        else:
            names += [('var%d(t+i)' % (j + 1)) for j in range(n_dcol)]
    # 将数据列表（cols）中现有的time_step+n_pred块（df-n_in,df-n_in+1,...df-1,df,..,df+n_pred+1）按列合并
    agg = pd.concat(cols, axis=1)
    # 为合并后的数据添加列名
    agg.columns = names
    
    #移位后产生的NaN值补0
    #agg.fillna(0,inplace=True)
    print('agg:',agg.shape)
    # 删除NaN值
    if dropnan:
        agg.dropna(inplace=True)  # dropna()
    agg = agg.reset_index(drop=True)  # 删除原始索引，否则会生成一列index
    
    return agg 


# In[16]:


read_dir = 'F:\\DeepLearning\\DataSet\\mydataset\\process\\kddcup\\log_action_time'
enroll_list = os.listdir(read_dir)
all_enroll_data=pd.DataFrame()
enroll_i = 1
for en in enroll_list:
    if enroll_i > enroll_num:
        break
    # 加载数据
    dataset = pd.read_csv(os.path.join(read_dir, en),usecols=['action','time_span'])
    dataset.dropna(axis=0,how='any',inplace = True)

    #sub(模式串，（去替换模式串的字符串），主串)
    en_id=re.sub('.csv','',en) 

    if enroll_i == 1:
        all_enroll_data= dataset
    else:
        all_enroll_data = pd.concat((all_enroll_data, dataset), axis=0)
    if len(dataset) >= time_step:
        enroll_i += 1


# In[17]:


all_enroll_data=all_enroll_data.reset_index(drop=True)
#print(all_enroll_data)
#获得全局不同action的outlier
#Outliers_dict=get_mu_sigma_alldata(all_enroll_data)


# In[ ]:





# In[18]:


plt.plot(all_enroll_data.values[1000:2000,1],label='true_value')
#plt.plot(inv_yhat[:,enroll_features-1],label='predict_value')

plt.legend()
plt.title('action_time_span')
plt.show()


# In[ ]:





# In[19]:


#read_dir = 'F:\\DeepLearning\\DataSet\\mydataset\\process\\kddcup\\log_action_time'
read_user_path='F:\\DeepLearning\\DataSet\\mydataset\\process\\kddcup\\user_info\\user_log_info.csv'
read_course_path='F:\\DeepLearning\\DataSet\\mydataset\\process\\kddcup\\user_info\\process_course_info.csv'

stu_data=pd.read_csv(read_user_path)
course_data=pd.read_csv(read_course_path)

#enroll_list = os.listdir(read_dir)
enroll_i = 1
data_value = np.empty((1, 9), dtype='float32')
all_enroll_reframed=pd.DataFrame()
all_stu_value=np.zeros(n_onehot_gender+n_onehot_edu+n_age)
all_course_value=np.zeros(n_onehot_type+n_onehot_category)
all_action_sta_value=np.zeros(n_action_statistic_features)


# print(all_data_reframed)
for en in enroll_list:
    if enroll_i >enroll_num:
        break
    # 加载数据
    dataset = pd.read_csv(os.path.join(read_dir, en),usecols=['action', 'start_time', 'time_span'])
    dataset.dropna(axis=0,how='any',inplace = True)
    #处理'time_span'数据
    #dataset=dataset.drop(dataset[dataset['time_span']==0].index)
    #dataset.reset_index(drop=True,inplace=True)
    if len(dataset) < time_step:
        continue
    else:
        #sub(模式串，（去替换模式串的字符串），主串)
        en_id=re.sub('.csv','',en) 
        stu_flag,course_id=judge_user(en_id,stu_data)
        cour_flag=judge_course(course_id,course_data)
        #print(stu_flag,cour_flag)
        #if(stu_flag==0) and (cour_flag==0):
            #continue
        # 数据预处理
        enroll_info,dataset= data_preprocess(dataset)
        enroll_info  = enroll_info.astype('float32')
        #添加用户信息
        #log_value.shape[0]-time_step
        stu_info,course_id=add_user_info(en_id,stu_data,enroll_info.shape[0]-time_step)
        stu_info=stu_info.astype('float32')
        #添加课程信息
        course_info=add_course_info(course_id,course_data,enroll_info.shape[0]-time_step)
        course_info=course_info.astype('float32')
        #统计前timestep特征增强信息
        action_stainfo=get_timespan_statistics(dataset,time_step)
        
        #print(stu_scaled,course_scaled)
        # 构造监督型时序数据time_step---->n_pred
        enroll_reframed = series_to_supervised(enroll_info, time_step, n_pred)
        #print(log_reframed.shape[0],'hhhhhh',log_info.shape[0]-time_step)
        if enroll_i == 1:
            all_enroll_reframed = enroll_reframed
            all_stu_value=stu_info
            all_course_value=course_info 
            all_action_sta_value=action_stainfo
        else:
            all_enroll_reframed = pd.concat((all_enroll_reframed, enroll_reframed), axis=0)
            all_stu_value=np.concatenate((all_stu_value,stu_info),axis=0)
            all_course_value=np.concatenate((all_course_value,course_info),axis=0)
            all_action_sta_value=np.concatenate((all_action_sta_value,action_stainfo),axis=0)
        enroll_i += 1
        print(en_id,"ddddddddddd", enroll_i)


# In[20]:


print(enroll_list.index(en))


# In[21]:


print(enroll_info[4,:])


# In[22]:


# 取张量
all_enroll_value = all_enroll_reframed.values
#拼接统计信息
statistic_value=all_action_sta_value

# 判断是否有nan值
if (np.isnan(enroll_reframed.values).any()):
    print("enroll:nan")
if (np.isnan(all_stu_value).any()):
    print("enroll:nan")
if (np.isnan(all_course_value).any()):
    print("enroll:nan")
if(np.isnan(all_action_sta_value).any()):
    print("##enroll:nan")
    
#拼接动态信息

#拼接静态信息
static_value=np.concatenate((all_stu_value,all_course_value),axis=1)


# In[23]:


print(static_value.shape)
print(statistic_value.shape)


# In[24]:


print(all_action_sta_value[:,:].max())


# In[25]:


#all_enroll_value,static_value=shuffle(all_enroll_value,static_value)


# In[26]:


enroll_features=all_enroll_value.shape[1]
static_features=static_value.shape[1]
all_value=np.concatenate((all_enroll_value,static_value),axis=1)
all_value=np.concatenate((all_value,statistic_value),axis=1)

all_scale=all_enroll_value.shape[0]
# 分割训练集、验证集和测试集6:1:3
train_scale = int(all_scale * 0.6)
val_scale = int(all_scale * 0.1)
test_scale = all_scale - train_scale - val_scale

value_train=all_value[:train_scale,:]
value_val=all_value[train_scale:train_scale + val_scale,:]
value_test=all_value[train_scale + val_scale:,:]

#scaler_test=StandardScaler().fit(value_train)
scaler_test=MinMaxScaler(feature_range=(0, 1)).fit(value_train)
scaled_value_train=scaler_test.transform(value_train)
scaled_value_val=scaler_test.transform(value_val)
scaled_value_test=scaler_test.transform(value_test)



#enroll
enroll_train,enroll_val,enroll_test = scaled_value_train[:,:enroll_features], scaled_value_val[:,:enroll_features],scaled_value_test[:,:enroll_features]
#static:
static_scaled_train,static_scaled_val,static_scaled_test=scaled_value_train[:,enroll_features:(enroll_features+static_features)], scaled_value_val[:,enroll_features:(enroll_features+static_features)],scaled_value_test[:,enroll_features:(enroll_features+static_features)]
#statistic:
statistic_scaled_train,statistic_scaled_val,statistic_scaled_test=scaled_value_train[:,(enroll_features+static_features):], scaled_value_val[:,(enroll_features+static_features):],scaled_value_test[:,(enroll_features+static_features):]


# In[27]:


print(static_scaled_train.shape[1])


# In[28]:


# 有（time_step+n_pred）*features--
#enroll动态信息里面加入了时序信息，扩充了维度
n_enroll = time_step * n_enroll_features
print(n_enroll)
#动态信息未扩充维度--n_static_features
enroll_scaled_trainX, enroll_train_Y, enroll_train_Yclass = enroll_train[:, :n_enroll], enroll_train[:, -n_timespan], enroll_train[:, -n_enroll_features:-(n_time+n_timespan)]
enroll_scaled_valX, enroll_val_Y, enroll_val_Yclass = enroll_val[:, :n_enroll], enroll_val[:, -n_timespan], enroll_val[:, -n_enroll_features:-(n_time+n_timespan)]
enroll_scaled_testX, enroll_test_Y, enroll_test_Yclass = enroll_test[:, :n_enroll], enroll_test[:, -n_timespan], enroll_test[:, -n_enroll_features:-(n_time+n_timespan)]

#拼接enroll_statistic
enroll_scaled_trainx=enroll_scaled_trainX
enroll_scaled_valx=enroll_scaled_valX
enroll_scaled_testx=enroll_scaled_testX
enroll_statistic_train=np.concatenate((enroll_scaled_trainx,statistic_scaled_train),axis=1)
enroll_statistic_val=np.concatenate((enroll_scaled_valx,statistic_scaled_val),axis=1)
enroll_statistic_test=np.concatenate((enroll_scaled_testx,statistic_scaled_test),axis=1)
enroll_statistic_train=enroll_statistic_train.reshape(enroll_statistic_train.shape[0],1,enroll_statistic_train.shape[1])
enroll_statistic_val=enroll_statistic_val.reshape(enroll_statistic_val.shape[0],1,enroll_statistic_val.shape[1])
enroll_statistic_test=enroll_statistic_test.reshape(enroll_statistic_test.shape[0],1,enroll_statistic_test.shape[1])
#######
# 将enroll数据转换为3D输入，timesteps=time_step,time_step预测n_pred
# [samples,timesteps,features]
enroll_scaled_trainX= enroll_scaled_trainX.reshape((enroll_scaled_trainX.shape[0], time_step, n_enroll_features))
enroll_scaled_valX = enroll_scaled_valX.reshape((enroll_scaled_valX.shape[0], time_step, n_enroll_features))
enroll_scaled_testX= enroll_scaled_testX.reshape((enroll_scaled_testX.shape[0], time_step, n_enroll_features))
# print(train_X.shape,train_Y.shape)
#static.reshape
static_scaled_train=static_scaled_train.reshape(static_scaled_train.shape[0],1,static_scaled_train.shape[1])
static_scaled_val=static_scaled_val.reshape(static_scaled_val.shape[0],1,static_scaled_val.shape[1])
static_scaled_test=static_scaled_test.reshape(static_scaled_test.shape[0],1,static_scaled_test.shape[1])

print(static_scaled_train.shape)
#static_scaled_train= static_scaled_train.reshape((static_scaled_train.shape[0],static_scaled_train.shape[1]))
#static_scaled_val= static_scaled_val.reshape((static_scaled_val.shape[0],static_scaled_val.shape[1]))
#static_scaled_test= static_scaled_test.reshape((static_scaled_test.shape[0],static_scaled_test.shape[1]))
print(enroll_scaled_testX.shape,static_scaled_test.shape,static_scaled_train.shape)


# In[ ]:





# In[29]:


#LSTM
#LSTM层是循环层，需要3维输入(batch_size, timesteps, input_dim)，即(训练数据量，时间步长，特征量)。
#因此不能直接把 [数据量*特征量]的二维矩阵输入，要用reshape进行转换。比如[50000,3]转化成时间步长为1的输入，即变成[50000,1,3]


# In[ ]:





# In[30]:


#class_weight(类别不均衡)
y_integers = np.argmax(enroll_train_Yclass, axis=1)
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
d_class_weights = dict(enumerate(class_weights))


# In[31]:


#计算模型运行时间
import time
time_start = time.clock()  # 记录开始时间


# In[32]:


#设计网络
# Input
enroll_input = Input(shape=(enroll_scaled_trainX.shape[1], enroll_scaled_trainX.shape[2]),name='enroll_input')
static_input=Input(shape=(1,static_scaled_train.shape[2]),name='static_input')
enroll_statistic_input=Input(shape=(1,enroll_statistic_train.shape[2]),name='enroll_statistic_input')

#tf.nn.leaky_relu
#model1
#SELU 必须与 lecun_normal 初始化一起使用，且将 AlphaDropout 作为 dropout。
enroll_lstm1 = LSTM(128, activation=tf.nn.tanh,
                    #针对于Relu的初始化方法,he_normal。
                    kernel_initializer=glorot_normal(seed=1),
                    #bias_initializer=RandomNormal(mean=0.0,stddev=0.01,seed=1),
                    bias_initializer='random_uniform',
                    #kernel_regularizer=regularizers.l1_l2(l1=0.05,l2=0.1),
                    use_bias=True,
                    return_sequences=True)(enroll_input)
enroll_lstm1=BatchNormalization()(enroll_lstm1)
enroll_lstm1=Dropout(rate=0.3)(enroll_lstm1)

#时间维度的注意力机制
attention_time=attention_time_block(enroll_lstm1)
attention_time_flatten=Flatten()(attention_time)
enroll_dense1=Dense(128,activation=tf.nn.leaky_relu,
                            kernel_initializer=he_normal(seed=1),
                            bias_initializer='random_uniform',
                            #kernel_regularizer=regularizers.l1_l2(l1=0.01,l2=0.03),
                            use_bias=True)(attention_time_flatten)
enroll_dense1=Dropout(rate=0.3)(enroll_dense1)
lstm_action_model=Model(inputs=enroll_input,outputs=enroll_dense1)


#model2
#特征提取
emb_static=BatchNormalization()(static_input)
cnn_static1=Conv1D(filters=128,kernel_size=1,
                   activation=tf.nn.leaky_relu,
                   kernel_initializer=he_normal(seed=1),
                    bias_initializer='random_uniform',
                   use_bias=True)(emb_static)
cnn_static1=Dropout(rate=0.2)(cnn_static1)
#cnn_max_pool=MaxPool1D(pool_size=4,strides=2)(cnn_static1)
cnn_flatten=Flatten()(cnn_static1)
static_dense1=Dense(64,activation=tf.nn.leaky_relu,
                            kernel_initializer=he_normal(seed=1),
                            bias_initializer='random_uniform',
                            #kernel_regularizer=regularizers.l1_l2(l1=0.01,l2=0.03),
                            use_bias=True)(cnn_flatten)
static_dense1=Dropout(rate=0.2)(static_dense1)
static_info_model=Model(inputs=static_input,outputs=static_dense1)

#model3

enroll_statistic_lstm1 = LSTM(128, activation=tf.nn.tanh,
                              #针对于Relu的初始化方法,he_normal。
                              kernel_initializer=glorot_normal(seed=3),
                              #bias_initializer=RandomNormal(mean=0.0,stddev=0.01,seed=1),
                              bias_initializer='random_uniform',
                              #kernel_regularizer=regularizers.l1_l2(l1=0.3,l2=0.1),
                              use_bias=True,
                              return_sequences=True)(enroll_statistic_input)
enroll_statistic_lstm1=BatchNormalization()(enroll_statistic_lstm1)
enroll_statistic_lstm1=Dropout(rate=0.3)(enroll_statistic_lstm1)
cnn_enroll_statistic1=Conv1D(filters=64,
                             kernel_size=1,
                             activation=tf.nn.leaky_relu,
                             kernel_initializer=he_normal(seed=3),
                             use_bias=True)(enroll_statistic_lstm1)
cnn_enroll_statistic1=Dropout(rate=0.2)(cnn_enroll_statistic1)
enroll_statistic_flatten=Flatten()(cnn_enroll_statistic1)
enroll_statistic_dense1=Dense(64,activation=tf.nn.leaky_relu,
                            kernel_initializer=he_normal(seed=1),
                            bias_initializer='random_uniform',
                            #kernel_regularizer=regularizers.l1_l2(l1=0.01,l2=0.03),
                            use_bias=True)(enroll_statistic_flatten)
enroll_statistic_dense1=Dropout(rate=0.1)(enroll_statistic_dense1)


enroll_statistic_model=Model(inputs=enroll_statistic_input,outputs=enroll_statistic_dense1)


#concatenate融合所有特征，增加通道数
combined_classX=concatenate([lstm_action_model.output,enroll_statistic_model.output,static_info_model.output])
combined_regX=concatenate([lstm_action_model.output,enroll_statistic_model.output])
#combinedX=BatchNormalization()(combinedX)
#hidden-reg

reg_hidden1=Dense(128,activation=tf.nn.leaky_relu, 
                            kernel_initializer=he_normal(seed=3), 
                            #kernel_regularizer=regularizers.l1_l2(l1=0.01,l2=0.3),
                            use_bias=True)(combined_regX)
reg_hidden1=Dropout(rate=0.4)(reg_hidden1)
reg_hidden2=Dense(128,activation=tf.nn.leaky_relu, 
                            kernel_initializer=he_normal(seed=3), 
                            kernel_regularizer=regularizers.l1_l2(l1=0.05,l2=0.1),
                            use_bias=True)(reg_hidden1)
reg_hidden2=Dropout(rate=0.3)(reg_hidden2)
#预测正值，最后一层加relu
reg_hidden3=Dense(64,activation=tf.nn.leaky_relu, 
                            kernel_initializer=he_normal(seed=3), 
                            #kernel_regularizer=regularizers.l1_l2(l1=0.01,l2=0.3),
                            use_bias=True)(reg_hidden2)
reg_hidden3=Dropout(rate=0.2)(reg_hidden3)
reg_hidden4=Dense(32,activation=tf.nn.leaky_relu, 
                            kernel_initializer=he_normal(seed=3), 
                            #kernel_regularizer=regularizers.l1_l2(l1=0.01,l2=0.3),
                            use_bias=True)(reg_hidden3)
reg_hidden4=Dropout(rate=0.1)(reg_hidden4)

#dense-output-regression
out_reg = Dense(n_output_reg_cell,activation=tf.nn.sigmoid, name='out_reg')(reg_hidden4)

#hidden--class
class_hidden1=Dense(128,activation=tf.nn.leaky_relu,
                            kernel_initializer=glorot_normal(seed=3),
                            bias_initializer='random_uniform',
                            #kernel_regularizer=regularizers.l1_l2(l1=0.01,l2=0.03),
                            use_bias=True)(combined_classX)
class_hidden1=Dropout(rate=0.4)(class_hidden1)
class_hidden2=Dense(128,activation=tf.nn.leaky_relu, 
                            kernel_initializer=he_normal(seed=3),
                            #kernel_regularizer=regularizers.l1_l2(l1=0.05,l2=0.1),
                            use_bias=True)(class_hidden1)
class_hidden2=Dropout(rate=0.3)(class_hidden2)
class_hidden3=Dense(64,activation=tf.nn.relu, 
                            kernel_initializer=he_normal(seed=3),
                            #kernel_regularizer=regularizers.l1_l2(l1=0.01,l2=0.3),
                            use_bias=True)(class_hidden2)
class_hidden3=Dropout(rate=0.2)(class_hidden3)
class_hidden4=Dense(32,activation=tf.nn.relu, 
                            #kernel_initializer=he_normal(seed=3),
                            #kernel_regularizer=regularizers.l1_l2(l1=0.01,l2=0.3),
                            use_bias=True)(class_hidden3)
class_hidden4=Dropout(rate=0.1)(class_hidden4)
#dense-output_classification
#Sigmoid(在二进制情况下)或softmax(在多类情况下)
#自定义softmax:https://www.jianshu.com/p/09a83c90fa71
# 
out_clas = Dense(n_output_clas_cell,activation='softmax', name='out_clas')(class_hidden4)



# define model
model = Model(inputs=[lstm_action_model.input,enroll_statistic_model.input, static_info_model.input],
              outputs=[out_reg, out_clas])
# 编译器
# [tf.keras.metrics.AUC(name='auc')]
#categorical_crossentropy
#huber
model.compile(loss={'out_reg':huber, 'out_clas':focal_loss_softmax},  optimizer='adam',
              metrics={'out_reg':'mse', 'out_clas': 'accuracy'})

#

# 训练拟合网络
history = model.fit({'enroll_input':enroll_scaled_trainX,'enroll_statistic_input':enroll_statistic_train,'static_input':static_scaled_train},
                    {'out_reg':enroll_train_Y, 'out_clas':enroll_train_Yclass},
                    epochs=20, batch_size=batch_size, 
                    class_weight=d_class_weights,
                    validation_data=([enroll_scaled_valX,enroll_statistic_val,static_scaled_val], [enroll_val_Y, enroll_val_Yclass]),
                    verbose=1,callbacks=[early_stopping,lr_callback,weight_callback,nan_stoping])
# 打印网络结构
model.summary()
#callbacks=[early_stopping]


# In[33]:


# 绘图
# plot history
plt.plot(history.history['out_reg_loss'], label='train')
plt.plot(history.history['val_out_reg_loss'], label='validate')
plt.xlabel('epochs')
plt.ylabel('reg_loss')
plt.legend()
plt.title('model_reg_loss')
plt.show()

#
# plot history
plt.plot(history.history['out_clas_loss'], label='train')
plt.plot(history.history['val_out_clas_loss'], label='validate')
plt.xlabel('epochs')
plt.ylabel('class_loss')
plt.legend()
plt.title('model_class_loss')
plt.show()

# plot history
plt.plot(history.history['out_reg_mse'], label='train')
plt.plot(history.history['val_out_reg_mse'], label='validate')
plt.xlabel('epochs')
plt.ylabel('reg_mse')
plt.legend()
plt.title('timespan_mse')
plt.show()
#plt.close

# plt.plot(history.history['sparse_categorical_accuracy'],label='train')
# plt.plot(history.history['val_sparse_categorical_accuracy'],label='validate')
plt.plot(history.history['out_clas_accuracy'], label='train')
plt.plot(history.history['val_out_clas_accuracy'], label='validate')
plt.xlabel('epochs')
plt.ylabel('clas_acc')
plt.title('action_accuracy')
plt.legend()
plt.show()


# In[ ]:





# In[34]:


print(enroll_test_Y.max())


# In[35]:


#将最佳权重装入模型中
model.load_weights(checkpoint_filepath) 
reg_train_pred,clas_train_pred=model.predict({'enroll_input':enroll_scaled_trainX,'enroll_statistic_input':enroll_statistic_train,'static_input':static_scaled_train})
# 预测
reg_test_pred, clas_test_pred = model.predict({'enroll_input':enroll_scaled_testX,'enroll_statistic_input':enroll_statistic_test,'static_input':static_scaled_test})


# In[36]:


# function()   执行的程序
time_end = time.clock()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print('Running time-SCTPN:',time_sum)


# In[37]:


#plt.plot(enroll_test_Y,label='true_value')
plt.plot(reg_test_pred,label='predict_value')

plt.legend()
plt.title('action_time_span')
plt.show()


# In[ ]:





# In[ ]:





# In[38]:


# 评估模型,不输出预测结果
#metrics = model.evaluate({'enroll_input':enroll_scaled_testX,'static_input':static_scaled_test},{'out_reg':enroll_test_Y, 'out_clas':enroll_test_Yclass})
#print('\n test_reg_loss:',metrics[1])
#print('\n test_clas_accuracy:%.3f'%(metrics[4]*100),'%')


# In[39]:


#注意：修改了n_features,这里也要修改

# 逆缩放维度要求：n行*features列
#拼接之前需要知道开始怎么对数据进行拼接的
# 现进行数据拼接，再对预测数据进行逆缩放
# test_X=test_X.reshape(test_X.shape[0],n_step*n_features)

# 现进行数据拼接，再对真实数据进行逆缩放
inv_Ytrue_test = scaler_test.inverse_transform(scaled_value_test)
inv_Ytrue_train=scaler_test.inverse_transform(scaled_value_train)

#预测值缩放
reg_test_pred = reg_test_pred.reshape(len(reg_test_pred), 1)
test_pred=np.concatenate((scaled_value_test[:,:enroll_features-n_timespan],reg_test_pred),axis=1)
test_pred=np.concatenate((test_pred,scaled_value_test[:,enroll_features:]),axis=1)
inv_test_pred = scaler_test.inverse_transform(test_pred)

reg_train_pred = reg_train_pred.reshape(len(reg_train_pred), 1)
train_pred=np.concatenate((scaled_value_train[:,:enroll_features-n_timespan],reg_train_pred),axis=1)
train_pred=np.concatenate((train_pred,scaled_value_train[:,enroll_features:]),axis=1)
inv_train_pred = scaler_test.inverse_transform(train_pred)


# In[40]:


print(inv_test_pred[40:50,(enroll_features-n_timespan):enroll_features],'\n\n')
print(inv_Ytrue_test[40:50,(enroll_features-n_timespan):enroll_features])


# In[41]:


#取整
for i in range(0,len(inv_test_pred)):
    inv_test_pred[i,(enroll_features-n_timespan)]=int(inv_test_pred[i,(enroll_features-n_timespan)])


# In[42]:


# 计算回归模型RMSE误差值
#enroll_test_Y=np.power(2,enroll_test_Y)
#reg_yhat=np.power(2,reg_yhat)
print(inv_Ytrue_test[:,enroll_features-n_timespan].max())
print(inv_test_pred[:,enroll_features-n_timespan].max())

rmse = math.sqrt(mean_squared_error(abs(inv_Ytrue_test[:,(enroll_features-n_timespan)]),abs(inv_test_pred[:,(enroll_features-n_timespan)])))
print('Test RMSE:%.3f' %rmse)


# In[43]:


######全局
#全局准确率
class_pred = np.argmax(clas_test_pred, axis=-1).astype('int')  # 返回最大数的索引
yclas = np.argmax(enroll_test_Yclass, axis=-1).astype('int')
acc = accuracy_score(yclas,class_pred)
print("acurracy:%.3f" % (acc*100),'%')

#全局rmse
# 计算回归模型RMSE误差值
rmse = math.sqrt(mean_squared_error(inv_test_pred[:,enroll_features-n_timespan],inv_Ytrue_test[:,enroll_features-n_timespan]))
print('Test RMSE:%.3f' %rmse)


# In[44]:


#计算mae
mae = mean_absolute_error(abs(inv_Ytrue_test[:,(enroll_features-n_timespan)]),abs(inv_test_pred[:,(enroll_features-n_timespan)]))
print('Test mae:%.3f' %mae)


# In[45]:


#!pip install seaborn


# In[46]:


#计算混淆矩阵
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns
#统计每个类别的预测准确率、召回率、F1-score
print(classification_report(class_pred,yclas,digits=5))
#计算混淆矩阵
confusion_mat=confusion_matrix(class_pred,yclas)
sns.set()
figure,ax=plt.subplots()
sns.heatmap(confusion_mat,cmap='YlGnBu_r',annot=True,ax=ax)
#标题
ax.set_title('class_acc_confusion_matrix')
#x轴为预测类别
ax.set_xlabel('predict')
#y为实际类别
ax.set_ylabel('true')
#在 plt.show() 之前调用 plt.savefig()
plt.savefig('./mymodel/confusion_mat.png')
plt.show()


# In[48]:


#roc
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score,f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from itertools import cycle

class_test_one_hot = label_binarize(yclas, classes=np.arange(n_onehot_action))
class_test_one_hot_hat=label_binarize(class_pred, classes=np.arange(n_onehot_action))
# weighted：不均衡数量的类来说，计算二分类metrics的平均
precision = precision_score(class_test_one_hot , class_test_one_hot_hat, average='weighted')
recall = recall_score(class_test_one_hot , class_test_one_hot_hat, average='weighted')
f1_score = f1_score(class_test_one_hot , class_test_one_hot_hat, average='weighted')
accuracy_score = accuracy_score(class_test_one_hot , class_test_one_hot_hat)
print("Precision_score:",precision)


# roc_curve:真正率（True Positive Rate , TPR）或灵敏度（sensitivity）
# 横坐标：假正率（False Positive Rate , FPR）
 
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_onehot_action):
    fpr[i], tpr[i], _ = roc_curve(class_test_one_hot[:, i], class_test_one_hot_hat[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# Compute weight-average ROC curve and ROC area
fpr["weighted"], tpr["weighted"], _ = roc_curve(class_test_one_hot.ravel(), class_test_one_hot_hat.ravel())
roc_auc["weighted"] = auc(fpr["weighted"], tpr["weighted"])

# Plot all ROC curves
lw = 2
plt.figure()
plt.plot(fpr["weighted"], tpr["weighted"],
  label='weighted-average ROC curve (area = {0:0.5f})'
  ''.format(roc_auc["weighted"]),
  color='navy', linestyle=':', linewidth=3)
 
colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','purple'])
for i, color in zip(range(n_onehot_action), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.5f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
#plt.savefig("")
plt.show()


# In[49]:


#print


# In[ ]:




