#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8
# In[1]:
#!/usr/bin/env python
# coding: utf-8
# get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import time
import numpy as np
from datetime import datetime
from sklearn.externals import joblib 
import os
import glob
from konlpy.tag import Mecab
import lightgbm as lgb
print(lgb.__version__)
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib 
from sklearn.model_selection import StratifiedKFold, KFold
import gc
from tqdm import tqdm_notebook, tqdm
import json
from typing import NamedTuple
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings(action='ignore')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
print(torch.__version__)
# from tools import eval_summary, save_feature_importance, merge_preds
from tools import EarlyStopping

device = torch.device('cpu')
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# In[2]:


# In[2]:
torch.set_num_threads(8)
torch.get_num_threads()


# #### Load Data

# In[3]:


df_train = pd.read_csv('input/train.csv', dtype=np.float32)
df_test = pd.read_csv('input/test.csv', dtype=np.float32)
print(df_train.shape, df_test.shape)


# In[4]:


df_train


# In[5]:


layer_cols = [c for c in df_train.columns if 'layer_' in c]
fea_cols = [c for c in df_train.columns if c not in layer_cols]

len(fea_cols), len(layer_cols)


# In[6]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df_train[fea_cols].values)

scaler = joblib.load('scaler.bin')

df_train[fea_cols] = scaler.transform(df_train[fea_cols].values)
df_test[fea_cols] = scaler.transform(df_test[fea_cols].values)


# In[7]:


df_train


# In[8]:


df_model = df_train


# #### Model

# ##### DNN1Model

# In[9]:


# df1 = pd.read_csv('input/SiO2.txt', sep='\t')
# df2 = pd.read_csv('input/Si3N4.txt', sep='\t')

# df_nk = pd.merge(df1, df2, on='Wavelength(nm)')
# df_nk = df_nk[:226]

# df_nk.columns


# In[10]:


class DNN1Model(torch.nn.Module):
    def __init__(self, input_size, dropout_probability=0.3):
        super().__init__()
        relu = torch.nn.ReLU()
        dropout = torch.nn.Dropout(p=dropout_probability)

        self.layer_1 = torch.nn.Sequential(
            torch.nn.Linear(input_size, input_size), relu, torch.nn.BatchNorm1d(input_size), dropout, 
            torch.nn.Linear(input_size, input_size), relu, torch.nn.BatchNorm1d(input_size), dropout, 
            torch.nn.Linear(input_size, input_size), relu, torch.nn.BatchNorm1d(input_size), dropout, 
        )
        self.layer_2 = torch.nn.Sequential(
            torch.nn.Linear(input_size, input_size), relu, torch.nn.BatchNorm1d(input_size), dropout, 
            torch.nn.Linear(input_size, input_size), relu, torch.nn.BatchNorm1d(input_size), dropout, 
            torch.nn.Linear(input_size, input_size), relu, torch.nn.BatchNorm1d(input_size), dropout, 
        )
        self.layer_3 = torch.nn.Sequential(
            torch.nn.Linear(input_size, input_size), relu, torch.nn.BatchNorm1d(input_size), dropout, 
            torch.nn.Linear(input_size, input_size), relu, torch.nn.BatchNorm1d(input_size), dropout, 
            torch.nn.Linear(input_size, input_size), relu, torch.nn.BatchNorm1d(input_size), dropout, 
        )
        self.layer_4 = torch.nn.Sequential(
            torch.nn.Linear(input_size, input_size), relu, torch.nn.BatchNorm1d(input_size), dropout, 
            torch.nn.Linear(input_size, input_size), relu, torch.nn.BatchNorm1d(input_size), dropout, 
            torch.nn.Linear(input_size, input_size), relu, torch.nn.BatchNorm1d(input_size), dropout, 
        )
        
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_size, input_size), relu,
            torch.nn.Linear(input_size, input_size), relu,
            torch.nn.Linear(input_size, input_size), relu,
            torch.nn.Linear(input_size, 4),
        )
        
        self.layer13_n = torch.Tensor(df_nk[['n_x']].T.values).to(device)
        self.layer13_k = torch.Tensor(df_nk[['k_x']].T.values).to(device)
        self.layer24_n = torch.Tensor(df_nk[['n_y']].T.values).to(device)
        self.layer24_k = torch.Tensor(df_nk[['k_y']].T.values).to(device)
        
    
    def forward(self, x_fea):
        
        out_layer_1 = self.layer_1(torch.add(torch.mul(x_fea, self.layer13_n), self.layer13_k))
        out_layer_2 = self.layer_2(torch.add(torch.mul(out_layer_1, self.layer24_n), self.layer24_k))
        out_layer_3 = self.layer_3(torch.add(torch.mul(out_layer_2, self.layer13_n), self.layer13_k))
        out_layer_4 = self.layer_4(torch.add(torch.mul(out_layer_2, self.layer24_n), self.layer24_k))
        
        return self.fc(out_layer_4)
    


# ##### DCNNModel

# In[11]:


# # size = 48
# W = 15 # input_volume_size
# F = 6  # kernel_size
# S = 1   # strides
# P = 1
# # padding_size

# size = (W - F + 2*P) / S + 1
# size
# # ((size - 1) * S) - 2*P + F


# In[12]:


class DCNNModel(torch.nn.Module):
    def __init__(self, input_size, dropout_probability=0.3):
        super(DCNNModel,self).__init__()
#         relu = torch.nn.ReLU()
        act = torch.nn.ELU()
        dropout = torch.nn.Dropout(p=dropout_probability)

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 1024), torch.nn.BatchNorm1d(1024), act, dropout, 
            torch.nn.Linear(1024, 1024), torch.nn.BatchNorm1d(1024), act, dropout,            
            torch.nn.Linear(1024, 512), torch.nn.BatchNorm1d(512), act, dropout,
            torch.nn.Linear(512, 512), torch.nn.BatchNorm1d(512), act, dropout,
            torch.nn.Linear(512, 256), torch.nn.BatchNorm1d(256), act, dropout,            
            torch.nn.Linear(256, 128),
            
        )
        
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(1, 12, 3, stride=1, padding=1), torch.nn.BatchNorm1d(12), act,
            torch.nn.Conv1d(12, 12, 3, stride=1, padding=1), torch.nn.BatchNorm1d(12), act,
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(12, 24, 3, stride=1, padding=1), torch.nn.BatchNorm1d(24), act,
            torch.nn.Conv1d(24, 24, 3, stride=1, padding=1), torch.nn.BatchNorm1d(24), act,
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(24, 48, 3, stride=1, padding=1), torch.nn.BatchNorm1d(48), act,
            torch.nn.Conv1d(48, 48, 3, stride=1, padding=1), torch.nn.BatchNorm1d(48), act,
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(48, 96, 3, stride=1, padding=1), torch.nn.BatchNorm1d(96), act,
            torch.nn.Conv1d(96, 96, 3, stride=1, padding=1), torch.nn.BatchNorm1d(96), act,
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(96, 96, 3, stride=1, padding=1), torch.nn.BatchNorm1d(96), act,
            torch.nn.Conv1d(96, 96, 3, stride=1, padding=1), torch.nn.BatchNorm1d(96), act,
            torch.nn.MaxPool1d(2),
        )
        
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(128 + 672, 512), act, dropout,
            torch.nn.Linear(512, 512), act, dropout,
            torch.nn.Linear(512, 4),
        )
        
    def forward(self, x):
        out_cnn = self.cnn(x.unsqueeze(1))
        dim = 1
        for d in out_cnn.size()[1:]:
            dim = dim * d
        out_cnn = out_cnn.view(-1, dim)
        
        out = self.clf(torch.cat([self.model(x), out_cnn], axis=1))
        return out
        
 


# ##### DNNModel

# In[13]:


# class DNNModel(torch.nn.Module):
#     def __init__(self, input_size, dropout_probability=0.3):
#         super(DNNModel,self).__init__()
#         act = torch.nn.ELU()
#         dropout = torch.nn.Dropout(p=dropout_probability)

#         self.model = torch.nn.Sequential(
#             torch.nn.Linear(input_size, 2048), torch.nn.BatchNorm1d(2048), act, dropout, 
#             torch.nn.Linear(2048, 2048), torch.nn.BatchNorm1d(2048), act, dropout, 
#             torch.nn.Linear(2048, 1024), torch.nn.BatchNorm1d(1024), act, dropout,
#             torch.nn.Linear(1024, 1024), torch.nn.BatchNorm1d(1024), act, dropout,            
#             torch.nn.Linear(1024, 512), torch.nn.BatchNorm1d(512), act, dropout,
#             torch.nn.Linear(512, 512), torch.nn.BatchNorm1d(512), act, dropout,
#             torch.nn.Linear(512, 256), torch.nn.BatchNorm1d(256), act, dropout,            
#             torch.nn.Linear(256, 256), torch.nn.BatchNorm1d(256), act, dropout,            
#             torch.nn.Linear(256, 128), torch.nn.BatchNorm1d(128), act, dropout,            
#             torch.nn.Linear(128, 4)
#         )
#     def forward(self, x):
#         return self.model(x)
 


# In[14]:


# class DNNModel(torch.nn.Module):
#     def __init__(self, input_size, dropout_probability=0.3):
#         super(DNNModel,self).__init__()
#         act = torch.nn.ELU()
#         dropout = torch.nn.Dropout(p=dropout_probability)

#         self.model = torch.nn.Sequential(
#             torch.nn.Linear(input_size, 4096), torch.nn.BatchNorm1d(4096), act, dropout, 
#             torch.nn.Linear(4096, 4096), torch.nn.BatchNorm1d(4096), act, dropout,            
#             torch.nn.Linear(4096, 2048), torch.nn.BatchNorm1d(2048), act, dropout,            
#             torch.nn.Linear(2048, 2048), torch.nn.BatchNorm1d(2048), act, dropout,            
#             torch.nn.Linear(2048, 1024), torch.nn.BatchNorm1d(1024), act, dropout,            
#             torch.nn.Linear(1024, 1024), torch.nn.BatchNorm1d(1024), act, dropout,            
#             torch.nn.Linear(1024, 512), torch.nn.BatchNorm1d(512), act, dropout,
#             torch.nn.Linear(512, 512), torch.nn.BatchNorm1d(512), act, dropout,
#             torch.nn.Linear(512, 4)
#         )
#     def forward(self, x):
#         return self.model(x)
 


# In[15]:


# class DNNModel(torch.nn.Module):
#     def __init__(self, input_size, dropout_probability=0.3):
#         super(DNNModel,self).__init__()
#         act = torch.nn.ELU()
#         dropout = torch.nn.Dropout(p=dropout_probability)

#         self.model = torch.nn.Sequential(
#             torch.nn.Linear(input_size, 4096), torch.nn.BatchNorm1d(4096), act, dropout, 
#             torch.nn.Linear(4096, 4096), torch.nn.BatchNorm1d(4096), act, dropout,            
#             torch.nn.Linear(4096, 4096), torch.nn.BatchNorm1d(4096), act, dropout,            
#             torch.nn.Linear(4096, 2048), torch.nn.BatchNorm1d(2048), act, dropout,            
#             torch.nn.Linear(2048, 2048), torch.nn.BatchNorm1d(2048), act, dropout,            
#             torch.nn.Linear(2048, 4)
#         )
        
#     def forward(self, x):
#         return self.model(x)
 


# In[16]:


# class DNNModel(torch.nn.Module):
#     def __init__(self, input_size, dropout_probability=0.3):
#         super(DNNModel,self).__init__()
#         act = torch.nn.ELU()
#         dropout = torch.nn.Dropout(p=dropout_probability)

#         self.model = torch.nn.Sequential(
#             torch.nn.Linear(input_size, 4096), torch.nn.BatchNorm1d(4096), act, dropout, 
#             torch.nn.Linear(4096, 2048), torch.nn.BatchNorm1d(2048), act, dropout,            
#             torch.nn.Linear(2048, 1024), torch.nn.BatchNorm1d(1024), act, dropout,            
#             torch.nn.Linear(1024, 512), torch.nn.BatchNorm1d(512), act, dropout,            
#             torch.nn.Linear(512, 256), torch.nn.BatchNorm1d(256), act, dropout,            
#             torch.nn.Linear(256, 128), torch.nn.BatchNorm1d(128), act, dropout,            
#             torch.nn.Linear(128, 4)
#         )
        
#     def forward(self, x):
#         return self.model(x)
 


# In[17]:


# # 0.6631845073266462 0.0001

# class DNNModel(torch.nn.Module):
#     def __init__(self, input_size, dropout_probability=0.3):
#         super(DNNModel,self).__init__()
#         act = torch.nn.ELU()
#         dropout = torch.nn.Dropout(p=dropout_probability)

#         self.model = torch.nn.Sequential(
#             torch.nn.Linear(input_size, 4096), torch.nn.BatchNorm1d(4096), act, dropout, 
#             torch.nn.Linear(4096, 4096), torch.nn.BatchNorm1d(4096), act, dropout,            
#             torch.nn.Linear(4096, 4096), torch.nn.BatchNorm1d(4096), act, dropout,            
#             torch.nn.Linear(4096, 2048), torch.nn.BatchNorm1d(2048), act, dropout,            
#             torch.nn.Linear(2048, 2048), torch.nn.BatchNorm1d(2048), act, dropout,            
#             torch.nn.Linear(2048, 1024), torch.nn.BatchNorm1d(1024), act, dropout,            
#             torch.nn.Linear(1024, 512), torch.nn.BatchNorm1d(512), act, dropout,
#             torch.nn.Linear(512, 4)
#         )
        
#     def forward(self, x):
#         return self.model(x)
 


# In[18]:


# # 0.4473953328349374
# class DNNModel(torch.nn.Module):
#     def __init__(self, input_size, dropout_probability=0.3):
#         super(DNNModel,self).__init__()
#         act = torch.nn.ELU()
#         dropout = torch.nn.Dropout(p=dropout_probability)

#         self.model = torch.nn.Sequential(
#             torch.nn.Linear(input_size, 4096), torch.nn.BatchNorm1d(4096), act, dropout, 
#             torch.nn.Linear(4096, 4096), torch.nn.BatchNorm1d(4096), act, dropout,            
#             torch.nn.Linear(4096, 2048), torch.nn.BatchNorm1d(2048), act, dropout,       
#             torch.nn.Linear(2048, 2048), torch.nn.BatchNorm1d(2048), act, dropout, 
#             torch.nn.Linear(2048, 1024), torch.nn.BatchNorm1d(1024), act, dropout,
#             torch.nn.Linear(1024, 1024), torch.nn.BatchNorm1d(1024), act, dropout,            
#             torch.nn.Linear(1024, 512), torch.nn.BatchNorm1d(512), act, dropout,
#             torch.nn.Linear(512, 512), torch.nn.BatchNorm1d(512), act, dropout,
#             torch.nn.Linear(512, 256), torch.nn.BatchNorm1d(256), act, dropout,                        
#             torch.nn.Linear(256, 4)
#         )
#     def forward(self, x):
#         return self.model(x)
 


# In[19]:


# class DNNModel(torch.nn.Module):
#     def __init__(self, input_size, dropout_probability=0.3):
#         super(DNNModel,self).__init__()
#         act = torch.nn.ELU()
#         dropout = torch.nn.Dropout(p=dropout_probability)

#         self.model = torch.nn.Sequential(
#             torch.nn.Linear(input_size, 4096), torch.nn.BatchNorm1d(4096), act, dropout, 
#             torch.nn.Linear(4096, 4096), torch.nn.BatchNorm1d(4096), act, dropout,            
#             torch.nn.Linear(4096, 2048), torch.nn.BatchNorm1d(2048), act, dropout,       

#             torch.nn.Linear(2048, 2048), torch.nn.BatchNorm1d(2048), act, dropout, 
#             torch.nn.Linear(2048, 1024), torch.nn.BatchNorm1d(1024), act, dropout,
#             torch.nn.Linear(1024, 512), torch.nn.BatchNorm1d(512), act, dropout,
#             torch.nn.Linear(512, 4)
#         )
#     def forward(self, x):
#         return self.model(x)
 


# In[20]:


class DNNModel(torch.nn.Module):
    def __init__(self, input_size, dropout_probability=0.3):
        super(DNNModel,self).__init__()
        act = torch.nn.ELU()
        dropout = torch.nn.Dropout(p=dropout_probability)

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 512), torch.nn.BatchNorm1d(512), act, dropout, 
            torch.nn.Linear(512, 1024), torch.nn.BatchNorm1d(1024), act, dropout, 
            torch.nn.Linear(1024, 2048), torch.nn.BatchNorm1d(2048), act, dropout, 
            torch.nn.Linear(2048, 1024), torch.nn.BatchNorm1d(1024), act, dropout,
            torch.nn.Linear(1024, 1024), torch.nn.BatchNorm1d(1024), act, dropout,
            torch.nn.Linear(1024, 2048), torch.nn.BatchNorm1d(2048), act, dropout, 
            torch.nn.Linear(2048, 1024), torch.nn.BatchNorm1d(1024), act, dropout,
            torch.nn.Linear(1024, 512), torch.nn.BatchNorm1d(512), act, dropout,
            torch.nn.Linear(512, 4)
        )
    def forward(self, x):
        return self.model(x)
 


# In[21]:


# class DNNModel(torch.nn.Module):
#     def __init__(self, input_size, dropout_probability=0.3):
#         super(DNNModel,self).__init__()
#         act = torch.nn.ELU()
#         dropout = torch.nn.Dropout(p=dropout_probability)

#         self.model = torch.nn.Sequential(
#             torch.nn.Linear(input_size, 512), torch.nn.BatchNorm1d(512), act, dropout, 
#             torch.nn.Linear(512, 1024), torch.nn.BatchNorm1d(1024), act, dropout, 
#             torch.nn.Linear(1024, 2048), torch.nn.BatchNorm1d(2048), act, dropout, 
#             torch.nn.Linear(2048, 1024), torch.nn.BatchNorm1d(1024), act, dropout,
#             torch.nn.Linear(1024, 1024), torch.nn.BatchNorm1d(1024), act, dropout,
#             torch.nn.Linear(1024, 2048), torch.nn.BatchNorm1d(2048), act, dropout, 
#             torch.nn.Linear(2048, 1024), torch.nn.BatchNorm1d(1024), act, dropout,
#             torch.nn.Linear(1024, 2048), torch.nn.BatchNorm1d(2048), act, dropout, 
#             torch.nn.Linear(2048, 1024), torch.nn.BatchNorm1d(1024), act, dropout,
#             torch.nn.Linear(1024, 512), torch.nn.BatchNorm1d(512), act, dropout,
#             torch.nn.Linear(512, 4)
#         )
#     def forward(self, x):
#         return self.model(x)
 


# In[22]:


# class DNNModel(torch.nn.Module):
#     def __init__(self, input_size, dropout_probability=0.3):
#         super(DNNModel,self).__init__()
#         act = torch.nn.ELU()
#         dropout = torch.nn.Dropout(p=dropout_probability)

#         self.model = torch.nn.Sequential(
#             torch.nn.Linear(input_size, 512), torch.nn.BatchNorm1d(512), act, dropout, 
#             torch.nn.Linear(512, 1024), torch.nn.BatchNorm1d(1024), act, dropout, 
#             torch.nn.Linear(1024, 2048), torch.nn.BatchNorm1d(2048), act, dropout, 
#             torch.nn.Linear(2048, 1024), torch.nn.BatchNorm1d(1024), act, dropout,
#             torch.nn.Linear(1024, 512), torch.nn.BatchNorm1d(512), act, dropout,
#             torch.nn.Linear(512, 4)
#         )
#     def forward(self, x):
#         return self.model(x)
 


# In[23]:


# class DNNModel(torch.nn.Module):
#     def __init__(self, input_size, dropout_probability=0.3):
#         super(DNNModel,self).__init__()
#         act = torch.nn.ELU()
#         dropout = torch.nn.Dropout(p=dropout_probability)

#         self.model = torch.nn.Sequential(
# #             torch.nn.Linear(input_size, 2048), torch.nn.BatchNorm1d(2048), act, dropout, 
#             torch.nn.Linear(input_size, 4096), torch.nn.BatchNorm1d(4096), act, dropout, 
#             torch.nn.Linear(4096, 4096), torch.nn.BatchNorm1d(4096), act, dropout,            
#             torch.nn.Linear(4096, 2048), torch.nn.BatchNorm1d(2048), act, dropout,       
#             torch.nn.Linear(2048, 2048), torch.nn.BatchNorm1d(2048), act, dropout, 
#             torch.nn.Linear(2048, 1024), torch.nn.BatchNorm1d(1024), act, dropout,
#             torch.nn.Linear(1024, 1024), torch.nn.BatchNorm1d(1024), act, dropout,            
#             torch.nn.Linear(1024, 512), torch.nn.BatchNorm1d(512), act, dropout,
#             torch.nn.Linear(512, 512), torch.nn.BatchNorm1d(512), act, dropout,
#             torch.nn.Linear(512, 256), torch.nn.BatchNorm1d(256), act, dropout,                        
#             torch.nn.Linear(256, 4)
#         )
#     def forward(self, x):
#         return self.model(x)
 


# In[24]:


# class DNNModel(torch.nn.Module):
#     def __init__(self, input_size, dropout_probability=0.3):
#         super(DNNModel,self).__init__()
#         act = torch.nn.ELU()
#         dropout = torch.nn.Dropout(p=dropout_probability)

#         self.model = torch.nn.Sequential(
#             torch.nn.Linear(input_size, 3000), torch.nn.BatchNorm1d(3000), act, dropout, 
#              torch.nn.Linear(3000, 3000), torch.nn.BatchNorm1d(3000), act, dropout, 
#              torch.nn.Linear(3000, 2048), torch.nn.BatchNorm1d(2048), act, dropout, 
#             torch.nn.Linear(2048, 2048), torch.nn.BatchNorm1d(2048), act, dropout, 
#             torch.nn.Linear(2048, 1500), torch.nn.BatchNorm1d(1500), act, dropout,
#             torch.nn.Linear(1500, 1500), torch.nn.BatchNorm1d(1500), act, dropout,
#             torch.nn.Linear(1500, 1024), torch.nn.BatchNorm1d(1024), act, dropout,
#             torch.nn.Linear(1024, 1024), torch.nn.BatchNorm1d(1024), act, dropout,            
#             torch.nn.Linear(1024, 1024), torch.nn.BatchNorm1d(1024), act, dropout,            
#             torch.nn.Linear(1024, 512), torch.nn.BatchNorm1d(512), act, dropout,
#             torch.nn.Linear(512, 512), torch.nn.BatchNorm1d(512), act, dropout,
#             torch.nn.Linear(512, 256), torch.nn.BatchNorm1d(256), act, dropout,            
#             torch.nn.Linear(256, 4)
#         )
#     def forward(self, x):
#         return self.model(x)
 


# In[25]:


# class DNNModel(torch.nn.Module):
#     def __init__(self, input_size, dropout_probability=0.3):
#         super(DNNModel,self).__init__()
#         act = torch.nn.ELU()
#         dropout = torch.nn.Dropout(p=dropout_probability)

#         self.model = torch.nn.Sequential(
#             torch.nn.Linear(input_size, 1024), torch.nn.BatchNorm1d(1024), act,
#             torch.nn.Linear(1024, 1024), torch.nn.BatchNorm1d(1024), act, 
#             torch.nn.Linear(1024, 1024), torch.nn.BatchNorm1d(1024), act, 
#             torch.nn.Linear(1024, 512), torch.nn.BatchNorm1d(512), act, 
#             torch.nn.Linear(512, 512), torch.nn.BatchNorm1d(512), act, 
#             torch.nn.Linear(512, 256), torch.nn.BatchNorm1d(256), act, 
#             torch.nn.Linear(256, 4)
#         )
#     def forward(self, x):
#         return self.model(x)
 


# In[26]:


# class DNNModel(torch.nn.Module):
#     def __init__(self, input_size, dropout_probability=0.3):
#         super(DNNModel,self).__init__()
#         act = torch.nn.ELU()
#         dropout = torch.nn.Dropout(p=dropout_probability)

#         self.model = torch.nn.Sequential(
#             torch.nn.Linear(input_size, 1024), torch.nn.BatchNorm1d(1024), act, dropout,
#             torch.nn.Linear(1024, 2048), torch.nn.BatchNorm1d(2048), act, dropout,
#             torch.nn.Linear(2048, 4096), torch.nn.BatchNorm1d(4096), act, dropout,
#             torch.nn.Linear(4096, 4)
#         )
#     def forward(self, x):
#         return self.model(x)
 


# #### DataSet

# In[27]:



class SemiDataset(Dataset):
 def __init__(self, df, fea_cols, y_cols):        
     self.X = df[fea_cols].values
     self.y = df[y_cols].values
     
 def __len__(self):
     return len(self.X)
 
 def __getitem__(self, idx):
     return self.X[idx].astype(np.float32), self.y[idx].astype(np.float32)
 


# #### Trainer

# In[28]:


class Trainer(object):
    def __init__(self, model, criterion, optimizer, scheduler, device):
        self.device = device
        self.model = model#.to(self.device)
        self.criterion = criterion#.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        print(self.model.train())
        pass
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        
    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
    
    def train(self, data_loader):
        self.model.train()
        total_loss = 0
        for data in data_loader:
            X_batch, y_batch = data
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            y_pred = self.model(X_batch)
#             print(y_pred, y_batch)
            
            loss = self.criterion(y_pred, y_batch)
            total_loss = total_loss + loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()
        
        return total_loss / len(data_loader)
    
    def eval(self, data_loader):
        self.model.eval()
        criterion = nn.L1Loss(reduction='mean').to(device)
        total_loss = 0
#         print('valid_loader', len(valid_loader))
        for data in data_loader:
            X_batch, y_batch = data
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            with torch.no_grad():
                y_pred = self.model(X_batch)
#                 loss = self.criterion(y_pred, y_batch)
                loss = criterion(y_pred, y_batch)
                total_loss = total_loss + loss.item()
        return total_loss / len(data_loader)

    def save(self, model_path='checkpoint.pt'):
#         torch.save(self.model.state_dict(), 'checkpoint.pt')
        joblib.dump(self.model, model_path)
        return
    
    def load(self, model_path='checkpoint.pt'):
#         self.model.load_state_dict(torch.load(model_path))
        self.model = joblib.load(model_path)
        return


# #### Train

# In[29]:


model_ts = datetime.now().strftime('%Y%m%dT%H%M%S')
print(model_ts)

print(f'fea_size {len(fea_cols)} layer_cols {layer_cols}')


# In[30]:


torch.manual_seed(81511991154)
torch.initial_seed()


# In[31]:


dataset = SemiDataset(df_model[fea_cols + layer_cols], fea_cols, layer_cols)

train_set, val_set = torch.utils.data.random_split(dataset, [700000, 110000])

print(len(train_set), len(val_set))

# batch_size = 70000
batch_size = 25000
num_workers = 8

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=22000)

print(f'batch_size {batch_size} num_workers {num_workers}')
print(f'train_loader {len(train_loader)} val_loader {len(val_loader)}')


# In[32]:


# train_layer = list(range(10, 300, 20))
# train_layer_1 = list(range(0, 300, 20))

# cond = (df_model['layer_1'].isin(train_layer)) & (df_model['layer_2'].isin(train_layer)) & (df_model['layer_3'].isin(train_layer)) & (df_model['layer_4'].isin(train_layer))
# cond |= (df_model['layer_1'].isin(train_layer_1)) & (df_model['layer_2'].isin(train_layer_1)) & (df_model['layer_3'].isin(train_layer_1)) & (df_model['layer_4'].isin(train_layer_1))
# print(df_model[cond].shape, df_model[~cond].shape)

# # dataset = SemiDataset(df_model[fea_cols + layer_cols], fea_cols, layer_cols)

# # train_set, val_set = torch.utils.data.random_split(dataset, [700000, 110000])

# train_set = SemiDataset(df_model[cond][fea_cols + layer_cols], fea_cols, layer_cols)
# val_set = SemiDataset(df_model[~cond][fea_cols + layer_cols], fea_cols, layer_cols)

# print(len(train_set), len(val_set))

# batch_size = 45000
# num_workers = 8

# train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
# val_loader = DataLoader(dataset=val_set, batch_size=35000)

# print(f'batch_size {batch_size} num_workers {num_workers}')
# print(f'train_loader {len(train_loader)} val_loader {len(val_loader)}')


# In[33]:


# dataset = SemiDataset(df_model[fea_cols + layer_cols], fea_cols, layer_cols)

# train_set = dataset

# print(len(train_set))

# batch_size = 30000
# num_workers = 4

# train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=num_workers)

# print(f'batch_size {batch_size} num_workers {num_workers}')
# print(f'train_loader {len(train_loader)}')


# In[34]:


model = DNNModel(input_size=len(fea_cols), dropout_probability=0).to(device)

# model = joblib.load('model/20200129T032538_0.6176535964012146.model')



# In[35]:


val_loss_min = np.Inf
# val_loss_min = 0.4469637870788574


# In[36]:


# criterion = nn.L1Loss(reduction='mean').to(device)
criterion = nn.MSELoss(reduction='mean').to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
scheduler = StepLR(optimizer, step_size=50, gamma=1.0)
trainer = Trainer(model, criterion, optimizer, scheduler, device)


# In[37]:


lr_list = [
    (0.01, 100),
    (0.001, 100),
    (0.0001, 100),
    (0.00003, 50),
    (0.00001, 50),
    (0.000005, 50),
]


# In[38]:


# trainer.load('model/20200126T224137_1.8200454061681575.model')
# val_loss_min = trainer.eval(val_loader)
# val_loss_min


# In[39]:


# model.load_state_dict(torch.load('model/20200126T111356_0.3797373235225677.pt'))
# trainer.model = model
# val_loss_min = trainer.eval(val_loader)
# val_loss_min


# In[40]:


# trainer.save('model/20200126T111356_0.3797373235225677.model')


# In[41]:


# trainer.eval(val_loader)


# In[ ]:


total_epoch = 10000

for lr, patience in lr_list:
    print(lr, patience)
    if os.path.isfile('stop.flag'):
        print('stop!')
        break
    
    early_stopping = EarlyStopping(patience=patience, min_epoch=1, verbose=True)
    early_stopping.val_loss_min = val_loss_min
    early_stopping.best_score = None if val_loss_min==np.Inf else -val_loss_min 
    
#     criterion = nn.L1Loss(reduction='mean').to(device)

    trainer.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
#     trainer.optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    trainer.scheduler = StepLR(trainer.optimizer, step_size=50, gamma=1.0)
    
    for e in tqdm_notebook(range(total_epoch), total=total_epoch, desc='Epoch'):
        if os.path.isfile('stop.flag'):
            print(f'{e} stop!')
            break

        train_loss = trainer.train(train_loader)
        
        if e % 1 == 0:
            valid_loss = trainer.eval(val_loader)
    #         valid_loss = train_loss

            ts = datetime.now().strftime('%Y%m%dT%H%M%S')
            print(f'[{ts}] Epock {e} / {total_epoch}\t lr {trainer.scheduler.get_lr()[0]}')
            print(f'  train_loss: {train_loss}  valid_loss: {valid_loss}')

            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("\tEarly stopping epoch {}, valid loss {}".format(e, early_stopping.val_loss_min))
                break
            

    model.load_state_dict(torch.load('model/checkpoint.pt'))
#     trainer.load('model/checkpoint.pt')
    val_loss_min = early_stopping.val_loss_min
    
    
    model_path = 'model/{}_{}'.format(model_ts, val_loss_min)
#     joblib.dump(model, '{}.model'.format(model_path))
#     torch.save(model.state_dict(), '{}.pt'.format(model_path))
    trainer.save('{}.model'.format(model_path))
    print(model_path)

    # torch.save(model.state_dict(), f'checkpoint.pt.{train_loss}')
    


# In[ ]:


# 20200128T020612_0.6743781241503629.model
# 20200128T020612_0.9563284516334534.model
# 20200128T020612_1.077946825460954.model
# 20200128T020612_1.254012335430492.model
# 20200128T020612_3.721324747258967.model


# In[ ]:


model.eval()
y_pred = model(torch.Tensor(df_test[fea_cols].values).to(device))    
print(y_pred)


# In[ ]:


ts = datetime.now().strftime('%Y%m%dT%H%M%S')

df_submit = pd.read_csv('input/sample_submission.csv', index_col=0)

df_submit[layer_cols] = y_pred.cpu().detach().numpy()
df_submit.to_csv(f'submit/{ts}_{early_stopping.val_loss_min}.csv')

print(ts, early_stopping.val_loss_min)


# In[ ]:


# !/home/aiden/anaconda3/bin/jupyter nbconvert --to script deep_v1_nodropout.ipynb


# In[ ]:


# !cat deep_v1_nodropout.py


# In[ ]:


# nohup python deep_v1_nodropout.py &


# In[ ]:




