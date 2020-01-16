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

device = torch.device('cpu')
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

df_train = pd.read_csv('input/train.csv', dtype=np.float32)
df_test = pd.read_csv('input/test.csv', dtype=np.float32)
print(df_train.shape, df_test.shape)


y_cols = [c for c in df_train.columns if 'layer_' in c]
fea_cols = [c for c in df_train.columns if c not in y_cols]

len(fea_cols), len(y_cols)

df_model = df_train

class DNNModel(torch.nn.Module):
    def __init__(self, input_size, dropout_probability=0.3):
        super(DNNModel,self).__init__()
        relu = torch.nn.ReLU()
        dropout = torch.nn.Dropout(p=dropout_probability)

        self.model = torch.nn.Sequential(
#             torch.nn.Linear(input_size, 4),
#             torch.nn.Linear(input_size, 1),
            torch.nn.Linear(input_size, 200), relu, torch.nn.BatchNorm1d(200), dropout, 
            torch.nn.Linear(200, 150), relu, torch.nn.BatchNorm1d(150), dropout,
            torch.nn.Linear(150, 100), relu, torch.nn.BatchNorm1d(100), dropout,            
            torch.nn.Linear(100, 64), relu, torch.nn.BatchNorm1d(64), dropout,
            torch.nn.Linear(64, 32), relu, torch.nn.BatchNorm1d(32), dropout,            
            torch.nn.Linear(32, 4)
                           )
    def forward(self, x):
        return self.model(x)
    

model_ts = datetime.now().strftime('%Y%m%dT%H%M%S')
print(model_ts)

print(f'fea_size {len(fea_cols)} y_cols {y_cols}')

model = DNNModel(input_size=len(fea_cols), dropout_probability=0.5).to(device)
    
criterion = nn.L1Loss(reduction='mean').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
scheduler = StepLR(optimizer, step_size=400, gamma=0.97)



X_batch = torch.Tensor(df_model[fea_cols].values).float().to(device)
y_batch = torch.Tensor(df_model[y_cols].values).float().to(device)

model.load_state_dict(torch.load('checkpoint.pt'))

total_epoch = 100000
model.train()

for e in tqdm_notebook(range(total_epoch), total=total_epoch, desc='Epoch'):
    y_pred = model(X_batch)

    loss = criterion(y_pred, y_batch)
    print(f'Epock {e} / {total_epoch} loss: {loss.item()}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    if e % 1000 == 0:
        torch.save(model.state_dict(), 'checkpoint.pt')

torch.save(model.state_dict(), 'checkpoint.pt')




