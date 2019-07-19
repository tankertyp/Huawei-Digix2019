import numpy as np
import pandas as pd
import tensorflow as tf
import ctrNet    # ctrNet-Tool神器
import random
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from src import misc_utils as utils
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
#warnings.filterwarnings("ignore")

import os
import gc


dtypes = {
	'label':	'int64',
	'uId':		'str',
	'adId':		'int64',
	'operTime':	'object',
	'siteId':	'int64',
	'slotId':	'int64',
	'contentId':	'int64',
	'netType':	'int64'
}

user_dtypes = {
	'uId':           'category',
	'age':           'int64',
	'gender':        'int64',
	'city':	         'int64',
	'province':      'int64',
	'phoneType':     'int64',
	'carrier':       'int64'
}


ad_dtypes = {
	'adId':		'int64',
	'billId':	'category',
	'primId':	'int64',
	'creativeType':	'int64',	
	'interType':	'int64',
	'spreadAppId':  'int64'
}

content_dtypes = {
	'contentId':	'int64',
	'firstClass':	'category',
	'secondClass':	'category'
}




# 用户数据
user_info = pd.read_csv('./input/user_info.csv',header=None,dtype=user_dtypes,sep=',')
user_info.columns = [k for k, v in user_dtypes.items()]
print("user_info is over")
print(type(user_info['uId'][0]))


print(user_info.shape)
#user_info = user_info.dropna()
#print(user_info.shape)
print(user_info.head(10))

# 广告数据
ad_info = pd.read_csv('./input/ad_info.csv',header=None,dtype=ad_dtypes,sep=',')
ad_info.columns = [k for k, v in ad_dtypes.items()]
print("ad_info is over")

print(ad_info.shape)
ad_info = ad_info.dropna()
print(ad_info.shape)
print(ad_info.head(10))

# 素材信息数据
content_info = pd.read_csv('./input/content_info.csv',header=None,dtype=content_dtypes,sep=',')
content_info.columns = [k for k, v in content_dtypes.items()]
print("content_info is over")

print(content_info.shape)
content_info = content_info.dropna()
print(content_info.shape)

print(content_info.head(10))


# 训练集
train_df = pd.read_csv('./input/train_20190518.csv',header=None,dtype=dtypes,sep=',',nrows = 96000000)

# 随机采样
#train_df = train_df.sample(n=4000000).reset_index()
print("train_df is over")
train_df.columns = [k for k, v in dtypes.items()]
#print(type(train_df['operTime'][0]))


print("Train Set before drop duplicates:")
print(train_df.shape)
# 训练集去重
train_df = train_df.drop_duplicates()
print("Train Set after drop duplicates:")
print(train_df.shape)


#del train_df['operTime']
#del train_df['contentId']
print(train_df.head(10))


test_df = pd.read_csv('./input/test_20190518.csv',header=None,dtype=dtypes,sep=',')
print("test_df is over")
test_df.columns = [k for k, v in dtypes.items()]
#del test_df['operTime']
#del test_df['contentId']

test_df = test_df.rename(columns={'label':'testId'})

print(test_df.head(10))
#test_df['label']=[0]*len(test_df)

#sub_df = pd.DataFrame(columns=['id','probability'])
#sub_df['id'] = test_df['testId']


train_df = pd.merge(train_df,user_info,how='left',on="uId")
print("train_df after merging with user_info:")
print(train_df.head())

train_df = pd.merge(train_df,ad_info,how='left',on="adId")
print("train_df after merging with ad_info:")
print(train_df.head())

train_df['label2'] = train_df['label']
train_df.pop('label')
train_df = train_df.rename(columns={'label2':'label'})

print("After move label column")
print(train_df.head())
#features=train_df.columns.tolist()[1:-1]


test_df = pd.merge(test_df,user_info,how='left',on="uId")
print("test_df after merging with user_info:")
print(test_df.head())

test_df = pd.merge(test_df,ad_info,how='left',on="adId")
print("test_df after merging with ad_info:")
#print(test_df.head())

test_df['label']=[0]*len(test_df)
print(test_df.head())

sub_df = pd.DataFrame(columns=['id','probability'])
sub_df['id'] = test_df['testId']


train_df.to_csv('./output/train_df_todo.csv',index=False)
test_df.to_csv('./output/test_df_todo.csv',index=False)

#print(train_df['operTime'])

del train_df['operTime']
del train_df['contentId']
del train_df['uId']
del train_df['adId']
del test_df['operTime']
del test_df['contentId']
del test_df['testId']
del test_df['uId']
del test_df['adId']
'''
train_df.pop('operTime')
train_df.pop('contentId')
train_df.pop('uId')
train_df.pop('adId')
test_df.pop('operTime')
test_df.pop('contentId')
test_df.pop('uId')
test_df.pop('adId')
'''

# Feature Engineering ################################################################################


from datetime import datetime
#train_df['month'] = pd.to_datetime(train_df['operTime']).month
#train_df['day'] = pd.to_datetime(train_df['operTime']).day

#print("New feature month:")
#print(train_df['month'].head())
#print("New feature day:")
#print(train_df['day'].head())




gc.collect()
#sub_df = pd.DataFrame(columns=['id','probability'])
#sub_df['id'] = test_df['id']
#del test_df['id']


print("Drop nan before training...")
train_df = train_df.dropna()
#train_df = train_df.sample(n=4000000)
print(train_df.shape)



# train_df/test_df的特征列
features=train_df.columns.tolist()[0:-1]


print("features are :")
print(features)



categorical_columns = ['siteId','slotId','netType','age','gender','city','province','phoneType','carrier','billId','primId','creativeType','interType','spreadAppId']
#categorical_columns = features.remove('label')


for i in categorical_columns:
    train_df[i] = train_df[i].astype('category')
    test_df[i] = test_df[i].astype('category')



# Label encoding
indexer = {}
for col in tqdm(categorical_columns):
    if col == 'label': continue
    _, indexer[col] = pd.factorize(train_df[col])
    
for col in tqdm(categorical_columns):
    if col == 'label': continue
    train_df[col] = indexer[col].get_indexer(train_df[col])
    test_df[col] = indexer[col].get_indexer(test_df[col])




target = train_df['label']
del train_df['label']


# CUDA配置，使用CPU可以注释掉
os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"]='0'
#model=ctrNet.build_model(hparam)

param = {'num_leaves': 60,
         'min_data_in_leaf': 60, 
         'objective':'binary',
         'max_depth': 6,
         'learning_rate': 0.05,
         "boosting": "gbdt",
         #"feature_fraction": 0.8,
         "bagging_freq": 1,
         "bagging_fraction": 0.8 ,   # 数据采样
         "bagging_seed": 11,
         "metric": 'auc',
         "lambda_l1": 0.1,
         "random_state": 133,
         "verbosity": -1}

max_iter = 5

folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(len(train_df))
preds = np.zeros(len(test_df))

import time
start = time.time()
feature_importance_df = pd.DataFrame()
start_time= time.time()
score = [0 for _ in range(folds.n_splits)]

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features],
                           label=target.iloc[trn_idx],
                           categorical_feature = categorical_columns
                          )
    val_data = lgb.Dataset(train_df.iloc[val_idx][features],
                           label=target.iloc[val_idx],
                           categorical_feature = categorical_columns
                          )

    num_round = 10000
    clf = lgb.train(param,
                    trn_data,
                    num_round,
                    valid_sets = [trn_data, val_data],
                    verbose_eval=200,
                    early_stopping_rounds = 400)
    
    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    # we perform predictions by chunks
    initial_idx = 0
    chunk_size = 1000000
    current_pred = np.zeros(len(test_df))
    while initial_idx < test_df.shape[0]:
        final_idx = min(initial_idx + chunk_size, test_df.shape[0])
        idx = range(initial_idx, final_idx)
        current_pred[idx] = clf.predict(test_df.iloc[idx][features], num_iteration=clf.best_iteration)
        initial_idx = final_idx
    preds += current_pred / min(folds.n_splits, max_iter)
   
    print("time elapsed: {:<5.2}s".format((time.time() - start_time) / 3600))
    score[fold_] = metrics.roc_auc_score(target.iloc[val_idx], oof[val_idx])
    if fold_ == max_iter - 1: break
        
if (folds.n_splits == max_iter):
    print("CV score: {:<8.5f}".format(metrics.roc_auc_score(target, oof)))
else:
     print("CV score: {:<8.5f}".format(sum(score) / max_iter))


cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,25))
sns.barplot(x="importance",
            y="feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')



sub_df['probability'] = preds
#sub_df.index = np.arange(1, len(sub_df)+1)
sub_df.to_csv('./submission_lgb.csv',float_format='%.6f',index=False)
print('Submission is done !!!')

