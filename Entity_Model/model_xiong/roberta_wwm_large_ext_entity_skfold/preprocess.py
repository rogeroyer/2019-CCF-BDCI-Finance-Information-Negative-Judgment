import pandas as pd
import os
import random
import numpy as np
import re
import textdistance
from sklearn.model_selection import KFold, StratifiedKFold

train_df1 = pd.read_csv('./data/Train_Data.csv', encoding='utf-8')
train_df2 = pd.read_csv('./data/Round2_train.csv', engine='python', encoding='utf-8')
train_df = pd.concat([train_df1, train_df2], axis=0, sort=True)
test_data = pd.read_csv('./data/round2_test.csv', engine='python', encoding='utf-8')
train_df = train_df[train_df['entity'].notnull()]
test_data = test_data[test_data['entity'].notnull()]
train_df = train_df.drop_duplicates(['title','text','entity','negative','key_entity'])

print(train_df.shape)
print(test_data.shape)

def get_or_content(y, z):
    s=''
    if str(y) != 'nan':
        s += y
    if str(z) != 'nan':
        s += z
    return s

train_df['content'] = list(map(lambda y,z: get_or_content(y,z), train_df['title'], train_df['text']))
test_data['content'] = list(map(lambda y,z: get_or_content(y,z), test_data['title'], test_data['text']))

def entity_clear_row(entity, content):
    entities = entity.split(';')
    entities.sort(key=lambda x:len(x))
    n = len(entities)
    tmp = entities.copy()
    for i in range(n):
        entity_tmp = entities[i]
        if len(entity_tmp) <= 1:
            tmp.remove(entity_tmp)
            continue
        if i+1 >= n:
            break
        for entity_tmp2 in entities[i+1:]:
            if entity_tmp2.find(entity_tmp) != -1 and (entity_tmp2.find('?') != -1 or content.replace(entity_tmp2,'').find(entity_tmp) == -1):
                tmp.remove(entity_tmp)
                break
    return ';'.join(tmp)
train_df['entity'] = list(map(lambda entity,content:entity_clear_row(entity, content), train_df['entity'], train_df['content']))
test_data['entity'] = list(map(lambda entity,content:entity_clear_row(entity, content), test_data['entity'], test_data['content']))

fea_tmp = ['id', 'title', 'text', 'entity', 'negative', 'key_entity']
train_data = train_df[fea_tmp]
test_df = test_data[['id', 'title', 'text', 'entity']]

train_data.dropna(subset=['entity'], inplace=True)
train_data.reset_index(drop=True, inplace=True)
test_df.dropna(subset=['entity'], inplace=True)
test_df.reset_index(drop=True, inplace=True)

labels = train_data.negative.value_counts()
test_df['negative'] = 0

train_data['title'] = train_data['title'].fillna('无')
train_data['text'] = train_data['text'].fillna('无')
test_df['title'] = test_df['title'].fillna('无')
test_df['text'] = test_df['text'].fillna('无')

content_train = []
for index, row in train_data.iterrows():
    train_text_clear = re.sub(r'http.*$', "", row['text'])
    if abs(len(train_text_clear) - len(row['title'])) <= 4:
        content_train.append(row['title'])
    else:
        content_train.append(row['title'] + '。' + train_text_clear)
train_data['content'] = content_train
content_test = []
for index, row in test_df.iterrows():
    test_text_clear = re.sub(r'http.*$', "", row['text'])
    if abs(len(test_text_clear) - len(row['title'])) <= 4:
        content_test.append(row['title'])
    else:
        content_test.append(row['title'] + '。' + test_text_clear)
test_df['content'] = content_test

#对训练集数据进行转化为文本+单实体形式
def transform_data(df):
    label = []
    text_id = []
    content = []
    entity = []
    entity_lian = []
    for index, row in df.iterrows():
        text_entity = row['entity'].split(';')
        if row['key_entity'] is not np.nan:
            key_entity = row['key_entity'].split(';')
        else:
            key_entity = []
        for i in text_entity:
            if i in key_entity:
                label.append(1)
            else:
                label.append(0)
            entity.append(i)
            text_id.append(row['id'])
            content.append(row['content'])
            entity_lian.append(row['entity'])
    df2 = list(zip(text_id, content, entity, label, entity_lian))
    df2 =pd.DataFrame(df2)
    df2.columns = ['id', 'content', 'entity', 'label', 'entity_lian']
    return df2

#对无标签的数据转换为文本+单实体的形式
def transform_test_data(df):
    label = []
    text_id = []
    entity = []
    entity_lian = []
    content = []
    for index, row in df.iterrows():
        text_entity = row['entity'].split(';')
        for i in text_entity:
            entity.append(i)
            text_id.append(row['id'])
            content.append(row['content'])
            entity_lian.append(row['entity'])
            label.append(0)
    df2 = list(zip(text_id, content, entity, label, entity_lian))
    df2 = pd.DataFrame(df2)
    df2.columns = ['id', 'content', 'entity', 'label', 'entity_lian']
    return df2
test_df = transform_test_data(test_df)

#将文本中的其他实体替换为‘其他实体’
def get_other_content(x, y):
    entitys = x.strip(';').split(';')
    if len(entitys) <= 1:
        return np.nan
    l = []
    for e in entitys:
        if e != y:
            l.append(e)
    return ';'.join(l)

def get_content(x, y, z):
    if str(y) == 'nan':
        return x
    y = y.split(';')
    y = sorted(y, key=lambda x: len(x), reverse=True)
    if x.count(z) <= 1:
        return x
    for i in y:
        if i not in z:
            x = '其他实体'.join(x.split(i))
    return x

test_df['other_entity'] = list(map(lambda x,y: get_other_content(x,y), test_df['entity_lian'], test_df['entity']))
test_df['content'] = list(map(lambda x, y ,z:get_content(x,y,z), test_df['content'], test_df['other_entity'], test_df['entity']))

features = ['id', 'content', 'entity', 'label']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

i=0
for train,valid in skf.split(train_data, train_data.negative):
    if not os.path.exists('./data/data_' + str(i)):
        os.mkdir('./data/data_'+str(i))
    train_z = train_data.iloc[train,:]
    dev_df = train_data.iloc[valid,:]
    count_train = 0
    train_z = transform_data(train_z)
    count_train += len(train_z)
    dev_df = transform_data(dev_df)
    count_train += len(dev_df)
    train_z['other_entity'] = list(map(lambda x,y: get_other_content(x,y), train_z['entity_lian'], train_z['entity']))
    dev_df['other_entity'] = list(map(lambda x,y: get_other_content(x,y), dev_df['entity_lian'], dev_df['entity']))
    train_z['content'] = list(map(lambda x,y,z: get_content(x,y,z), train_z['content'], train_z['other_entity'], train_z['entity']))
    dev_df['content'] = list(map(lambda x,y,z: get_content(x,y,z), dev_df['content'], dev_df['other_entity'], dev_df['entity']))
    
    train_z[features].to_csv('./data/data_{}/train.csv'.format(i))
    dev_df[features].to_csv('./data/data_{}/dev.csv'.format(i))
    test_df[features].to_csv('./data/data_{}/test.csv'.format(i))
    i += 1
    print('number', i)
    print('count_train', count_train)
    print('count_test', len(test_df))