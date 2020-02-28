# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:39:24 2019

@author: xiong
"""

import os
import pandas as pd
import numpy as np
import argparse

for i in range(5):
    params = '--model_type bert \
            --model_name_or_path ../../../PreTrainModel/RoBERTa_zh_Large_PyTorch \
            --do_train \
            --do_eval \
            --do_test \
            --data_dir %s \
            --output_dir %s \
            --max_seq_length 512 \
            --split_num 1 \
            --lstm_hidden_size 512 \
            --lstm_layers 1 \
            --lstm_dropout 0.1 \
            --eval_steps 1000 \
            --per_gpu_train_batch_size 4 \
            --gradient_accumulation_steps 1 \
            --warmup_steps 0 \
            --per_gpu_eval_batch_size 32 \
            --learning_rate 8e-6 \
            --adam_epsilon 1e-6 \
            --weight_decay 0 \
            --train_steps 20000 \
            --device_id %d' % ('./data/data_'+str(i), './model_bert_wwm'+str(i), 0)
    ex = os.system("python3 run_bert.py %s" %params)
    print('The fold:', i)

parser = argparse.ArgumentParser()
parser.add_argument("--model_prefix", default='./model_bert_wwm', type=str)
args = parser.parse_args()

k = 5
df = pd.read_csv('data/data_0/test.csv')
df['0'] = 0
df['1'] = 1
for i in range(k):
    temp = pd.read_csv('{}{}/test_pb.csv'.format(args.model_prefix, i))
    df['0'] += temp['label_0'] / k
    df['1'] += temp['label_1'] / k
print('The end for combining.')

test_data_chu = pd.read_csv('./data/round2_test.csv')
#test_data = pd.read_csv('./Test_Data.csv')
def entity_clear(df):
    for index, row in df.iterrows():
        if type(row.entity) == float or type(row.text) == float:
            continue
        entities = row.entity.split(';')
        entities.sort(key =lambda x : len(x))
        n = len(entities)
        #tmp = entities.copy()
        tmp = list(entities)
        for i in range(n):
            entity_tmp = entities[i]
            if i + 1 >= n:
                break
            for entity_tmp2 in entities[i+1:]:
                if entity_tmp2.find(entity_tmp) != -1 and row.text.replace(entity_tmp2,'').find(entity_tmp) == -1:
                    tmp.remove(entity_tmp)
                    break
        df.loc[index, 'entity'] = ';'.join(tmp)
    return df
test_data1 =entity_clear(test_data_chu)

test_data1.dropna(subset=['entity'],inplace=True)

def transform_data(df):
    text_id = []
    entity = []
    for index,row in df.iterrows():
        text_entity = row['entity'].split(';')
        for i in text_entity:
            entity.append(i)
            text_id.append(row['id'])
    df2 = list(zip(text_id, entity))
    df2 = pd.DataFrame(df2)
    df2.columns = ['id','entity']
    return df2
test_data2 = transform_data(test_data1)
test_data2.drop_duplicates(['id','entity'], 'first', inplace=True)

features = ['id','entity','0','1']
result0 = df[features]
result0['pre_label'] = np.argmax(result0[['0','1']].values, -1)
sub = pd.merge(result0,test_data2,how='right',on=['id','entity'])

def return_list(group):
    return ';'.join(list(group))

sub = sub[['id', 'entity', 'pre_label']]
sub_label = sub[sub['pre_label'] == 1].groupby(['id'], as_index=False)['entity'].agg({'key_entity': return_list})

test_2 = pd.read_csv('./data/round2_test.csv', encoding='utf-8')
submit = test_2[['id']]
submit = submit.merge(sub_label, on='id', how='left')
submit['negative'] = submit['key_entity'].apply(lambda index: 0 if index is np.nan else 1)
submit = submit[['id', 'negative', 'key_entity']]

before = pd.read_csv('../../Emotion_Model/submit/emotion_voting_three_models.csv', encoding='utf-8')
negative_sample = before[before['negative'] == 1][['id']]

bert_negative_sample = submit[submit['negative'] == 1][['id', 'key_entity']]

negative_sample = negative_sample.merge(bert_negative_sample, on='id', how='left')
id_set = set(negative_sample[negative_sample['key_entity'].isnull()]['id'])

sub = df[features]
res = sub[['id', 'entity', '1']].rename(columns={'entity': 'entity_label'})
res = res[res['id'].isin(id_set)]
res = res.sort_values(by=['id', '1'], ascending=False).drop_duplicates('id', keep='first')

negative_sample = negative_sample.merge(res[['id', 'entity_label']], on='id', how='left')
negative_sample['key_entity'] = negative_sample.apply(lambda index: index.entity_label if index.key_entity is np.nan else index.key_entity, axis=1)
negative_sample.drop(['entity_label'], axis=1, inplace=True)
test_3 = pd.read_csv('./data/round2_test.csv', encoding='utf-8')
su = test_3[['id']]
su = su.merge(negative_sample, on='id', how='left')
su['negative'] = su['key_entity'].apply(lambda index: 0 if index is np.nan else 1)
su = su[['id', 'negative', 'key_entity']]
su[su['negative'] == 1]
su.to_csv('./result/submit.csv', encoding='utf-8', index=None)     #######right#######