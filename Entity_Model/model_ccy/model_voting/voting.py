import os
import numpy as np
import pandas as pd
from collections import Counter

sub_path = 'sub/'
teamates = os.listdir(sub_path)

data = pd.read_csv('./single/chinese-bert_chinese_wwm_ccy.csv', encoding='utf-8').rename(columns={'negative': 'negative_1', 'key_entity': 'key_entity_1'})

index = 2
for member in teamates:
        member_files = sub_path + member + '/'
        member_sub_files = os.listdir(member_files)
        for file in member_sub_files:
                sub = pd.read_csv(member_files+file, encoding='utf-8').rename(columns={'negative': 'negative_' + str(index), 'key_entity': 'key_entity_' + str(index)})
                data = data.merge(sub, on='id', how='left')
                index += 1
print(data)

print(data[data['negative_1'] == 1].shape)
print(data[data['negative_2'] == 1].shape)
print(data[data['negative_3'] == 1].shape)


# for row in data.itertuples:
negatives = ['negative_' + str(index) for index in range(1, index, 1)]
key_entitys = ['key_entity_' + str(index) for index in range(1, index, 1)]

ids = []
voting_entitys = []

thresh = int(index / 2)   # 阈值：保留词的最小出现次数
count = 0

for row in range(len(data)):
        negative = Counter()
        key_entity = Counter()
        for k in range(0, index-1, 1):
                negative[data.ix[row][negatives[k]]] += 1

        # print(negative)
        if (len(negative) == 1) & (data.ix[row]['negative_1'] == 1):

                for k in range(0, index-1, 1):
                        for entity in data.ix[row][key_entitys[k]].split(';'):
                                key_entity[entity] += 1

                # print(key_entity)
                entitys = []
                words = list(key_entity.keys())
                for word in words:
                        if key_entity[word] >= thresh:
                                entitys.append(word)
                if entitys == []:
                        entitys.append(key_entity.most_common(1)[0][0])
                entitys = list(set(entitys))
                voting_entitys.append(';'.join(entitys))
                ids.append(data.ix[row]['id'])
                count += 1


print(count)
voted = pd.DataFrame({'id': ids, 'key_entity': voting_entitys})
print(voted)

submit = data[['id', 'negative_1', 'key_entity_1']].rename(columns={'negative_1': 'negative'})
submit = submit.merge(voted, on='id', how='left')

submit['key_entity'] = submit.apply(lambda index: index.key_entity_1 if index.key_entity is np.nan else index.key_entity, axis=1)

print(submit)

def get_sun(x):
    if str(x)=='nan':
        return 1
    x=x.strip(';').split(';')
    new_x=[]
    for i in x:
        new_x.append(i.strip('?'))
    tag=int(len(set(new_x))==len(set(x)))
    return tag

submit['key_entity_tag_sun']=submit['key_entity'].apply(lambda x:get_sun(x))
print(submit[submit['key_entity_tag_sun']==0])


"""去?的子串函数'"""
def delete_sun(x,tag):
    if tag==1:
        return x
    x=x.split(';')
    #找到带？的实体
    p_e=''
    for i in x:
        if '?' in i:
            p_e=i
            break
    new_x=[]
    for i in x:
        if i in p_e and i!=p_e:
            continue
        new_x.append(i)
    return ';'.join(new_x)

submit['key_entity']=list(map(lambda x, tag: delete_sun(x, tag), submit['key_entity'], submit['key_entity_tag_sun']))
print(submit[submit['key_entity_tag_sun']==0])


print(submit)
submit[['id', 'negative', 'key_entity']].to_csv('three_model_voting_ccy.csv', index=None)
print('store done.')

# data =  pd.read_csv('./voting_drop_partial_words_delPosWords_replaceWithTrain_delYiZiDai.csv', encoding='utf-8')
# print(data[(data['negative'] == 1) & (data['key_entity'] is np.nan)])
