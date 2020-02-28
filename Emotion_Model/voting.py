import os
import path
import numpy as np
import pandas as pd
from collections import Counter

sub_path = './submit/'

sub1 = pd.read_csv(sub_path + 'emotion_res_1.csv', encoding='utf-8')[['id', 'negative']]
sub2 = pd.read_csv(sub_path + 'emotion_res_2.csv', encoding='utf-8')[['id', 'negative']]
sub3 = pd.read_csv('./roberta_wwm_large_ext_emotion_xiong/result/submit_emotion.csv', encoding='utf-8')[['id', 'negative']]

sub1.columns = ['id', 'negative_1']
sub2.columns = ['id', 'negative_2']
sub3.columns = ['id', 'negative_3']

sub = sub1.merge(sub2, on='id', how='left')
sub = sub.merge(sub3, on='id', how='left')
print(sub)

def vote(value_1, value_2, value_3):
    count = Counter()
    count[value_1] += 1
    count[value_2] += 1
    count[value_3] += 1
    # print(count)
    return count.most_common(1)[0][0]

sub['negative'] = sub.apply(lambda index: vote(index.negative_1, index.negative_2, index.negative_3), axis=1)
sub['key_entity'] = [np.nan for index in range(len(sub))]
print(sub)
sub[['id', 'negative', 'key_entity']].to_csv('./submit/emotion_voting_three_models.csv', encoding='utf-8', index=None)
print('store done.')
print(sub[sub['negative'] == 1].shape)
