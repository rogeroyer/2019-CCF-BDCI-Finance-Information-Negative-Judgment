#! -*- coding:utf-8 -*-
import os
import re
import gc
import sys
import json
import codecs
import random
import warnings
import numpy as np
import pandas as pd
import textdistance
from tqdm import tqdm
import tensorflow as tf
from random import choice
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

import keras.backend as K
from keras.layers import *
from keras.callbacks import *
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
from keras_bert import load_trained_model_from_checkpoint, Tokenizer

tqdm.pandas()
seed = 2019
random.seed(seed)
tf.set_random_seed(seed)
np.random.seed(seed)
warnings.filterwarnings('ignore')


data_path = '../dataSet/'
train = pd.read_csv(data_path + 'Round2_train.csv', encoding='utf-8')
test = pd.read_csv(data_path + 'round2_test.csv', encoding='utf-8')

train_preliminary = pd.read_csv(data_path + 'Train_Data.csv', encoding='utf-8')
train = pd.concat([train, train_preliminary], axis=0, ignore_index=True)
train = train.drop_duplicates(['title', 'text', 'entity', 'negative', 'key_entity'])

def duplicate_entity(entity):
    def is_empty(x):
        return (x != '') & (x != ' ')

    if entity is np.nan:
        return entity
    else:
        entity = filter(is_empty, entity.split(';'))
        return ';'.join(list(set(entity)))

train['entity'] = train['entity'].apply(lambda index: duplicate_entity(index))
test['entity'] = test['entity'].apply(lambda index: duplicate_entity(index))

test['text'] = test.apply(lambda index: index.title if index.text is np.nan else index.text, axis=1)

train.fillna('', inplace=True)
train['title'] = train['title'].map(lambda index: index.replace(' ', ''))
train['text'] = train['text'].map(lambda index: index.replace(' ', ''))
train['title_len'] = train['title'].map(lambda index: len(index))
train['text_len'] = train['text'].map(lambda index: len(index))

test.fillna('', inplace=True)
test['title'] = test['title'].map(lambda index: index.replace(' ', ''))
test['text'] = test['text'].map(lambda index: index.replace(' ', ''))
test['title_len'] = test['title'].map(lambda index: len(index))
test['text_len'] = test['text'].map(lambda index: len(index))

distance = textdistance.Levenshtein(external=False)
train['distance'] = train.progress_apply(lambda index: distance(index.title, index.text), axis=1)   # distance   similarity
test['distance'] = test.progress_apply(lambda index: distance(index.title, index.text), axis=1)   # distance   similarity

train['title_in_text'] = train.progress_apply(lambda index: 1 if index.text.find(index.title) != -1 else 0, axis=1)
test['title_in_text'] = test.progress_apply(lambda index: 1 if index.text.find(index.title) != -1 else 0, axis=1)

train['text'] = train.progress_apply(lambda index: index.title + ';' + index.text if (index.title_len != 0) & (index.distance > 200) & (index.title_in_text != 1) else index.text, axis=1)
test['text'] = test.progress_apply(lambda index: index.title + ';' + index.text if (index.title_len != 0) & (index.distance > 200) & (index.title_in_text != 1) else index.text, axis=1)

train['text'] = train.progress_apply(lambda index: index.title + ';' + index.text if index.title_len + index.text_len < 512 else index.text, axis=1)
test['text'] = test.progress_apply(lambda index: index.title + ';' + index.text if index.title_len + index.text_len < 512 else index.text, axis=1)

train.drop(['title_len', 'distance', 'title_in_text', 'text_len'], axis=1, inplace=True)
test.drop(['title_len', 'distance', 'title_in_text', 'text_len'], axis=1, inplace=True)

train['entity_len'] = train['entity'].progress_apply(lambda index: len(index))
test['entity_len'] = test['entity'].progress_apply(lambda index: len(index))

# 替换实体链长度超过512的样本
train['entity'] = train.apply(lambda index: '' if index.entity_len > 509 else index.entity, axis=1)
test['entity'] = test.apply(lambda index: '' if index.entity_len > 509 else index.entity, axis=1)

train['entity_len'] = train['entity'].progress_apply(lambda index: len(index))
test['entity_len'] = test['entity'].progress_apply(lambda index: len(index))


# 增加实体替换
count = 0
def get_content(x,y):
    global count
    try:
        if y == '':  # y == ''  ??  str(y)=='nan'
            return x
        y=y.split(";")
        y = sorted(y, key=lambda i:len(i),reverse=True)
        for i in y:
            x = '实体词'.join(x.split(i))
        return x
    except:
        count += 1
        return y

print("samples's num with empty entity:", count)
train['text'] = list(map(lambda x,y: get_content(x,y),train['text'], train['entity']))
test['text'] = list(map(lambda x,y: get_content(x,y),test['text'], test['entity']))

######################################################################
maxlen = 509
bert_path = '../../PreTrainModel/chinese_wwm_ext_L-12_H-768_A-12/'    # 预训练模型路径

config_path = bert_path + 'bert_config.json'
checkpoint_path = bert_path + 'bert_model.ckpt'
dict_path = bert_path + 'vocab.txt'

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)  # 给每个token 按序编号

class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

tokenizer = OurTokenizer(token_dict)

def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X])

class data_generator:
    def __init__(self, data, batch_size=4, shuffle=True):    # 8
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            
            if self.shuffle:
                np.random.shuffle(idxs)
            
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                first_text = d[0]
                second_text = d[2][:maxlen - d[1]]
                x1, x2 = tokenizer.encode(first=first_text, second=second_text)   # , max_len=512
                y = d[3]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2, padding=0)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y[:, 0, :]
                    [X1, X2, Y] = [], [], []

from keras.metrics import top_k_categorical_accuracy
from keras.metrics import categorical_accuracy

def acc_top2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


def f1_metric(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
                    

def build_bert(nclass):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

    for l in bert_model.layers:
#         print(l)
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)
    p = Dense(nclass, activation='softmax', kernel_initializer=glorot_uniform(seed=seed))(x)

    model = Model([x1_in, x2_in], p)
    model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(1e-5),                # lr: 5e-5   3e-5   2e-5    epoch: 3, 4    batch_size: 16, 32    
                  metrics=['accuracy', f1_metric])
    print(model.summary())
    return model


from keras.utils import to_categorical

DATA_LIST = []
for data_row in train.iloc[:].itertuples():
    DATA_LIST.append((data_row.entity, data_row.entity_len, data_row.text, to_categorical(data_row.negative, 2)))
DATA_LIST = np.array(DATA_LIST)

DATA_LIST_TEST = []
for data_row in test.iloc[:].itertuples():
    DATA_LIST_TEST.append((data_row.entity, data_row.entity_len, data_row.text, to_categorical(0, 2)))
DATA_LIST_TEST = np.array(DATA_LIST_TEST)


def run_cv(nfold, data, data_label, data_test):
    kf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed).split(data, train['negative'])
    train_model_pred = np.zeros((len(data), 2))
    test_model_pred = np.zeros((len(data_test), 2))

    for i, (train_fold, test_fold) in enumerate(kf):
        X_train, X_valid, = data[train_fold, :], data[test_fold, :]

        model = build_bert(2)
        early_stopping = EarlyStopping(monitor='val_acc', patience=3)
        plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5, patience=1)
        checkpoint = ModelCheckpoint('./model/' + str(i) + '.hdf5', monitor='val_acc', 
                                         verbose=2, save_best_only=True, mode='max',save_weights_only=True)
        
        train_D = data_generator(X_train, shuffle=True)
        valid_D = data_generator(X_valid, shuffle=False)
        test_D = data_generator(data_test, shuffle=False)

        model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),   ## ?? ##
            epochs=5,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            callbacks=[early_stopping, plateau, checkpoint],
            verbose=2
        )
        
        model.load_weights('./model/' + str(i) + '.hdf5')
        
        # return model
        val = model.predict_generator(valid_D.__iter__(), steps=len(valid_D),verbose=0)
        train_model_pred[test_fold, :] = val
        print('{}th f1_score:{}'.format(i+1, f1_score(train['negative'].values[test_fold], [np.argmax(index) for index in val])))
        print('{}th accuracy:{}'.format(i+1, accuracy_score(train['negative'].values[test_fold], [np.argmax(index) for index in val])))
        test_model_pred += model.predict_generator(test_D.__iter__(), steps=len(test_D),verbose=0)
        
        del model; gc.collect()
        K.clear_session()
        
    return train_model_pred, test_model_pred


train_model_pred, test_model_pred = run_cv(5, DATA_LIST, None, DATA_LIST_TEST)
np.save('weights/train_bert_negtive_extend_trainSet-bert_wwm.npy', train_model_pred)
np.save('weights/test_bert_negtive__extend_trainSet-bert_wwm.npy', test_model_pred)


test_prob = [np.argmax(index) for index in test_model_pred]
test_index = test[['id']]
test_index['negative'] = test_prob
test_index['key_entity'] = ['' for index in range(len(test_index))]
test_index.to_csv('./submit/emotion_res_2.csv', encoding='utf-8', index=None)
print('store over')