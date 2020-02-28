#!/bin/bash

cd ../Entity_Model/model_lj/
python3 ./bert_classify_train_with_pos_samples_add_feature_extend_trainSet-PreProcess-roberta-large.py
python3 ./bert_classify_train_with_pos_samples_add_feature_extend_trainSet-PreProcess-roberta-large-wwm-ext.py
cp -a ./submit/entity_roberta-large-wwm-ext.csv ../../MethodOne/sub/other/
cp -a ./submit/entity_roberta-large.csv ../../MethodOne/sub/other/

cd ../model_ccy/
python3 ./RoBERTa-large_ccy.py
python3 ./RoBERTa-wwm-ext-large_ccy.py
cp -a ./submission/RoBERTa-large_ccy.csv ../../MethodOne/sub/other/
cp -a ./submission/RoBERTa-wwm-ext-large_ccy.csv ../../MethodOne/sub/other/

cd ../model_zy/
python3 ./zy_robetawmmlargeext.py
cp -a ./result/robertawmmlarge_result_mean.csv ../../MethodOne/single

cd ../../MethodOne
python3 voting.py

echo "entity model voting done."
echo "file five_models_voting_three_method.csv is the submit file."
