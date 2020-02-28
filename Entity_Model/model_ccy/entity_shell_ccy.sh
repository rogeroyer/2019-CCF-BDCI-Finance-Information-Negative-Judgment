#!/bin/bash

python3 ./chinese-bert_chinese_wwm_ccy.py
python3 ./chinese_roberta_wwm_ext_10folds_ccy.py
python3 ./chinese_roberta_wwm_ext_5folds_ccy.py
python3 ./chinese_wwm_ext_ccy.py
python3 ./roberta_zh_112_ccy.py
python3 ./RoBERTa-large_ccy.py
python3 ./RoBERTa-wwm-ext-large_ccy.py

cp -a ./submission/chinese-bert_chinese_wwm_ccy.csv ./model_voting/single
cp -a ./submission/chinese_roberta_wwm_ext_10folds_ccy.csv ./model_voting/sub/other
cp -a ./submission/chinese_roberta_wwm_ext_5folds_ccy.csv ./model_voting/sub/other
cp -a ./submission/chinese_wwm_ext_ccy.csv ./model_voting/sub/other
cp -a ./submission/roberta_zh_112_ccy.csv ./model_voting/sub/other

cd model_voting
python3 ./voting.py

echo "entity voting done."
