#!/bin/bash

python3 ./zy_bert_base.py
python3 ./zy_bert_wmmext.py
python3 ./zy_roberta.py
python3 ./zy_robertawmmext.py
python3 ./zy_robetalarge.py
python3 ./zy_robetawmmlargeext.py

cp -a ./result/bertbase_result_mean.csv ./model_voting/single
cp -a ./result/bertwmmext_result_mean.csv ./model_voting/sub/other
cp -a ./result/roberta_result_mean.csv ./model_voting/sub/other
cp -a ./result/robertawmmext_result_mean.csv ./model_voting/sub/other
cp -a ./result/robertalarge_result_mean.csv ./model_voting/sub/other

cd model_voting
python3 ./voting.py

echo "entity voting done."
