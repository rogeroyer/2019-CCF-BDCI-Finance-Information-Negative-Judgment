#!/bin/bash

cd bert_wwm_kfold
python3 ./pystart.py
cd ..

cd roberta_large_kfold
python3 ./pystart.py
cd ..

cd roberta_large_skfold
python3 ./pystart.py
cd ..

cd roberta_wwm_large_ext_entity_kfold
python3 ./pystart.py
cd ..

cd roberta_wwm_large_ext_entity_skfold
python3 ./pystart.py
cd ..

cp -a ./bert_wwm_kfold/result/submit.csv ./model_voting/single/bert_wwm_kfold.csv
cp -a ./roberta_large_kfold/result/submit.csv ./model_voting/sub/other/roberta_large_kfold.csv
cp -a ./roberta_large_skfold/result/submit.csv ./model_voting/sub/other/roberta_large_skfold.csv
cp -a ./roberta_wwm_large_ext_entity_kfold/result/submit.csv ./model_voting/sub/other/roberta_wwm_large_ext_entity_kfold.csv
cp -a ./roberta_wwm_large_ext_entity_skfold/result/submit.csv ./model_voting/sub/other/roberta_wwm_large_ext_entity_skfold.csv

cd model_voting
python3 ./voting.py

echo "entity voting done."
