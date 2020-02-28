#!/bin/bash

cd ./roberta_wwm_large_ext_emotion_xiong
python3 pystart.py
cd ../

python3 ./negative_classify_with_clearn_and_concat_entity_maskEntitys_extend_trainSet-bert_wwm.py
python3 ./negative_classify_with_clearn_and_concat_entity_maskEntitys-roberta-large.py

python3 ./voting.py

echo "emotion voting done."
