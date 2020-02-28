#!/bin/bash

cp -a /home/ubuntu/transfer/Entity_Model/model_xiong/model_voting/three_model_voting_xiong.csv  ./sub/other/
cp -a /home/ubuntu/transfer/Entity_Model/model_ccy/model_voting/three_model_voting_ccy.csv  ./sub/other/
cp -a /home/ubuntu/transfer/Entity_Model/model_lj/model_voting/three_model_voting_lj.csv  ./sub/other/
cp -a /home/ubuntu/transfer/Entity_Model/model_zy/model_voting/three_model_voting_zy.csv  ./sub/other/
cp -a /home/ubuntu/transfer/MethodOne/five_models_voting_three_method.csv ./single/

python3 voting.py

echo "entity model voting done."
echo "file five_res_all_voting.csv is the final submit file."
