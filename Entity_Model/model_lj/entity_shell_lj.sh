python3 ./bert_classify_train_with_pos_samples_add_feature_extend_trainSet-PreProcess-bert_base.py
python3 ./bert_classify_train_with_pos_samples_add_feature_extend_trainSet-PreProcess-bert_wwm.py
python3 ./bert_classify_train_with_pos_samples_add_feature_extend_trainSet-PreProcess-roberta-large.py
python3 ./bert_classify_train_with_pos_samples_add_feature_extend_trainSet-PreProcess-roberta-large-wwm-ext.py
python3 ./bert_classify_train_with_pos_samples_add_feature_extend_trainSet-PreProcess-roberta-wwm.py

cp -a ./submit/entity_bert_base.csv ./model_voting/single
cp -a ./submit/entity_bert_wwm.csv ./model_voting/sub/other
cp -a ./submit/entity_roberta_wwm.csv ./model_voting/sub/other

cd model_voting
python3 ./voting.py

echo "entity voting done."