export PYTHONPATH=/data7/MEmoBert

# python mk_txtdb_onlytext.py --input /data7/emobert/text_emo_corpus/all_5corpus/emo5_bert_data/train.csv \
#                 --dataset_name  all_5corpus_emo5_bert_data \
#                 --output /data7/emobert/txt_db/onlytext_all_5corpus_emo5_bert_data_emolabel.db \
#                 --toker bert-base-uncased --use_emo_label --use_emo_dim 5

# python mk_txtdb_onlytext.py --input /data7/emobert/text_emo_corpus/all_5corpus/emo5_bert_data/val.csv \
#                 --dataset_name  all_5corpus_emo5_bert_data_val \
#                 --output /data7/emobert/txt_db/onlytext_all_5corpus_emo5_bert_data_emolabel_val.db \
#                 --toker bert-base-uncased --use_emo_label --use_emo_dim 5

# 构建opensub的文本数据集，用于获取EmoCls伪标注
# python mk_txtdb_onlytext.py --input /data7/emobert/exp/mlm_pretrain/datasets/OpenSubtitlesV2018/opensub1000w_p4.csv\
#                 --dataset_name  opensub_p4_emo5 \
#                 --output /data7/emobert/txt_db/onlytext_opensub_p4_emo5_bert_data.db \
#                 --toker bert-base-uncased

## 随机挑选 100w
# python mk_txtdb_onlytext_wwm.py --input /data7/emobert/exp/mlm_pretrain/datasets/OpenSubtitlesV2018/opensub1000w_p4.csv\
#                 --dataset_name  opensub_p4_wwm --num_samples 1000000 \
#                 --output /data7/emobert/txt_db/onlytext_opensub_p4_wwm_random_100w.db \
#                 --toker bert-base-uncased

## 只挑选情感词的句子作为语料, 100w
# python mk_txtdb_onlytext_wwm.py --input /data7/emobert/exp/mlm_pretrain/datasets/OpenSubtitlesV2018/opensub1000w_p4.csv\
#                 --dataset_name  opensub_p4_wwm_selectemowords --num_samples 1000000 --select_emowords \
#                 --output /data7/emobert/txt_db/onlytext_opensub_p4_wwm_selectemowords.db \
#                 --toker bert-base-uncased

## 根据文本分类模型选情感显著的100w
# based on the 

## 根据下游任务分类模型选情感显著的100w