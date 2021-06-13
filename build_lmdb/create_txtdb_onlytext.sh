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
python mk_txtdb_onlytext.py --input /data7/emobert/exp/mlm_pretrain/datasets/OpenSubtitlesV2018/opensub1000w_p4.csv\
                --dataset_name  opensub_p4_emo5 \
                --output /data7/emobert/txt_db/onlytext_opensub_p4_emo5_bert_data.db \
                --toker bert-base-uncased