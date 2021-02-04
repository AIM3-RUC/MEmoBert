# 根据人脸特征数据来构建文本数据，一个文本对应一段video.
export PYTHONPATH=/data7/MEmoBert

### 根据人脸特征数据来构建文本数据，一个文本对应一段video. for movies data
python mk_txtdb_by_faces.py --input /data7/emobert/data_nomask/movies_v1/ref_captions.json \
                --output /data7/emobert/txt_db/movies_v1_th0.0_emowords_trn.db \
                --filter_path /data7/emobert/img_db_nomask/movies_v1/nbb_th0.0_max100_min10.json \
                --toker bert-base-uncased  --dataset_name movies_v1 \
                --use_emo --use_emo_type

# # for iemocap
# for i in `seq 2 10`; do
#   for setname in val tst trn; do
#     python mk_txtdb_by_names.py --input /data7/emobert/exp/evaluation/IEMOCAP/refs/${i}/${setname}_ref.json \
#                     --output /data7/emobert/exp/evaluation/IEMOCAP/txt_db/${i}/${setname}.db \
#                     --toker bert-base-uncased  --dataset_name ${setname}
#   done
# done

# for msp
# for i in `seq 1 12`; do
#   for setname in val tst trn; do
#     python mk_txtdb_by_names.py --input /data7/emobert/exp/evaluation/MSP-IMPROV/refs/${i}/${setname}_ref.json \
#                     --output /data7/emobert/exp/evaluation/MSP-IMPROV/txt_db/${i}/${setname}.db \
#                     --toker bert-base-uncased  --dataset_name ${setname}
#   done
# done