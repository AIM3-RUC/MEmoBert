# 根据人脸特征数据来构建文本数据，一个文本对应一段video.
export PYTHONPATH=/data7/MEmoBert

### 根据人脸特征数据来构建文本数据，一个文本对应一段video. for movies data
# 构建不同的测试集和训练集，先构建测试集合，然后构建训练集，但是影响应该不大, 没有选模型的阶段，但是可以看一下val的结果
# step1
# python mk_txtdb_by_faces.py --input /data7/emobert/data_nomask/movies_v3/ref_captions.json \
#                 --output /data7/emobert/txt_db/movies_v3_th0.1_emowords_all.db \
#                 --filter_path /data7/emobert/img_db_nomask/movies_v3/nbb_th0.1_max64_min10.json \
#                 --toker bert-base-uncased  --dataset_name movies_v3 \
#                 --use_emo

# step2
python mk_txtdb_by_faces.py --input /data7/emobert/data_nomask/movies_v2/ref_captions.json \
                --output /data7/emobert/txt_db/movies_v2_th0.1_emowords_all_4000.db \
                --filter_path /data7/emobert/img_db_nomask/movies_v2/nbb_th0.1_max64_min10.json \
                --toker bert-base-uncased  --dataset_name movies_v2_val \
                --num_samples 4000 --use_emo

# step3
## build the trn set by the all set and the all_2000 set. to build a new filter path
python mk_txtdb_by_faces.py --input /data7/emobert/data_nomask/movies_v2/ref_captions.json \
                --output /data7/emobert/txt_db/movies_v2_th0.1_emowords_all_new_trn.db \
                --filter_path /data7/emobert/img_db_nomask/movies_v2/nbb_th0.1_max64_min10.json \
                --filter_path_val /data7/emobert/txt_db/movies_v2_th0.1_emowords_all_4000.db/img2txts.json \
                --toker bert-base-uncased  --dataset_name movies_v2_new_trn \
                --use_emo

# step4
## build the trn set by the all set and the all_2000 set. to build a new filter path
python mk_txtdb_by_faces.py --input /data7/emobert/data_nomask/movies_v2/ref_captions.json \
                --output /data7/emobert/txt_db/movies_v2_th0.1_emowords_all_new_trn_4000.db \
                --filter_path /data7/emobert/img_db_nomask/movies_v2/nbb_th0.1_max64_min10.json \
                --include_path /data7/emobert/txt_db/movies_v2_th0.1_emowords_all_new_trn.db/img2txts.json \
                --toker bert-base-uncased  --dataset_name movies_v2_all_new_trn_4000 \
                --num_samples 4000 --use_emo

# for iemocap
# for i in `seq 2 10`; do
#   for setname in val tst trn; do
#     python mk_txtdb_by_names.py --input /data7/emobert/exp/evaluation/IEMOCAP/refs/${i}/${setname}_ref.json \
#                     --output /data7/emobert/exp/evaluation/IEMOCAP/txt_db/${i}/${setname}_emo4.db \
#                     --toker bert-base-uncased  --dataset_name ${setname} \
#                     --use_emo --use_emo_type 'emo4'
#   done
# done

### for msp
# for i in `seq 1 12`; do
#   for setname in val tst trn; do
#     python mk_txtdb_by_names.py --input /data7/emobert/exp/evaluation/MSP-IMPROV/refs/${i}/${setname}_ref.json \
#                     --output /data7/emobert/exp/evaluation/MSP-IMPROV/txt_db/${i}/${setname}_emo4.db \
#                     --toker bert-base-uncased  --dataset_name ${setname} \
#                      --use_emo --use_emo_type 'emo4'
#   done
# done