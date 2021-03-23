# 根据人脸特征数据来构建文本数据，一个文本对应一段video.
export PYTHONPATH=/data7/MEmoBert

### 根据人脸特征数据来构建文本数据，一个文本对应一段video. for movies data
# 构建不同的测试集和训练集，先构建测试集合，然后构建训练集，但是影响应该不大, 没有选模型的阶段，但是可以看一下val的结果
# step1
python mk_txtdb_by_faces.py --input /data7/emobert/data_nomask_new/movies_v3/ref_captions.json \
                --output /data7/emobert/txt_db/movies_v3_th0.5_emowords_emotype_all.db \
                --filter_path /data7/emobert/img_db_nomask/movies_v3/fc/nbb_th0.5_max36_min10.json \
                --toker bert-base-uncased  --dataset_name movies_v3_all \
                --use_emo --use_emo_type emo4

# step2
python mk_txtdb_by_faces.py --input /data7/emobert/data_nomask_new/movies_v3/ref_captions.json \
                --output /data7/emobert/txt_db/movies_v3_th0.5_emowords_emotype_all_val5k.db \
                --filter_path /data7/emobert/img_db_nomask/movies_v3/fc/nbb_th0.5_max36_min10.json \
                --toker bert-base-uncased  --dataset_name movies_v3_val5k \
                --num_samples 5000 --use_emo --use_emo_type emo4

# step3
## build the trn set by the all set and the all_2000 set. to build a new filter path
python mk_txtdb_by_faces.py --input /data7/emobert/data_nomask_new/movies_v3/ref_captions.json \
                --output /data7/emobert/txt_db/movies_v3_th0.5_emowords_emotype_all_trn.db \
                --filter_path /data7/emobert/img_db_nomask/movies_v3/fc/nbb_th0.5_max36_min10.json \
                --filter_path_val /data7/emobert/txt_db/movies_v3_th0.5_emowords_emotype_all_val5k.db/img2txts.json \
                --toker bert-base-uncased  --dataset_name movies_v3_trn \
                --use_emo --use_emo_type emo4

# step4
## build the trn set by the all set and the all_2000 set. to build a new filter path
python mk_txtdb_by_faces.py --input /data7/emobert/data_nomask_new/movies_v3/ref_captions.json \
                --output /data7/emobert/txt_db/movies_v3_th0.5_emowords_emotype_all_trn5k.db \
                --filter_path /data7/emobert/img_db_nomask/movies_v3/fc/nbb_th0.5_max36_min10.json \
                --include_path /data7/emobert/txt_db/movies_v3_th0.5_emowords_emotype_all_trn.db/img2txts.json \
                --toker bert-base-uncased  --dataset_name movies_v3_all_trn5k \
                --num_samples 5000 --use_emo --use_emo_type emo4

# for iemocap
# for i in `seq 1 10`; do
#   for setname in val tst trn; do
#     python mk_txtdb_by_names.py --input /data7/emobert/exp/evaluation/IEMOCAP/refs/${i}/${setname}_ref.json \
#                     --output /data7/emobert/exp/evaluation/IEMOCAP/txt_db/${i}/${setname}_emowords_emotype.db \
#                     --toker bert-base-uncased  --dataset_name iemocap_${setname} \
#                     --use_emo --use_emo_type 'emo4'
#   done
# done

### for msp
# for i in `seq 1 12`; do
#   for setname in val tst trn; do
#     python mk_txtdb_by_names.py --input /data7/emobert/exp/evaluation/MSP/refs/${i}/${setname}_ref.json \
#                     --output /data7/emobert/exp/evaluation/MSP/txt_db/${i}/${setname}_emowords_emotype.db \
#                     --toker bert-base-uncased  --dataset_name msp_${setname} \
#                      --use_emo --use_emo_type 'emo4'
#   done
# done

### for meld, 注意ref的id 和 img的id不一样，需要手动修改一下 -- pending
# for setname in val test train; do
#     python mk_txtdb_by_names.py --input /data7/emobert/exp/evaluation/MELD/refs/${setname}.json \
#                 --output /data7/emobert/exp/evaluation/MELD/txt_db/${setname}_emowords_emotype.db \
#                 --toker bert-base-uncased  --dataset_name meld_${setname} \
#                 --use_emo --use_emo_type 'emo4'
# done