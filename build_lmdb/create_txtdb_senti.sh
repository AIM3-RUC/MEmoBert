# 根据人脸特征数据来构建文本数据，一个文本对应一段video.
export PYTHONPATH=/data7/MEmoBert

### 根据人脸特征数据来构建文本数据，一个文本对应一段video. for movies data
# 构建不同的测试集和训练集，先构建测试集合，然后构建训练集，但是影响应该不大, 没有选模型的阶段，但是可以看一下val的结果
# step1
# python mk_txtdb_by_faces_sentiword.py --input /data7/emobert/data_nomask_new/movies_v2/ref_captions.json \
#                 --output /data7/emobert/txt_db/movies_v2_th0.5_emowords_sentiword_all.db \
#                 --filter_path /data7/emobert/img_db_nomask/movies_v2/fc/nbb_th0.5_max36_min10.json \
#                 --toker bert-base-uncased  --dataset_name movies_v2_all \
#                 --use_emo

# step2 select validation set
# python mk_txtdb_by_faces_sentiword.py --input /data7/emobert/data_nomask_new/movies_v2/ref_captions.json \
#                 --output /data7/emobert/txt_db/movies_v2_th0.5_emowords_sentiword_all_val3k.db \
#                 --filter_path /data7/emobert/img_db_nomask/movies_v2/fc/nbb_th0.5_max36_min10.json \
#                 --toker bert-base-uncased  --dataset_name movies_v2_val3k \
#                 --num_samples 3000 --use_emo 

# step3
## build the trn set by the all set and the all_3000 set. to build a new filter path
# python mk_txtdb_by_faces_sentiword.py --input /data7/emobert/data_nomask_new/movies_v2/ref_captions.json \
#                 --output /data7/emobert/txt_db/movies_v2_th0.5_emowords_sentiword_all_trn.db \
#                 --filter_path /data7/emobert/img_db_nomask/movies_v2/fc/nbb_th0.5_max36_min10.json \
#                 --filter_path_val /data7/emobert/txt_db/movies_v2_th0.5_emowords_sentiword_all_val3k.db/img2txts.json \
#                 --toker bert-base-uncased  --dataset_name movies_v2_trn \
#                 --use_emo 

# step4
## build the trn set by the all set and the all_2000 set. to build a new filter path
# python mk_txtdb_by_faces_sentiword.py --input /data7/emobert/data_nomask_new/movies_v2/ref_captions.json \
#                 --output /data7/emobert/txt_db/movies_v2_th0.5_emowords_sentiword_all_trn3k.db \
#                 --filter_path /data7/emobert/img_db_nomask/movies_v2/fc/nbb_th0.5_max36_min10.json \
#                 --include_path /data7/emobert/txt_db/movies_v2_th0.5_emowords_sentiword_all_trn.db/img2txts.json \
#                 --toker bert-base-uncased  --dataset_name movies_v2_all_trn3k \
#                 --num_samples 3000 --use_emo 

# for iemocap
# for i in `seq 1 10`; do
#   for setname in val tst trn; do
#     python mk_txtdb_by_names_sentiword.py --input /data7/emobert/exp/evaluation/IEMOCAP/refs/${i}/${setname}_ref.json \
#                     --output /data7/emobert/exp/evaluation/IEMOCAP/txt_db/${i}/${setname}_emowords_sentiword.db \
#                     --toker bert-base-uncased  --dataset_name iemocap_${setname} \
#                     --use_emo 
#   done
# done

### for msp
for i in `seq 1 12`; do
  for setname in val tst trn; do
    python mk_txtdb_by_names_sentiword.py --input /data7/emobert/exp/evaluation/MSP/refs/${i}/${setname}_ref.json \
                    --output /data7/emobert/exp/evaluation/MSP/txt_db/${i}/${setname}_emowords_sentiword.db \
                    --toker bert-base-uncased  --dataset_name msp_${setname} \
                     --use_emo 
  done
done

### for meld, 注意ref的id 和 img的id不一样，需要手动修改一下
# for setname in val test train; do
#     python mk_txtdb_by_names_sentiword.py --input /data7/emobert/exp/evaluation/MELD/refs/${setname}.json \
#                 --output /data7/emobert/exp/evaluation/MELD/txt_db/1/${setname}_emowords_sentiword_hq.db \
#                 --remove_low_quality_path /data7/MEmoBert/emobert/exp/evaluation/MELD/txt_db/${setname}_low_quality_less2.json \
#                 --toker bert-base-uncased  --dataset_name meld_${setname} \
#                 --use_emo 
# done
for setname in val test train; do
    python mk_txtdb_by_names_sentiword.py --input /data7/emobert/exp/evaluation/MELD/refs/${setname}.json \
                --output /data7/emobert/exp/evaluation/MELD/txt_db/1/${setname}_emowords_sentiword.db \
                --toker bert-base-uncased  --dataset_name meld_${setname} \
                --use_emo 
done