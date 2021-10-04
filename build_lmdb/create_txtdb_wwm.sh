# 根据人脸特征数据来构建文本数据，一个文本对应一段video.
export PYTHONPATH=/data7/MEmoBert
# 构建 whole word masking method

# # step1
# python mk_txtdb_by_faces_wwm.py --input /data7/emobert/data_nomask_new/movies_v2/ref_captions.json \
#                 --output /data7/emobert/txt_db/movies_v2_th0.5_wwm_all.db \
#                 --filter_path /data7/emobert/img_db_nomask/movies_v2/fc/nbb_th0.5_max36_min10.json \
#                 --toker bert-base-uncased  --dataset_name movies_v2_all

# ### step2 select validation set
# python mk_txtdb_by_faces_wwm.py --input /data7/emobert/data_nomask_new/movies_v2/ref_captions.json \
#                 --output /data7/emobert/txt_db/movies_v2_th0.5_wwm_all_val3k.db \
#                 --filter_path /data7/emobert/img_db_nomask/movies_v2/fc/nbb_th0.5_max36_min10.json \
#                 --toker bert-base-uncased  --dataset_name movies_v2_val3k \
#                 --num_samples 3000 

# ## step3
# # ## build the trn set by the all set and the all_3000 set. to build a new filter path
# python mk_txtdb_by_faces_wwm.py --input /data7/emobert/data_nomask_new/movies_v2/ref_captions.json \
#                 --output /data7/emobert/txt_db/movies_v2_th0.5_wwm_all_trn.db \
#                 --filter_path /data7/emobert/img_db_nomask/movies_v2/fc/nbb_th0.5_max36_min10.json \
#                 --filter_path_val /data7/emobert/txt_db/movies_v2_th0.5_wwm_all_val3k.db/img2txts.json \
#                 --toker bert-base-uncased  --dataset_name movies_v2_trn
                
# ##step4
# ### build the trn set by the all set and the all_2000 set. to build a new filter path
# python mk_txtdb_by_faces_wwm.py --input /data7/emobert/data_nomask_new/movies_v2/ref_captions.json \
#                 --output /data7/emobert/txt_db/movies_v2_th0.5_wwm_all_trn3k.db \
#                 --filter_path /data7/emobert/img_db_nomask/movies_v2/fc/nbb_th0.5_max36_min10.json \
#                 --include_path /data7/emobert/txt_db/movies_v2_th0.5_wwm_all_trn.db/img2txts.json \
#                 --toker bert-base-uncased  --dataset_name movies_v2_all_trn3k \
#                 --num_samples 3000


# # # step1-emowords
# python mk_txtdb_by_faces_wwm.py --input /data7/emobert/data_nomask_new/movies_v2/ref_captions.json \
#                 --output /data7/emobert/txt_db/movies_v2_th0.5_wwm_nrcemolex_all.db \
#                 --filter_path /data7/emobert/img_db_nomask/movies_v2/fc/nbb_th0.5_max36_min10.json \
#                 --toker bert-base-uncased  --dataset_name movies_v2_all \
#                 --use_emo

# # ### step2-emowords select validation set
# python mk_txtdb_by_faces_wwm.py --input /data7/emobert/data_nomask_new/movies_v2/ref_captions.json \
#                 --output /data7/emobert/txt_db/movies_v2_th0.5_wwm_nrcemolex_all_val3k.db \
#                 --filter_path /data7/emobert/img_db_nomask/movies_v2/fc/nbb_th0.5_max36_min10.json \
#                 --toker bert-base-uncased  --dataset_name movies_v2_val3k \
#                 --num_samples 3000  --use_emo

# # ## step3-emowords
# # ## build the trn set by the all set and the all_3000 set. to build a new filter path
# python mk_txtdb_by_faces_wwm.py --input /data7/emobert/data_nomask_new/movies_v2/ref_captions.json \
#                 --output /data7/emobert/txt_db/movies_v2_th0.5_wwm_nrcemolex_all_trn.db \
#                 --filter_path /data7/emobert/img_db_nomask/movies_v2/fc/nbb_th0.5_max36_min10.json \
#                 --filter_path_val /data7/emobert/txt_db/movies_v2_th0.5_wwm_nrcemolex_all_val3k.db/img2txts.json \
#                 --toker bert-base-uncased  --dataset_name movies_v2_trn  --use_emo
                
# # ##step4-emowords
# # ### build the trn set by the all set and the all_2000 set. to build a new filter path
# python mk_txtdb_by_faces_wwm.py --input /data7/emobert/data_nomask_new/movies_v2/ref_captions.json \
#                 --output /data7/emobert/txt_db/movies_v2_th0.5_wwm_nrcemolex_all_trn3k.db \
#                 --filter_path /data7/emobert/img_db_nomask/movies_v2/fc/nbb_th0.5_max36_min10.json \
#                 --include_path /data7/emobert/txt_db/movies_v2_th0.5_wwm_nrcemolex_all_trn.db/img2txts.json \
#                 --toker bert-base-uncased  --dataset_name movies_v2_all_trn3k \
#                 --num_samples 3000  --use_emo


# # step1
# python mk_txtdb_by_faces_wwm.py --input /data7/emobert/data_nomask_new/voxceleb2_v1/ref_captions.json \
#                 --output /data7/emobert/txt_db/voxceleb2_v1_th1.0_wwm_all.db \
#                 --filter_path /data7/emobert/img_db_nomask/voxceleb2_v1/fc/nbb_th1.0_max64_min10.json \
#                 --toker bert-base-uncased  --dataset_name voxceleb2_v1_all

# # # step2 select validation set
# python mk_txtdb_by_faces_wwm.py --input /data7/emobert/data_nomask_new/voxceleb2_v1/ref_captions.json \
#                 --output /data7/emobert/txt_db/voxceleb2_v1_th1.0_wwm_all_val3k.db \
#                 --filter_path /data7/emobert/img_db_nomask/voxceleb2_v1/fc/nbb_th1.0_max64_min10.json \
#                 --toker bert-base-uncased  --dataset_name voxceleb2_v1_val3k \
#                 --num_samples 3000  

# # # step3
# # ## build the trn set by the all set and the all_3000 set. to build a new filter path
# python mk_txtdb_by_faces_wwm.py --input /data7/emobert/data_nomask_new/voxceleb2_v1/ref_captions.json \
#                 --output /data7/emobert/txt_db/voxceleb2_v1_th1.0_wwm_all_trn.db \
#                 --filter_path /data7/emobert/img_db_nomask/voxceleb2_v1/fc/nbb_th1.0_max64_min10.json \
#                 --filter_path_val /data7/emobert/txt_db/voxceleb2_v1_th1.0_wwm_all_val3k.db/img2txts.json \
#                 --toker bert-base-uncased  --dataset_name voxceleb2_v1_trn

# # # step4
# # ## build the trn set by the all set and the all_3000 set. to build a new filter path
# python mk_txtdb_by_faces_wwm.py --input /data7/emobert/data_nomask_new/voxceleb2_v1/ref_captions.json \
#                 --output /data7/emobert/txt_db/voxceleb2_v1_th1.0_wwm_all_trn3k.db \
#                 --filter_path /data7/emobert/img_db_nomask/voxceleb2_v1/fc/nbb_th1.0_max64_min10.json \
#                 --include_path /data7/emobert/txt_db/voxceleb2_v1_th1.0_wwm_all_trn.db/img2txts.json \
#                 --toker bert-base-uncased  --dataset_name voxceleb2_v1_all_trn3k \
#                 --num_samples 3000 


# for iemocap
# for i in `seq 1 10`; do
#   for setname in val tst trn; do
#     python mk_txtdb_by_names_wwm.py --input /data7/emobert/exp/evaluation/IEMOCAP/refs/${i}/${setname}_ref.json \
#                     --output /data7/emobert/exp/evaluation/IEMOCAP/txt_db/${i}/${setname}_wwm_nrcemolex.db \
#                     --toker bert-base-uncased  --dataset_name iemocap_${setname}  --use_emo
#   done
# done

# # ### for msp
# for i in `seq 1 12`; do
#   for setname in val tst trn; do
#     python mk_txtdb_by_names_wwm.py --input /data7/emobert/exp/evaluation/MSP/refs/${i}/${setname}_ref.json \
#                     --output /data7/emobert/exp/evaluation/MSP/txt_db/${i}/${setname}_wwm_nrcemolex.db \
#                     --toker bert-base-uncased  --dataset_name msp_${setname} --use_emo
#   done
# done