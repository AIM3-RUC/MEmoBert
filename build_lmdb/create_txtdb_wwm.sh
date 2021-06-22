# 根据人脸特征数据来构建文本数据，一个文本对应一段video.
export PYTHONPATH=/data7/MEmoBert
# 构建 whole word masking method

# # step1
# python mk_txtdb_by_faces_wwm.py --input /data7/emobert/data_nomask_new/movies_v3/ref_captions.json \
#                 --output /data7/emobert/txt_db/movies_v3_th0.5_wwm_all.db \
#                 --filter_path /data7/emobert/img_db_nomask/movies_v3/fc/nbb_th0.5_max36_min10.json \
#                 --toker bert-base-uncased  --dataset_name movies_v3_all

# ### step2 select validation set
# python mk_txtdb_by_faces_wwm.py --input /data7/emobert/data_nomask_new/movies_v3/ref_captions.json \
#                 --output /data7/emobert/txt_db/movies_v3_th0.5_wwm_all_val3k.db \
#                 --filter_path /data7/emobert/img_db_nomask/movies_v3/fc/nbb_th0.5_max36_min10.json \
#                 --toker bert-base-uncased  --dataset_name movies_v3_val3k \
#                 --num_samples 3000 

# ## step3
# # ## build the trn set by the all set and the all_3000 set. to build a new filter path
# python mk_txtdb_by_faces_wwm.py --input /data7/emobert/data_nomask_new/movies_v3/ref_captions.json \
#                 --output /data7/emobert/txt_db/movies_v3_th0.5_wwm_all_trn.db \
#                 --filter_path /data7/emobert/img_db_nomask/movies_v3/fc/nbb_th0.5_max36_min10.json \
#                 --filter_path_val /data7/emobert/txt_db/movies_v3_th0.5_wwm_all_val3k.db/img2txts.json \
#                 --toker bert-base-uncased  --dataset_name movies_v3_trn
                
# ##step4
# ### build the trn set by the all set and the all_2000 set. to build a new filter path
# python mk_txtdb_by_faces_wwm.py --input /data7/emobert/data_nomask_new/movies_v3/ref_captions.json \
#                 --output /data7/emobert/txt_db/movies_v3_th0.5_wwm_all_trn3k.db \
#                 --filter_path /data7/emobert/img_db_nomask/movies_v3/fc/nbb_th0.5_max36_min10.json \
#                 --include_path /data7/emobert/txt_db/movies_v3_th0.5_wwm_all_trn.db/img2txts.json \
#                 --toker bert-base-uncased  --dataset_name movies_v3_all_trn3k \
#                 --num_samples 3000

