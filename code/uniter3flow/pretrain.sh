export PYTHONPATH=/data7/MEmoBert

# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python pretrain.py \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_rawimg_2optim_res_onlymlm.json \
#         --output_dir /data7/emobert/exp/pretrain/rawimg_nomask_movies_v1v2v3_uniter_onlymlm_2optim_5e5_1e3_initscratch_res \
#         --learning_rate 5e-05 --backbone_learning_rate 1e-03

# CUDA_VISIBLE_DEVICES=2,3 horovodrun -np 2 python pretrain.py \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_rawimg_2optim_res_onlymlm.json \
#         --output_dir /data7/emobert/exp/pretrain/rawimg_nomask_movies_v1v2v3_uniter_onlymlm_2optim_5e5_1e4_initscratch_res \
#         --learning_rate 5e-05 --backbone_learning_rate 1e-04

# CUDA_VISIBLE_DEVICES=4,5 horovodrun -np 2 python pretrain.py \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_rawimg_2optim_res_onlymlm.json \
#         --output_dir /data7/emobert/exp/pretrain/rawimg_nomask_movies_v1v2v3_uniter_onlymlm_2optim_1e5_1e3_initscratch_res \
#         --learning_rate 1e-05 --backbone_learning_rate 1e-03

CUDA_VISIBLE_DEVICES=6,7 horovodrun -np 2 python pretrain.py \
        --config config/pretrain-movies-v1v2v3-base-2gpu_rawimg_2optim_res_onlymlm.json \
        --output_dir /data7/emobert/exp/pretrain/rawimg_nomask_movies_v1v2v3_uniter_onlymlm_2optim_1e5_1e4_initfer_res \
        --learning_rate 1e-05 --backbone_learning_rate 1e-04