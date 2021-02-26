export PYTHONPATH=/data7/MEmoBert

# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python pretrain.py \
#         --config config/pretrain-movies-v1-base-2gpu_melmTM4.json

# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python pretrain.py \
#         --config config/pretrain-movies-v1-base-2gpu_faceth0.1.json

# CUDA_VISIBLE_DEVICES=2,3 horovodrun -np 2 python pretrain.py \
#         --config config/pretrain-movies-v1-base-2gpu_melm_faceth0.1.json

# CUDA_VISIBLE_DEVICES=4,5 horovodrun -np 2 python pretrain.py \
#         --config config/pretrain-movies-v1-base-2gpu_melm_faceth0.1_multitask.json

# CUDA_VISIBLE_DEVICES=2,3 horovodrun -np 2 python pretrain.py \
#         --config config/pretrain-movies-v1-base-2gpu_mlm_mrfr_melm.json

# CUDA_VISIBLE_DEVICES=5 horovodrun -np 1 python pretrain.py \
#         --config config/pretrain-movies-v1-base-2gpu_mlm_mrfr_mrckl_melm.json

# CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python pretrain.py \
#         --config config/pretrain-movies-v1-base-2gpu_mlm_itm_melm.json

# CUDA_VISIBLE_DEVICES=4,5 horovodrun -np 2 python pretrain.py \
#         --config config/pretrain-movies-v1-base-2gpu_mlm_mrckl.json

# CUDA_VISIBLE_DEVICES=3 horovodrun -np 1 python pretrain.py \
#         --config config/pretrain-movies-v1-base-2gpu.json

# CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python pretrain.py \
#         --config config/pretrain-movies-v1-small-2gpu_mlm.json \
#         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1_uniter_small_onlymlm_faceth0.1/ckpt/model_step_100000.pt

# CUDA_VISIBLE_DEVICES=1 horovodrun -np 1 python pretrain.py \
#         --config config/pretrain-movies-v1-small-2gpu_mlm.json 

# CUDA_VISIBLE_DEVICES=3,4 horovodrun -np 2 python pretrain.py \
#         --config config/pretrain-movies-v1-medium-2gpu_mlm_itm.json

# CUDA_VISIBLE_DEVICES=2,5 horovodrun -np 2 python pretrain.py \
#         --config config/pretrain-movies-v1-medium-2gpu_trans1.json

# CUDA_VISIBLE_DEVICES=6,7 horovodrun -np 2 python pretrain.py \
#         --config config/pretrain-movies-v1-medium-2gpu_trans2.json

# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python pretrain.py \
#         --config config/pretrain-movies-v1-medium-2gpu_mlm.json \
#         --learning_rate 1e-05 --weight_decay 0.01 --max_txt_len 50 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1_uniter_medium_onlymlm_lr5_decay0.01_max50

# CUDA_VISIBLE_DEVICES=2,3 horovodrun -np 2 python pretrain.py \
#         --config config/pretrain-movies-v1-medium-2gpu_mlm.json \
#         --learning_rate 1e-06 --weight_decay 0.01 --max_txt_len 50 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1_uniter_medium_onlymlm_lr6_decay0.01_max50

# CUDA_VISIBLE_DEVICES=4,5 horovodrun -np 2 python pretrain.py \
#         --config config/pretrain-movies-v1-medium-2gpu_mlm.json \
#         --learning_rate 1e-05 --weight_decay 0.1 --max_txt_len 50 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1_uniter_medium_onlymlm_lr5_decay0.1_max50

# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python pretrain.py \
#         --config config/pretrain-movies-v1-medium-2gpu_v1v2.json \
#         --learning_rate 5e-06 --weight_decay 0.01 --max_txt_len 50 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1_uniter_medium_3tasks_lr5e6_max50

CUDA_VISIBLE_DEVICES=4,5 horovodrun -np 2 python pretrain.py \
        --config config/pretrain-movies-v1-medium-2gpu_v1v2_trans1.json \
        --learning_rate 1e-05 --weight_decay 0.01 --max_txt_len 50 \
        --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1_uniter_medium_3tasks_v1v2_trans1_lr1e5_max50