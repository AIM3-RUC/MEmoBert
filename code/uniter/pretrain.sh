export PYTHONPATH=/data7/MEmoBert

# CUDA_VISIBLE_DEVICES=4,5 horovodrun -np 2 python pretrain.py \
#         --config config/pretrain-movies-v1v2-base-2gpu_4tasks.json

# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python pretrain.py \
#         --config config/pretrain-movies-v1-base-2gpu_4tasks.json

# CUDA_VISIBLE_DEVICES=5 horovodrun -np 1 python pretrain.py \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_4tasks.json \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_4tasks_faceth0.1 \
#         --checkpoint /data7/MEmoBert/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_4tasks_faceth0.1/ckpt/model_step_50000.pt \
#         --checkpoint_step 50000

CUDA_VISIBLE_DEVICES=6 horovodrun -np 1 python pretrain.py \
        --config config/pretrain-movies-v1v2v3-base-2gpu_4tasks.json \
        --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_4tasks_faceth0.1_reinitlr \
        --checkpoint /data7/MEmoBert/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_4tasks_faceth0.1/ckpt/model_step_50000.pt \
        --checkpoint_step 50000 --is_reinit_lr --warmup_steps 0

# CUDA_VISIBLE_DEVICES=4,5 horovodrun -np 2 python pretrain.py \
#         --config config/pretrain-movies-v1-medium-2gpu_v1v2_trans1.json \
#         --learning_rate 1e-05 --weight_decay 0.01 --max_txt_len 50 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1_uniter_medium_3tasks_v1v2_trans1_lr1e5_max50