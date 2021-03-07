export PYTHONPATH=/data7/MEmoBert

# CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 python pretrain.py \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_3tasks.json \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_3tasks_faceth0.1_new_5e5_wd.01 \
#         --learning_rate 5e-05 --weight_decay 0.01 

# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python pretrain.py \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_3tasks.json \
#         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_3tasks_faceth0.1_new_5e5_wd.01/ckpt/model_step_4000.pt \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_3tasks_faceth0.1_new_5e5_wd.01_continue_4k_fix5e5 \
#         --checkpoint_step 4000 --lr_sched_type fixed \
#         --learning_rate 5e-05 --weight_decay 0.01

CUDA_VISIBLE_DEVICES=2,3 horovodrun -np 2 python pretrain.py \
        --config config/pretrain-movies-v1v2v3-base-2gpu_3tasks.json \
        --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_3tasks_faceth0.1_new_5e5_wd.01/ckpt/model_step_16000.pt \
        --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_3tasks_faceth0.1_new_5e5_wd.01_continue_16k_fix1e5 \
        --checkpoint_step 16000 --lr_sched_type fixed \
        --learning_rate 1e-05 --weight_decay 0.01

# CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np 4 python pretrain.py \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_3tasks.json \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_3tasks_faceth0.1_new_1e4_wd.001 \
#         --learning_rate 1e-04 --weight_decay 0.001