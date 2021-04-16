export PYTHONPATH=/data7/MEmoBert

# for 180000 / 1024 * 30 = 5200, bs= 128 * 2 * 4
# CUDA_VISIBLE_DEVICES=4,5 horovodrun -np 4 python pretrain.py \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_4tasks.json \
#         --model_config config/uniter-base-emoword_nomultitask.json \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_4tasks_lr5e5_bs1024_faceth0.5 \
#         --learning_rate 5e-05 --gradient_accumulation_steps 2 \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --conf_th 0.5 --max_bb 36 \
#         --train_batch_size 128 --val_batch_size 128 \
#         --num_train_steps 6000 --warmup_steps 600 --valid_steps 600

# CUDA_VISIBLE_DEVICES=6,7 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_4tasks_melm.json \
#         --melm_prob 0.5 --model_config config/uniter-base-emoword_multitask.json \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_4tasks_lr5e5_bs1024_faceth0.5_mlt-melm5 \
#         --learning_rate 5e-05 --lr_sched_type 'linear'  --gradient_accumulation_steps 4 \
#         --conf_th 0.5 --max_bb 36 \
#         --train_batch_size 128 --val_batch_size 128 \
#         --num_train_steps 6000 --warmup_steps 600 --valid_steps 600

# CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 python pretrain.py \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_4tasks.json \
#         --model_config config/uniter-base-emoword_nomultitask.json \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_4tasks_lr5e5_bs1024_faceth0.5 \
#         --learning_rate 5e-05 --gradient_accumulation_steps 2 \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --conf_th 0.5 --max_bb 36 \
#         --train_batch_size 128 --val_batch_size 128 \
#         --num_train_steps 6000 --warmup_steps 600 --valid_steps 600
