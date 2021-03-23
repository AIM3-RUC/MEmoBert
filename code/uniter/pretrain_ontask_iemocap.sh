export PYTHONPATH=/data7/MEmoBert

gpu_id=$1

## First use trn-data to pretrain on the task dataset
## Then use same trn-data train on downsteam task 

# iemocap maxbb=64, conf_th=0.0, 4500 / 128  * 10 = 400
# corpus_name='iemocap'
# for cvNo in `seq 1 10`;
# do
# CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#         --cvNo ${cvNo}  --config config/pretrain-task-${corpus_name}-base-2gpu_4tasks.json \
#         --melm_prob 0.5 --model_config config/uniter-base-emoword_nomultitask.json \
#         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2_uniter_4tasks_lr5e5_bs1024_faceth0.5/ckpt/model_step_5000.pt \
#         --learning_rate 2e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --conf_th 0.0 --max_bb 64 \
#         --train_batch_size 64 --val_batch_size 64 \
#         --num_train_steps 4000 --warmup_steps 0 --valid_steps 1000 \
#         --output_dir /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-nomask_movies_v1v2_uniter_4tasks_faceth0.5_5k-4tasks_maxbb64_faceth0.0_train4k/${cvNo}
# done

# corpus_name='iemocap'
# for cvNo in `seq 9 10`;
# do
# CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#         --cvNo ${cvNo}  --config config/pretrain-task-${corpus_name}-base-2gpu_4tasks_melm.json \
#         --melm_prob 0.5 --model_config config/uniter-base-emoword_multitask.json \
#         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2_uniter_4tasks_lr5e5_bs1024_faceth0.5_mlt-melm5/ckpt/model_step_5000.pt \
#         --learning_rate 2e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --conf_th 0.0 --max_bb 64 \
#         --train_batch_size 64 --val_batch_size 64 \
#         --num_train_steps 400 --warmup_steps 0 --valid_steps 100 \
#         --output_dir /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-nomask_movies_v1v2_uniter_4tasks_faceth0.5_mltmelm5_5k-4tasks_maxbb64_faceth0.0_mltmelm5/${cvNo}
# done

# corpus_name='msp'
# for cvNo in `seq 10 12`;
# do
# CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#         --cvNo ${cvNo}  --config config/pretrain-task-${corpus_name}-base-2gpu_4tasks.json \
#         --model_config config/uniter-base-emoword_nomultitask.json \
#         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2_uniter_4tasks_lr5e5_bs1024_faceth0.5/ckpt/model_step_5000.pt \
#         --learning_rate 2e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 2 \
#         --conf_th 0.0 --max_bb 36 \
#         --train_batch_size 128 --val_batch_size 128 \
#         --num_train_steps 3000 --warmup_steps 0 --valid_steps 500 \
#         --output_dir /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-nomask_movies_v1v2_uniter_4tasks_faceth0.5_5k-4tasks_maxbb36_faceth0.0_train3k/${cvNo}
# done

# corpus_name='msp'
# for cvNo in `seq 1 12`;
# do
# CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#         --cvNo ${cvNo}  --config config/pretrain-task-${corpus_name}-base-2gpu_4tasks_melm.json \
#         --melm_prob 0.5 --model_config config/uniter-base-emoword_multitask.json \
#         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2_uniter_4tasks_lr5e5_bs1024_faceth0.5_mlt-melm5/ckpt/model_step_5000.pt \
#         --learning_rate 2e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 2 \
#         --conf_th 0.0 --max_bb 36 \
#         --train_batch_size 128 --val_batch_size 128 \
#         --num_train_steps 400 --warmup_steps 0 --valid_steps 100 \
#         --output_dir /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-nomask_movies_v1v2_uniter_4tasks_faceth0.5_5k-4tasks_maxbb36_faceth0.0_mltmelm5/${cvNo}
# done

# 10000 / 128 * 10 = 800
# corpus_name='meld'
# for cvNo in `seq 1 1`;
# do
# CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#         --cvNo ${cvNo}  --config config/pretrain-task-${corpus_name}-base-2gpu_4tasks.json \
#         --model_config config/uniter-base-emoword_nomultitask.json \
#         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2_uniter_4tasks_lr5e5_bs1024_faceth0.5/ckpt/model_step_5000.pt \
#         --learning_rate 2e-05 --lr_sched_type 'linear'  --gradient_accumulation_steps 2 \
#         --conf_th 0.1 --max_bb 36 \
#         --train_batch_size 128 --val_batch_size 128 \
#         --num_train_steps 600 --warmup_steps 0 --valid_steps 100 \
#         --output_dir /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-nomask_movies_v1v2_uniter_4tasks_faceth0.5_5k-4tasks_maxbb36_faceth0.1/${cvNo}
# done

# corpus_name='meld'
# for cvNo in `seq 1 1`;
# do
# CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#         --cvNo ${cvNo}  --config config/pretrain-task-${corpus_name}-base-2gpu_4tasks_melm.json \
#         --melm_prob 0.5 --model_config config/uniter-base-emoword_multitask.json \
#         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2_uniter_4tasks_lr5e5_bs1024_faceth0.5_mlt-melm5/ckpt/model_step_5000.pt \
#         --learning_rate 2e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 2 \
#         --conf_th 0.1 --max_bb 36 \
#         --train_batch_size 128 --val_batch_size 128 \
#         --num_train_steps 800 --warmup_steps 0 --valid_steps 200 \
#         --output_dir /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-nomask_movies_v1v2_uniter_4tasks_faceth0.5_mltmelm5_5k-4tasks_maxbb36_faceth0.1_mltmelm5/${cvNo}
# done