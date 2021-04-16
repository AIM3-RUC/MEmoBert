export PYTHONPATH=/data7/MEmoBert

gpu_id=$1
corpus_name='msp'

for cvNo in `seq 6 12`;
do
CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
        --cvNo ${cvNo}  --config config/pretrain-task-${corpus_name}-base-2gpu_2tasks.json \
        --model_config config/uniter-base-backbone_3dresnet.json \
        --checkpoint /data7/emobert/exp/pretrain/resnet3d_nomask_movies_v1v2v3_uniter_mlmitm_lr5e5-backbone_scratch_optimFalse-bs480_faceth0.5_continue1w/ckpt/model_step_10000.pt \
        --learning_rate 2e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
        --conf_th 0.0 --max_bb 36 --image_data_augmentation \
        --train_batch_size 32 --val_batch_size 32 \
        --num_train_steps 1500 --warmup_steps 0 --valid_steps 500 \
        --output_dir /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-resnet3d_nomask_movies_v1v2v3_uniter_lr5e5-backbone_scratch_optimFalse_continue1w-2tasks_maxbb36_faceth0.0_trnval/${cvNo}
done

# for cvNo in `seq 1 12`;
# do
# CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#         --cvNo ${cvNo}  --config config/pretrain-task-${corpus_name}-base-2gpu_4tasks_melm.json \
#         --melm_prob 0.5 --model_config config/uniter-base-emoword_multitask.json \
#         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_4tasks_lr5e5_bs1024_faceth0.5_mlt-melm5/ckpt/model_step_6000.pt \
#         --learning_rate 2e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --conf_th 0.0 --max_bb 36 \
#         --train_batch_size 64 --val_batch_size 64 \
#         --num_train_steps 400 --warmup_steps 0 --valid_steps 100 \
#         --output_dir /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-nomask_movies_v1v2v3_uniter_4tasks_faceth0.5_mltmelm5_5k-4tasks_maxbb36_faceth0.0_mltmelm5/${cvNo}
# done