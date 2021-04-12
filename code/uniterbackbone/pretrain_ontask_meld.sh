export PYTHONPATH=/data7/MEmoBert

gpu_id=$1
corpus_name='meld'

## First use trn-data to pretrain on the task dataset
## Then use same trn-data train on downsteam task, train_emo_meld.sh

### 10000 / 24 / 4 * 10 = 1500
for cvNo in `seq 1 1`;
do
        for conth in 0.5;
        do
        CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
                --cvNo ${cvNo}  --config config/pretrain-task-${corpus_name}-base-2gpu_2tasks.json \
                --model_config config/uniter-base-backbone_3dresnet.json \
                --checkpoint /data7/emobert/exp/pretrain/resnet3d_nomask_movies_v1v2v3_uniter_mlmitm_lr5e5-backbone_scratch_optimFalse-bs480_faceth0.5_continue1w/ckpt/model_step_10000.pt \
                --learning_rate 2e-5 --lr_sched_type 'linear'  --gradient_accumulation_steps 4 \
                --max_txt_len 30 --IMG_DIM 112 --image_data_augmentation \
                --conf_th 0.5 --max_bb 36  \
                --train_batch_size 24 --val_batch_size 24 \
                --num_train_steps 2000 --valid_steps 500 \
                --output_dir /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-resnet3d_nomask_movies_v1v2v3_uniter_lr5e5-backbone_scratch_optimFalse_continue1w-2tasks_maxbb36_faceth${conth}/${cvNo}
        done
done