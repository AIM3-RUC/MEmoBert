export PYTHONPATH=/data7/MEmoBert

gpu_id=$1

## First use trn-data to pretrain on the task dataset
## Then use same trn-data train on downsteam task, train_emo_meld.sh

### 10000 / 128 * 10 = 1000
corpus_name='meld'
for cvNo in `seq 1 1`;
do
        for conth in 0.5;
        do
        CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
                --cvNo ${cvNo} --use_speech --use_visual \
                --config config/pretrain-task-${corpus_name}_norm-base-2gpu_4tasks.json \
                --model_config config/uniter-base-emoword_nomultitask.json \
                --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_4tasks_lr5e5_bs1024_faceth0.5/ckpt/model_step_20000.pt \
                --learning_rate 2e-05 --lr_sched_type 'linear'  --gradient_accumulation_steps 4 \
                --conf_th ${conth} --max_bb 36 \
                --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
                --train_batch_size 64 --val_batch_size 64 \
                --num_train_steps 2500 --warmup_steps 200 --valid_steps 500 \
                --output_dir /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-nomask_movies_v1v2v3_uniter3m_4tasks_lr5e5_bs1024-4tasks_maxbb36_faceth${conth}_trnval/${cvNo}
        done
done