export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
corpus_name='meld'

for cvNo in $(seq 1 1)
do
        CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
                --cvNo ${cvNo} --use_visual \
                --config config/pretrain-task-${corpus_name}-base-2gpu_4tasks-emo.json \
                --model_config config/uniter-base-emoword_multitask.json \
                --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_4tasks_lr5e5_faceth0.5_melm.5/ckpt/model_step_20000.pt \
                --learning_rate 2e-05 --lr_sched_type 'linear'  --gradient_accumulation_steps 4 \
                --IMG_DIM 342 --Speech_DIM 768 \
                --conf_th 0.5 --max_bb 36 \
                --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
                --train_batch_size 64 --val_batch_size 64 \
                --num_train_steps 2500 --warmup_steps 250 --valid_steps 2500 \
                --output_dir /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_text_4tasks_lr5e5_melm.5-faceth0.0_trnval/${cvNo}
done