export PYTHONPATH=/data7/MEmoBert

# for 180000 / 256 * 30 =20000, bs= 32 * 2 * 4 = 256
CUDA_VISIBLE_DEVICES=5,6 horovodrun -np 2 python pretrain.py \
        --cvNo 0 --use_speech --use_visual \
        --config config/pretrain-movies-v1v2v3-base-2gpu_4tasks.json \
        --model_config config/uniter-base-emoword_nomultitask.json \
        --learning_rate 5e-05 --gradient_accumulation_steps 4 \
        --learning_rate 5e-5 --lr_sched_type 'linear' \
        --conf_th 0.5 --max_txt_len 30 --max_bb 36 \
        --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
        --train_batch_size 64 --val_batch_size 64 \
        --num_train_steps 20000 --warmup_steps 2000 --valid_steps 500 \
        --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_4tasks_lr5e5_bs1024_faceth0.5