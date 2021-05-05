export PYTHONPATH=/data7/MEmoBert

# for 180000 / 256 * 30 =20000, bs= 32 * 2 * 4 = 256
# CUDA_VISIBLE_DEVICES=5,6 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --use_speech --use_visual \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_4tasks.json \
#         --model_config config/uniter-base-emoword_nomultitask.json \
#         --learning_rate 5e-05 --gradient_accumulation_steps 4 \
#         --learning_rate 5e-5 --lr_sched_type 'linear' \
#         --conf_th 0.5 --max_txt_len 30 --max_bb 36 \
#         --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#         --train_batch_size 64 --val_batch_size 64 \
#         --num_train_steps 20000 --warmup_steps 2000 --valid_steps 500 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_4tasks_lr5e5_bs1024_faceth0.5

# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech  \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechtext_2tasks_mlmitm.json \
#         --model_config config/uniter-base-emoword_nomultitask.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --conf_th 0.5 --max_txt_len 30 --max_bb 36 \
#         --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#         --train_batch_size 64 --val_batch_size 64 \
#         --num_train_steps 20000 --warmup_steps 2000 --valid_steps 2000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speech-berttype_text_2tasks_mlmitm_lr5e5_bs1024_faceth0.5

# CUDA_VISIBLE_DEVICES=2,3,4,5 horovodrun -np 4 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech  \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_text_2tasks_mlmitm.json \
#         --model_config config/uniter-base-emoword_nomultitask.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --conf_th 0.5 --max_txt_len 30 --max_bb 36 \
#         --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#         --train_batch_size 64 --val_batch_size 64 \
#         --num_train_steps 10000 --warmup_steps 1000 --valid_steps 1000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speech_wav2vec-berttype_text_2tasks_mlmitm_lr5e5_bs1024_faceth0.5

# CUDA_VISIBLE_DEVICES=4,5 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech --use_visual \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks.json \
#         --model_config config/uniter-base-emoword_multitask.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 8 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --conf_th 0.5 --max_txt_len 30 --max_bb 36 \
#         --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#         --train_batch_size 64 --val_batch_size 64 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 4000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_5tasks_vstype1_lr5e5_bs512_faceth0.5_debug

# CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech  \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vecasr_text_3tasks_mlmitmmsm.json \
#         --model_config config/uniter-base-emoword_nomultitask.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --conf_th 0.5 --max_txt_len 30 --max_bb 36 \
#         --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#         --train_batch_size 64 --val_batch_size 64 \
#         --num_train_steps 10000 --warmup_steps 1000 --valid_steps 1000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speech_wav2vecasr-berttype_text_3tasks_mlmitmmsrfr_lr5e5_bs1024_faceth0.5

### 在当前代码中，text+visual是正常.
CUDA_VISIBLE_DEVICES=2,6 horovodrun -np 2 python pretrain.py \
        --cvNo 0 --n_workers 4 --use_visual  \
        --config config/pretrain-movies-v1v2v3-base-2gpu_4tasks.json \
        --model_config config/uniter-base-emoword_nomultitask.json \
        --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
        --IMG_DIM 342 --Speech_DIM 768 \
        --conf_th 0.5 --max_txt_len 30 --max_bb 36 \
        --train_batch_size 64 --val_batch_size 64 \
        --num_train_steps 10000 --warmup_steps 1000 --valid_steps 1000 \
        --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_4tasks_lr5e5_bs1024_faceth0.5_test2

# 环境的问题？？ 换到emobert里面跑一下试试？？ 但是uniter的代码在baidu40上没问题啊 --一定还有bug, 再仔细排查一遍

### based on three modality to train two modality
# CUDA_VISIBLE_DEVICES=2,6 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech  \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_text_3tasks_mlmitmmsm.json \
#         --model_config config/uniter-base-emoword_nomultitask.json \
#         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_5tasks_vstype1_lr5e5_bs1024_faceth0.5/ckpt/model_step_20000.pt \
#         --learning_rate 2e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --conf_th 0.5 --max_txt_len 30 --max_bb 36 \
#         --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#         --train_batch_size 64 --val_batch_size 64 \
#         --num_train_steps 10000 --warmup_steps 0 --valid_steps 1000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask-basedon_nomask_movies_v1v2v3_uniter3m_5tasks-text_wav2vec_3tasks_lr2e5_bs1024_faceth0.5


