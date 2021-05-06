export PYTHONPATH=/data7/MEmoBert

### 重新跑一下多组unite3m的实验，其中语音相关的实验要重新跑

## case1: visual + text running on gpu0
# CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual  \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_4tasks.json \
#         --model_config config/uniter-base-emoword_nomultitask.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --conf_th 0.5 --max_bb 36 \
#         --train_batch_size 128 --val_batch_size 128 \
#         --num_train_steps 20000 --warmup_steps 2000 --valid_steps 2000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_4tasks_lr5e5_bs512_faceth0.5

## case2: wav2vec + text running on gpu1
# CUDA_VISIBLE_DEVICES=1 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech  \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_text_3tasks_mlmitmmsm.json \
#         --model_config config/uniter-base-emoword_nomultitask.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --conf_th 0.5 --max_txt_len 30 --max_bb 36 \
#         --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#         --train_batch_size 128 --val_batch_size 128 \
#         --num_train_steps 20000 --warmup_steps 2000 --valid_steps 2000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_wav2vec_text_3tasks_mlmitmmsrfr_lr5e5_bs512_faceth0.5

## case3: wav2vec + text + visual running on gpu2
# CUDA_VISIBLE_DEVICES=2 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech --use_visual \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks.json \
#         --model_config config/uniter-base-emoword_nomultitask.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --conf_th 0.5 --max_txt_len 30 --max_bb 36 \
#         --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#         --train_batch_size 128 --val_batch_size 128 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 3000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_vstype1_lr5e5_bs512_faceth0.5

# case4: text + visual + sentiword-emo running on gpu3
# CUDA_VISIBLE_DEVICES=3 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_4tasks_emo_sentiword.json \
#         --model_config config/uniter-base-emoword_multitask.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --conf_th 0.5 --max_txt_len 30 --max_bb 36 \
#         --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#         --train_batch_size 128 --val_batch_size 128 \
#         --num_train_steps 20000 --warmup_steps 2000 --valid_steps 2000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_5tasks_vstype1_lr5e5_bs512_faceth0.5_sentiword

# case5: text + wav2vec + sentiword-emo running on gpu5
# CUDA_VISIBLE_DEVICES=5 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_wav2vec_3tasks_emo_sentiword.json \
#         --model_config config/uniter-base-emoword_multitask.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --conf_th 0.5 --max_txt_len 30 --max_bb 36 \
#         --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#         --train_batch_size 128 --val_batch_size 128 \
#         --num_train_steps 20000 --warmup_steps 2000 --valid_steps 2000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_wav2vec_text_4tasks_vstype1_lr5e5_bs512_faceth0.5_sentiword