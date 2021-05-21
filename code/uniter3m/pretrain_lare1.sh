export PYTHONPATH=/data7/MEmoBert

### case10.1: text + wav2vec + visual, diff emo-lare task with withitm task running on a100
emolare_LStask_ratio=0.2
CUDA_VISIBLE_DEVICES=7 horovodrun -np 1 python pretrain.py \
        --cvNo 0 --n_workers 4 --use_speech --use_visual \
        --emolare_LStask_ratio ${emolare_LStask_ratio} \
        --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_emolare_debug.json \
        --model_config config/uniter-base-emoword_nomultitask_difftype_lare.json \
        --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
        --IMG_DIM 342 --Speech_DIM 768 \
        --conf_th 0.5 --max_txt_len 30 --max_bb 36 \
        --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
        --train_batch_size 20 --val_batch_size 20 \
        --num_train_steps 30000 --warmup_steps 3000 --valid_steps 10 \
        --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_emoclslare${emolare_LStask_ratio}_withitm_vstype2_lr5e5_bs1024

### case10.2: text + wav2vec + visual, diff emo-lare task with withitm task running on a100
# emolare_LStask_ratio=0.4
# CUDA_VISIBLE_DEVICES=6 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech --use_visual \
#         --emolare_LStask_ratio ${emolare_LStask_ratio} \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_emolare.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_lare.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --conf_th 0.5 --max_txt_len 30 --max_bb 36 \
#         --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#         --train_batch_size 20 --val_batch_size 20 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 30 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_emoclslare${emolare_LStask_ratio}_withitm_vstype2_lr5e5_bs1024


# ### case10.3: text + wav2vec + visual, diff emo-lare task with noitm task running on a100
# emolare_LStask_ratio=0.2
# CUDA_VISIBLE_DEVICES=7 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech --use_visual \
#         --emolare_LStask_ratio ${emolare_LStask_ratio} \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_4tasks_emolare.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_lare.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --conf_th 0.5 --max_txt_len 30 --max_bb 36 \
#         --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#         --train_batch_size 20 --val_batch_size 20 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 30 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_emoclslare${emolare_LStask_ratio}_noitm_vstype2_lr5e5_bs1024

# ### case10.4: text + wav2vec + visual, diff emo-lare task with noitm task running on a100
# emolare_LStask_ratio=0.4
# CUDA_VISIBLE_DEVICES=7 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech --use_visual \
#         --emolare_LStask_ratio ${emolare_LStask_ratio} \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_4tasks_emolare.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_lare.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --conf_th 0.5 --max_txt_len 30 --max_bb 36 \
#         --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#         --train_batch_size 20 --val_batch_size 20 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 30 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_emoclslare${emolare_LStask_ratio}_noitm_vstype2_lr5e5_bs1024