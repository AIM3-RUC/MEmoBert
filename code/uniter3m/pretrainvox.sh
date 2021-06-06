export PYTHONPATH=/data7/MEmoBert

### for voxceleb2

## case1: visual + text running on gpu0, only voxceleb2
# CUDA_VISIBLE_DEVICES=2 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_visual  \
#         --config config/pretrain-vox2-v1-base-2gpu_4tasks_emo_sentiword.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --max_txt_len 60 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 200 --val_batch_size 200 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 4000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_vox2_v1_uniter3m_visual_text_4tasks_lr5e5_bs800_faceth0.5

## case2: visual + text running on gpu0,  moviesv1v2v3 + voxceleb2
# CUDA_VISIBLE_DEVICES=3,4 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_visual  \
#         --config config/pretrain-movies-v1v2v3-vox2-v1-base-2gpu_4tasks_emo_sentiword.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --max_txt_len 60 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 200 --val_batch_size 200 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 4000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_moviesv1v2v3_vox2_v1_uniter3m_visual_text_4tasks_lr5e5_bs800_faceth0.5

## case3: visual + text + emocls running on gpu0, moviesv1v2v3 + voxceleb2 
# CUDA_VISIBLE_DEVICES=5,6 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_visual  \
#         --config config/pretrain-movies-v1v2v3-vox2-v1-base-2gpu_4tasks_emo_sentiword_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --max_txt_len 60 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 200 --val_batch_size 200 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 4000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_moviesv1v2v3_vox2_v1_uniter3m_visual_text_4tasks_emoclsSoft_lr5e5_bs800_faceth0.5


## case4: visual + text + emocls running on gpu0, moviesv1v2v3 + voxceleb2 + projs
# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_visual  \
#         --config config/pretrain-movies-v1v2v3-vox2-v1-base-2gpu_4tasks_emo_sentiword_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_pojs_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --max_txt_len 60 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 200 --val_batch_size 200 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 4000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_moviesv1v2v3_vox2_v1_uniter3m_visual_text_4tasks_emoclsSoft_avprojs_lr5e5_bs800_faceth0.5

## case5: visual + text + speech running on gpu0, + voxceleb2
# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_visual --use_speech  \
#         --config config/pretrain-movies-v1v2v3-vox2-v1-base-2gpu_speechwav2vec_5tasks_emo_sentiword.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 8 \
#         --max_txt_len 60 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 32 --val_batch_size 32 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies-v1v2v3-vox2-v1-base-uniter3m_visual_wav2vec_text_5tasks_vstype2_lr5e5_bs512_faceth0.5

## case6: visual + text + speech running on gpu0, + voxceleb2
# CUDA_VISIBLE_DEVICES=3,5 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_visual --use_speech  \
#         --config config/pretrain-movies-v1v2v3-vox2-v1-base-2gpu_speechwav2vec_5tasks_emo_sentiword_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --max_txt_len 60 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 32 --val_batch_size 32 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies-v1v2v3-vox2-v1-base-uniter3m_visual_wav2vec_text_5tasks_emocls_vstype2_lr5e5_bs512_faceth0.5

## case6: visual + text + speech running on gpu0, + voxceleb2_v1v2
# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_visual --use_speech  \
#         --config config/pretrain-movies-v1v2v3-vox2-v1v2-base-2gpu_speechwav2vec_5tasks_emo_sentiword.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --max_txt_len 50 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 32 --val_batch_size 32 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies-v1v2v3-vox2-v1v2-base-uniter3m_visual_wav2vec_text_5tasks_vstype2_lr5e5_bs512_faceth0.5

## case6: visual + text + speech running on gpu0, + voxceleb2_v1v2 + + emocls
# CUDA_VISIBLE_DEVICES=4,5 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_visual --use_speech  \
#         --config config/pretrain-movies-v1v2v3-vox2-v1v2-base-2gpu_speechwav2vec_5tasks_emo_sentiword_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --max_txt_len 50 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 32 --val_batch_size 32 \
#         --num_train_steps 50000 --warmup_steps 5000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies-v1v2v3-vox2-v1v2-base-uniter3m_visual_wav2vec_text_5tasks_emocls_vstype2_lr5e5_bs512

## case6: visual + text running on gpu0, + voxceleb2_v1v2
# CUDA_VISIBLE_DEVICES=2,3 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_visual  \
#         --config config/pretrain-movies-v1v2v3-vox2-v1v2-base-2gpu_speechwav2vec_4tasks_emo_sentiword.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --max_txt_len 50 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 64 --val_batch_size 64 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies-v1v2v3-vox2-v1v2-base-uniter3m_visual_text_4tasks_vstype2_lr5e5_bs512

## case6: visual + text running on gpu0,  movies_v1v2v3 + voxceleb2_v1v2 + emocls
# CUDA_VISIBLE_DEVICES=2,3 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_visual  \
#         --config config/pretrain-movies-v1v2v3-vox2-v1v2-base-2gpu_speechwav2vec_4tasks_emo_sentiword_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --max_txt_len 50 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 64 --val_batch_size 64 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies-v1v2v3-vox2-v1v2-base-uniter3m_visual_text_4tasks_emocls_vstype2_lr5e5_bs512

### 晚上跑起来 -- AL 的两组 --running
## case6: text + speech running on gpu0, + voxceleb2_v1
# CUDA_VISIBLE_DEVICES=2 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech  \
#         --config config/pretrain-movies-v1v2v3-vox2-v1-base-2gpu_wav2vec_text_3tasks_emo_sentiword.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --max_txt_len 50 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 80 --val_batch_size 80 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies-v1v2v3-vox2-v1-base-uniter3m_wav2vec_text_3tasks_emo_sentiword_vstype2_lr5e5_bs512_faceth0.5

### 晚上跑起来 -- AL 的两组 --running
## case6: text + speech running on gpu0, + voxceleb2 + emocls.
# CUDA_VISIBLE_DEVICES=3 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech  \
#         --config config/pretrain-movies-v1v2v3-vox2-v1-base-2gpu_wav2vec_text_3tasks_emo_sentiword_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --max_txt_len 50 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 64 --val_batch_size 64 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies-v1v2v3-vox2-v1-base-uniter3m_wav2vec_text_3tasks_emo_sentiword_emocls_vstype2_lr5e5_bs512_faceth0.5

## case6: visual + text + speech running on gpu0, + voxceleb2, 由于 voxceleb2 label不准，所以在 voxceleb2 数据上去除 emocls.
# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_visual --use_speech  \
#         --config config/pretrain-movies-v1v2v3-vox2-v1-base-2gpu_speechwav2vec_5tasks_emo_sentiword_emocls_voxno.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --max_txt_len 60 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 32 --val_batch_size 32 \
#         --num_train_steps 40000 --warmup_steps 3000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies-v1v2v3-vox2-v1-base-uniter3m_visual_wav2vec_text_5tasks_emocls_voxno_vstype2_lr5e5_bs512_faceth0.5