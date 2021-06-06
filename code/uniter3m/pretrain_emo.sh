export PYTHONPATH=/data7/MEmoBert

# 在第一阶段预训练以general pretrain为主, 第二阶段采用情感相关的预训练任务, 比如 melm, merm, eitm, emolare.
# 由于没有语音相关的情感任务，所以暂时只考虑视觉和文本两个模态进行验证，对几个不同的任务做 做一些 abaltion study.

### case1: Only Emocls, based on nomask_moviesv1v2v3_vox2_v1_uniter3m_visual_text_4tasks_lr5e5_bs800_faceth0.5 
# CUDA_VISIBLE_DEVICES=1 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual \
#         --config config/pretrain-movies-v1v2v3-vox2-v1-base-2gpu_speechwav2vec_emotasks_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --checkpoint /data7/emobert/exp/pretrain/nomask_moviesv1v2v3_vox2_v1_uniter3m_visual_text_4tasks_lr5e5_bs800_faceth0.5/ckpt/model_step_40000.pt \
#         --learning_rate 3e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 --max_txt_len 60 \
#         --train_batch_size 100 --val_batch_size 100 \
#         --num_train_steps 30000 --warmup_steps 2000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_4tasks_emotasks_emocls_lr3e5

# ### case2: Emocls + MELM-nomultitask, based on nomask_moviesv1v2v3_vox2_v1_uniter3m_visual_text_4tasks_lr5e5_bs800_faceth0.5
# CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual \
#         --config config/pretrain-movies-v1v2v3-vox2-v1-base-2gpu_speechwav2vec_emotasks_melm_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --checkpoint /data7/emobert/exp/pretrain/nomask_moviesv1v2v3_vox2_v1_uniter3m_visual_text_4tasks_lr5e5_bs800_faceth0.5/ckpt/model_step_40000.pt \
#         --learning_rate 3e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 --max_txt_len 60 \
#         --train_batch_size 100 --val_batch_size 100 \
#         --num_train_steps 30000 --warmup_steps 1000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_4tasks_emotasks_melm_emocls_lr3e5

# ### case3: Only Emocls + MELM-nomultitask + MERM, based on nomask_moviesv1v2v3_vox2_v1_uniter3m_visual_text_4tasks_lr5e5_bs800_faceth0.5
# CUDA_VISIBLE_DEVICES=2 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual \
#         --config config/pretrain-movies-v1v2v3-vox2-v1-base-2gpu_speechwav2vec_emotasks_melm_merm_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --checkpoint /data7/emobert/exp/pretrain/nomask_moviesv1v2v3_vox2_v1_uniter3m_visual_text_4tasks_lr5e5_bs800_faceth0.5/ckpt/model_step_40000.pt \
#         --learning_rate 3e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 --max_txt_len 60 \
#         --train_batch_size 80 --val_batch_size 80 \
#         --num_train_steps 30000 --warmup_steps 1000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_4tasks_emotasks_melm_merm_emocls_lr3e5

# ### case4: Only Emocls + MERM, based on nomask_moviesv1v2v3_vox2_v1_uniter3m_visual_text_4tasks_lr5e5_bs800_faceth0.5
# CUDA_VISIBLE_DEVICES=3 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual \
#         --config config/pretrain-movies-v1v2v3-vox2-v1-base-2gpu_speechwav2vec_emotasks_merm_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --checkpoint /data7/emobert/exp/pretrain/nomask_moviesv1v2v3_vox2_v1_uniter3m_visual_text_4tasks_lr5e5_bs800_faceth0.5/ckpt/model_step_40000.pt \
#         --learning_rate 3e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 --max_txt_len 60 \
#         --train_batch_size 100 --val_batch_size 100 \
#         --num_train_steps 30000 --warmup_steps 1000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_4tasks_emotasks_merm_emocls_lr3e5

# ### case5: Only Emocls + Eitm, based on nomask_moviesv1v2v3_vox2_v1_uniter3m_visual_text_4tasks_lr5e5_bs800_faceth0.5
# CUDA_VISIBLE_DEVICES=4 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual \
#         --use_total_eitm \
#         --config config/pretrain-movies-v1v2v3-vox2-v1-base-2gpu_speechwav2vec_emotasks_eitm_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --checkpoint /data7/emobert/exp/pretrain/nomask_moviesv1v2v3_vox2_v1_uniter3m_visual_text_4tasks_lr5e5_bs800_faceth0.5/ckpt/model_step_40000.pt \
#         --learning_rate 3e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 --max_txt_len 60 \
#         --train_batch_size 100 --val_batch_size 100 \
#         --num_train_steps 30000 --warmup_steps 1000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_4tasks_emotasks_eitm_emocls_lr3e5

# ### case6: Only Emocls + Eitm + MERM + MELM, based on nomask_moviesv1v2v3_vox2_v1_uniter3m_visual_text_4tasks_lr5e5_bs800_faceth0.5
# CUDA_VISIBLE_DEVICES=1 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual \
#         --use_total_eitm \
#         --config config/pretrain-movies-v1v2v3-vox2-v1-base-2gpu_speechwav2vec_emotasks_melm_merm_eitm_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --checkpoint /data7/emobert/exp/pretrain/nomask_moviesv1v2v3_vox2_v1_uniter3m_visual_text_4tasks_lr5e5_bs800_faceth0.5/ckpt/model_step_40000.pt \
#         --learning_rate 3e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 --max_txt_len 60 \
#         --train_batch_size 200 --val_batch_size 200 \
#         --num_train_steps 40000 --warmup_steps 1000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_4tasks_emotasks_melm_merm_eitm_emocls_lr3e5

# ### case6: Only MERM + MELM, based on nomask_moviesv1v2v3_vox2_v1_uniter3m_visual_text_4tasks_lr5e5_bs800_faceth0.5
# CUDA_VISIBLE_DEVICES=7 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual \
#         --config config/pretrain-movies-v1v2v3-vox2-v1-base-2gpu_speechwav2vec_emotasks_melm_merm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --checkpoint /data7/emobert/exp/pretrain/nomask_moviesv1v2v3_vox2_v1_uniter3m_visual_text_4tasks_lr5e5_bs800_faceth0.5/ckpt/model_step_40000.pt \
#         --learning_rate 3e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 --max_txt_len 60 \
#         --train_batch_size 80 --val_batch_size 80 \
#         --num_train_steps 30000 --warmup_steps 1000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_3tasks_emotasks_melm_merm_lr3e5