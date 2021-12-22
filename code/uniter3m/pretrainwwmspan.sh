source deactivate base
export PYTHONPATH=/data7/MEmoBert

## 可以同时利用单模态，或者任意模态的组合进行训练，注意此时预训练任务不能 --use_visual 来进行判断，而是config中每个db模态信息是否存在.

## case1: visual + text running on wwm 
# CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_visual  \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_visual_text_4tasks_wwm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 200 --val_batch_size 200 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_4tasks_wwm_lr5e5_bs800

## case1.2: visual + text running on wwm + emocls - itm
# CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_visual  \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_visual_text_4tasks_wwm_emocls_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 200 --val_batch_size 200 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_4tasks_wwm_emocls_noitm_lr5e5_bs800

## case1.3: visual + text running on wwm + span + emocls - itm
# CUDA_VISIBLE_DEVICES=3 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_visual  \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_visual_text_4tasks_wwm_span_emocls_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 200 --val_batch_size 200 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_4tasks_wwm_span_emocls_noitm_lr5e5_bs800

# ## case2: speech + text running on wwm 
# CUDA_VISIBLE_DEVICES=1 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_speech  \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_wav2vec_text_3tasks_wwm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 200 --val_batch_size 200 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_wav2vec_text_3tasks_wwm_lr5e5_bs800

# ## case3: speech + visual + text running on wwm 
# CUDA_VISIBLE_DEVICES=2 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_lr5e5_bs800

# ## case3.1: speech + visual + text running on wwm - itm
# CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_noitm_lr5e5_bs800

# ## case4: speech + visual + text running on wwm + span
# CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_span.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_lr5e5_bs800

# # # ## case5: speech + visual + text running on wwm + span - itm -- Use this 
# CUDA_VISIBLE_DEVICES=3 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_span_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800

# # # ## case5: speech + visual + text running on wwm + span - itm -- no bert init
# CUDA_VISIBLE_DEVICES=4 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_span_noitm_noinit.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 10000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_noinit_lr5e5_bs800

# # # ## case5: speech + visual + text running on wwm + span - itm -- low-level trans1 is 216 speech global-cnn 
# CUDA_VISIBLE_DEVICES=5 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_visual_trans1_speechwav2vec_cnn_5tasks_wwm_span_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 216 --Speech_DIM 512 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 10000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_trans1_speechwav2vec_cnn_5tasks_wwm_span_noitm_lr5e5_bs800

# # ## case5.1: speech + visual + text running on wwm + span - itm + large-span=5 7 9
# CUDA_VISIBLE_DEVICES=1,5 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --mask_speech_consecutive 5 --mask_speech_len_ratio 0.5 --mask_visual_consecutive 5 --mask_visual_len_ratio 0.5 \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_span_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_spanS5.5V5.5_lr5e5_bs800

# # ## case5.2: speech + visual + text running on wwm + span - itm + large-span=5 7 9
# CUDA_VISIBLE_DEVICES=6,7 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --mask_speech_consecutive 7 --mask_speech_len_ratio 0.5 --mask_visual_consecutive 5 --mask_visual_len_ratio 0.5 \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_span_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_spanS7.5V5.5_lr5e5_bs800

# # ## case6: speech + visual + text running on wwm + span + emocls
# CUDA_VISIBLE_DEVICES=4 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_span_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_emocls_lr5e5_bs800

# # ## case7: speech + visual + text running on wwm + span + emocls
# CUDA_VISIBLE_DEVICES=5 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_span_emocls_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_emocls_noitm_lr5e5_bs800

# # ## case8.1: speech + affectnet-visual + text running on wwm + span + emocls
# CUDA_VISIBLE_DEVICES=6,7 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_affectnet_5tasks_wwm_span_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --max_txt_len 50  --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 25000 --warmup_steps 2500 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_affectnet_5tasks_wwm_span_noitm_lr5e5_bs800

# # ## case8.2: speech + affectnet-visual + text running on wwm + span + emocls
# CUDA_VISIBLE_DEVICES=5 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_visual  \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_affectnet_visual_text_4tasks_wwm_span_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --max_txt_len 50  --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 200 --val_batch_size 200 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_affectnet_visual_text_4tasks_wwm_span_noitm_lr5e5_bs800

# # # ## case9.1: speech + visual + text running on wwm + span - itm - visual span
# CUDA_VISIBLE_DEVICES=5 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_span_noitm_noaudio.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --max_txt_len 50 --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 5000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_noaudio_lr5e5_bs800

# # # ## case9.1: speech + visual + text running on wwm + span - itm - acoustic span
# CUDA_VISIBLE_DEVICES=6 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_span_noitm_novisual.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --max_txt_len 50 --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 5000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_novisual_lr5e5_bs800

#### case9: speech + visual + text running on noitm + wwm + mrm and msrm with large-ratio = 0.3 0.5 0.7
# CUDA_VISIBLE_DEVICES=3 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --mrm_prob 0.3 --msrm_prob 0.3  \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_mrm_msrm_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.3s.3_noitm_lr5e5_bs800

#### case9.2: speech + visual + text running on noitm + wwm + mrm and msrm with large-ratio = 0.3 0.5 0.7
# CUDA_VISIBLE_DEVICES=4 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --mrm_prob 0.5 --msrm_prob 0.5 \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_mrm_msrm_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.5s.5_noitm_lr5e5_bs800

#### case9.3: speech + visual + text running on noitm + wwm + mrm and msrm with large-ratio = 0.3 0.5 0.7
# CUDA_VISIBLE_DEVICES=5 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --mrm_prob 0.7 --msrm_prob 0.7 \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_mrm_msrm_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.7s.7_noitm_lr5e5_bs800

#### case10: speech + visual + text running on noitm + wwm + span + comparE
# CUDA_VISIBLE_DEVICES=3 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechcomparE_4tasks_wwm_span_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 130 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechcomparE_4tasks_wwm_span_noitm_lr5e5_bs800

## case11: speech + visual + text running on wwm + span - itm -- Use this 
# CUDA_VISIBLE_DEVICES=4 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --mixed_ratios \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_span_noitm_mix.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 60000 --warmup_steps 6000 --valid_steps 10000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_mix_ratios_noitm_lr5e5_bs800

## case11: speech + visual + text running on wwm + span - itm -- Use this 
# CUDA_VISIBLE_DEVICES=5 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --mask_speech_consecutive 5 --mask_speech_len_ratio 0.5 --mask_visual_consecutive 5 --mask_visual_len_ratio 0.5 \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_span_noitm_mix.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_mores5.5v5.5_noitm_lr5e5_bs800

#### case12: speech + visual + text running on noitm + wwm + mrm and msrm with whole mask
# CUDA_VISIBLE_DEVICES=1 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --mrm_prob 1.0 --msrm_prob 1.0 \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_mrm_msrm_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 60000 --warmup_steps 6000 --valid_steps 10000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.1s.1_noitm_lr5e5_bs800

#### case12.1: speech + visual + text running on noitm + wwm + mrm and msrm with whole mask
# CUDA_VISIBLE_DEVICES=4 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --mrm_prob 0.1 --msrm_prob 0.1 \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_mrm_msrm_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 10000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.01s.01_noitm_lr5e5_bs800

# CUDA_VISIBLE_DEVICES=5 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --mrm_prob 0.2 --msrm_prob 0.2 \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_mrm_msrm_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 10000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.2s.2_noitm_lr5e5_bs800

# CUDA_VISIBLE_DEVICES=6 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --mrm_prob 0.4 --msrm_prob 0.4 \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_mrm_msrm_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 10000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.4s.4_noitm_lr5e5_bs800

#### case12.1: speech + visual + text running on noitm + wwm + mrm and msrm with whole mask
# CUDA_VISIBLE_DEVICES=4 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --mrm_prob 0.6 --msrm_prob 0.6 \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_mrm_msrm_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 10000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.6s.6_noitm_lr5e5_bs800

#### case12.2: speech + visual + text running on noitm + wwm + mrm and msrm with whole mask
# CUDA_VISIBLE_DEVICES=5 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --mrm_prob 0.7 --msrm_prob 0.7 \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_mrm_msrm_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 10000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.7s.7_noitm_lr5e5_bs800

#### case12.3: speech + visual + text running on noitm + wwm + mrm and msrm with whole mask
# CUDA_VISIBLE_DEVICES=6 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --mrm_prob 0.8 --msrm_prob 0.8 \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_mrm_msrm_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 10000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.8s.8_noitm_lr5e5_bs800

#### case12.4: speech + visual + text running on noitm + wwm + mrm and msrm with whole mask
# CUDA_VISIBLE_DEVICES=7 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --mrm_prob 0.9 --msrm_prob 0.9 \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_mrm_msrm_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 10000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.9s.9_noitm_lr5e5_bs800

#### case12: speech + visual + text running on noitm + wwm + mrm and msrm with whole mask
# CUDA_VISIBLE_DEVICES=2 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --mrm_prob 1.0 --msrm_prob 1.0 --mlm_prob 1.0 \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_mrm_msrm_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 60000 --warmup_steps 6000 --valid_steps 10000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.1s.1t.1_noitm_lr5e5_bs800

## case13: speech + visual + text running on wwm + onespan - itm -- Use masked span ration 10% 20% 30% 40% 50% (一整个连续的区域)
# CUDA_VISIBLE_DEVICES=6 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --mask_speech_len_ratio 0.1 --mask_visual_len_ratio 0.1 \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_onespan_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 10000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_onespans.1v.1_noitm_lr5e5_bs800

# CUDA_VISIBLE_DEVICES=5 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --mask_speech_len_ratio 0.2 --mask_visual_len_ratio 0.2 \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_onespan_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 10000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_onespans.2v.2_noitm_lr5e5_bs800

# CUDA_VISIBLE_DEVICES=6 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --mask_speech_len_ratio 0.3 --mask_visual_len_ratio 0.3 \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_onespan_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 10000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_onespans.3v.3_noitm_lr5e5_bs800

# CUDA_VISIBLE_DEVICES=7 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --mask_speech_len_ratio 0.4 --mask_visual_len_ratio 0.4 \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_onespan_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 10000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_onespans.4v.4_noitm_lr5e5_bs800

# CUDA_VISIBLE_DEVICES=4 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --mask_speech_len_ratio 0.5 --mask_visual_len_ratio 0.5 \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_onespan_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 10000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_onespans.5v.5_noitm_lr5e5_bs800