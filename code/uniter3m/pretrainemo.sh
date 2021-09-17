export PYTHONPATH=/data7/MEmoBert

# 正常的第一阶段加入情感预训练 melm-wwm merm-span 

## case1: visual + speech + text - noitm + melm-wwm
# CUDA_VISIBLE_DEVICES=2,3 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_visual --use_speech  \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_melm_wwm_span_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#          --max_txt_len 50 --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_speech_text_5tasks_melm_wwm_span_noitm_lr5e5_bs800

## case2: visual + speech + text - noitm + melm-wwm
# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_visual  \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_visual_text_3tasks_melm_wwm_span_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#          --max_txt_len 50 --IMG_DIM 342 --Speech_DIM 768 \
#          --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visaul_text_3tasks_melm_wwm_span_noitm_lr5e5_bs800


## case3: speech + text - noitm + melm-wwm
# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_speech  \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_wav2vec_text_2tasks_melm_wwm_span_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#          --max_txt_len 50 --IMG_DIM 342 --Speech_DIM 768 \
#          --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speech_text_2tasks_melm_wwm_span_noitm_lr5e5_bs800

# 方案2: 在第一阶段预训练以general pretrain为主, 第二阶段采用情感相关的预训练任务, 比如 melm, merm, eitm, emolare.
