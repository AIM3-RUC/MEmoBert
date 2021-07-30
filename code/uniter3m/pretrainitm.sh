export PYTHONPATH=/data7/MEmoBert


################# Part4: Explore the ITM task ################################################### 
## case1: visual + text - itm
# CUDA_VISIBLE_DEVICES=1 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_visual  \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_visual_text_4tasks_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 50 \
#         --train_batch_size 200 --val_batch_size 200 \
#         --num_train_steps 20000 --warmup_steps 2000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_4tasks_noitm_lr5e5_bs800

## case2: speech + text - itm
# CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_speech  \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_wav2vec_text_3tasks_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 50 \
#         --train_batch_size 200 --val_batch_size 200 \
#         --num_train_steps 20000 --warmup_steps 2000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speech_text_3tasks_noitm_lr5e5_bs800

## case3: visual + text + speech - itm
# CUDA_VISIBLE_DEVICES=2 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_visual --use_speech \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 50 \
#         --train_batch_size 200 --val_batch_size 200 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_speech_text_5tasks_noitm_lr5e5_bs800

## case4: visual + text + speech + onemodalnegitm + itm
# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --itm_neg_prob 0.5 \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_span_onemodalnegitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --max_txt_len 50 --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 25000 --warmup_steps 3000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_onemodalnegitm_lr5e5_bs800

# ## case5: visual + text + speech + onemodalnegitm + itm + more negative samples
# CUDA_VISIBLE_DEVICES=0,1 HOROVOD_CACHE_CAPACITY=1024 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --itm_neg_prob 0.9 \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_span_onemodalnegitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --max_txt_len 50 --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 150 --val_batch_size 150 \
#         --num_train_steps 25000 --warmup_steps 3000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_onemodalnegitm_neg0.9_lr5e5_bs800

# ## case6: visual + text + speech + only onemodalnegitm + more negative samples
# CUDA_VISIBLE_DEVICES=2,3 HOROVOD_CACHE_CAPACITY=1024 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --itm_neg_prob 0.9 \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_span_only_onemodalnegitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --max_txt_len 50 --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 150 --val_batch_size 150 \
#         --num_train_steps 25000 --warmup_steps 3000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_only_onemodalnegitm_neg0.9_lr5e5_bs800

# ## case5: visual + text + speech + itm + hard-negative
# CUDA_VISIBLE_DEVICES=4,5 HOROVOD_CACHE_CAPACITY=1024 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --itm_neg_samples 150 \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_span_onemodalnegitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --max_txt_len 50 --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 150 --val_batch_size 150 \
#         --num_train_steps 25000 --warmup_steps 3000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_onemodalnegitm_neg0.9_lr5e5_bs800