export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
corpus_name='iemocap'
corpus_name_L='IEMOCAP'

# case1: taskptrain visual + speech + text - itm
# for cvNo in $(seq 1 10)
# do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                 --cvNo ${cvNo} --n_workers 4 --use_speech --use_visual \
#                 --config config/pretrain-task-${corpus_name}-base-2gpu_5tasks-noitm.json \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_speech_text_5tasks_noitm_lr5e5_bs800/ckpt/model_step_30000.pt \
#                 --learning_rate 2e-5 --lr_sched_type 'linear'  --gradient_accumulation_steps 4 \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --train_batch_size 64 --val_batch_size 64 \
#                 --num_train_steps 2000 --warmup_steps 0 --valid_steps 2000 \
#                 --output_dir /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_noitm-5tasks_noitm_trnval/${cvNo}
# done

# # case2: taskptrain visual + speech + text - itm + emocls
# for cvNo in $(seq 1 10)
# do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                 --cvNo ${cvNo} --n_workers 4 --use_speech --use_visual \
#                 --config config/pretrain-task-${corpus_name}-base-2gpu_5tasks-emocls_noitm.json \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_emocls_noitm_lr5e5_bs800/ckpt/model_step_30000.pt \
#                 --learning_rate 2e-5 --lr_sched_type 'linear'  --gradient_accumulation_steps 4 \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --train_batch_size 64 --val_batch_size 64 \
#                 --num_train_steps 2000 --warmup_steps 0 --valid_steps 2000 \
#                 --output_dir /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_emocls_noitm-5tasks_emocls_noitm_trnval/${cvNo}
# done

# case3: taskptrain visual + speech + text - itm + wwm + span
# for cvNo in $(seq 7 7)
# do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                 --cvNo ${cvNo} --n_workers 4 --use_speech --use_visual \
#                 --config config/pretrain-task-${corpus_name}-base-2gpu_5tasks-wwm_span_noitm.json \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                 --learning_rate 2e-5 --lr_sched_type 'linear'  --gradient_accumulation_steps 4 \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --train_batch_size 100 --val_batch_size 100 \
#                 --num_train_steps 2000 --warmup_steps 0 --valid_steps 2000 \
#                 --output_dir /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm-5tasks_wwm_span_noitm_trnval/${cvNo}
# done


# case4: taskptrain visual + speech + text - itm + wwm + span + emocls
# for cvNo in $(seq 1 1)
# do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                 --cvNo ${cvNo} --n_workers 4 --use_speech --use_visual \
#                 --config config/pretrain-task-${corpus_name}-base-2gpu_5tasks-wwm_span_emocls_noitm.json \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_emocls_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                 --learning_rate 2e-5 --lr_sched_type 'linear'  --gradient_accumulation_steps 4 \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --train_batch_size 100 --val_batch_size 100 \
#                 --num_train_steps 2000 --warmup_steps 0 --valid_steps 2000 \
#                 --output_dir /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_emocls_noitm-5tasks_wwm_span_emocls_noitm_trnval/${cvNo}
# done