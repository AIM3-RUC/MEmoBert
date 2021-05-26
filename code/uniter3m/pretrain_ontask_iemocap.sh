export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
corpus_name='iemocap'

# # case1: taskptrain visual + text
# for cvNo in $(seq 1 10)
# do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                 --cvNo ${cvNo} --use_visual \
#                 --config config/pretrain-task-${corpus_name}-base-2gpu_4tasks.json \
#                 --model_config config/uniter-base-emoword_nomultitask.json \
#                 --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_4tasks_lr5e5_bs512_faceth0.5/ckpt/model_step_20000.pt \
#                 --learning_rate 2e-05 --lr_sched_type 'linear'  --gradient_accumulation_steps 4 \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.0 --max_bb 36 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size 64 --val_batch_size 64 \
#                 --num_train_steps 1000 --warmup_steps 100 --valid_steps 1000 \
#                 --output_dir /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_text_4tasks-faceth0.0_trnval/${cvNo}
# done

# # case2: taskptrain visual + speech
# for cvNo in $(seq 1 10)
# do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                 --cvNo ${cvNo} --use_speech\
#                 --config config/pretrain-task-${corpus_name}-base_wav2vec-2gpu_3tasks.json \
#                 --model_config config/uniter-base-emoword_nomultitask.json \
#                 --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_wav2vec_text_3tasks_mlmitmmsrfr_lr5e5_bs512_faceth0.5/ckpt/model_step_20000.pt \
#                 --learning_rate 2e-05 --lr_sched_type 'linear'  --gradient_accumulation_steps 4 \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.0 --max_bb 36 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size 64 --val_batch_size 64 \
#                 --num_train_steps 1000 --warmup_steps 100 --valid_steps 1000 \
#                 --output_dir /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_wav2vec_text_3tasks_lr5e5-faceth0.0_trnval/${cvNo}
# done

# # case3: taskptrain visual + speech + text
# for cvNo in $(seq 1 10)
# do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                 --cvNo ${cvNo} --use_speech --use_visual \
#                 --config config/pretrain-task-${corpus_name}-base-2gpu_5tasks.json \
#                 --model_config config/uniter-base-emoword_nomultitask.json \
#                 --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_vstype1_lr5e5_bs512_faceth0.5/ckpt/model_step_30000.pt \
#                 --learning_rate 2e-05 --lr_sched_type 'linear'  --gradient_accumulation_steps 4 \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.0 --max_bb 36 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size 64 --val_batch_size 64 \
#                 --num_train_steps 1000 --warmup_steps 100 --valid_steps 1000 \
#                 --output_dir /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_vstype1_lr5e5-faceth0.0_trnval/${cvNo}
# done

# case4: taskptrain visual + speech + text
# for cvNo in $(seq 1 10)
# do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                 --cvNo ${cvNo} --use_speech --use_visual \
#                 --config config/pretrain-task-${corpus_name}-base-2gpu_4tasks-sentiword_emocls.json \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_emoclsSoft_withitm_vstype2_lr5e5_bs1024/ckpt/model_step_30000.pt \
#                 --learning_rate 2e-5 --lr_sched_type 'linear'  --gradient_accumulation_steps 4 \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.0 --max_bb 36 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size 64 --val_batch_size 64 \
#                 --num_train_steps 2000 --warmup_steps 0 --valid_steps 2000 \
#                 --output_dir /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_emoclsSoft_withitm_vstype2_lr5e5-4tasks_emoclsSoft_noitm_trnval/${cvNo}
# done

# case4: taskptrain visual + speech + text
for cvNo in $(seq 1 10)
do
        CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
                --cvNo ${cvNo} --n_workers 4 --use_speech --use_visual \
                --config config/pretrain-task-${corpus_name}-base-2gpu_5tasks-sentiword_emocls.json \
                --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
                --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_emoclsSoft_withitm_vstype2_lr5e5_bs1024/ckpt/model_step_30000.pt \
                --learning_rate 2e-5 --lr_sched_type 'linear'  --gradient_accumulation_steps 4 \
                --IMG_DIM 342 --Speech_DIM 768 \
                --conf_th 0.0 --max_bb 36 \
                --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
                --train_batch_size 64 --val_batch_size 64 \
                --num_train_steps 2000 --warmup_steps 0 --valid_steps 2000 \
                --output_dir /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_emoclsSoft_withitm_vstype2_lr5e5-5tasks_emoclsSoft_withitm_trnval/${cvNo}
done