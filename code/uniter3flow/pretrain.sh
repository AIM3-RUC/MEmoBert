export PYTHONPATH=/data7/MEmoBert

# case1: stage1: text + wav2vec running on gpu7
# CUDA_VISIBLE_DEVICES=3,4 horovodrun -np 2 python pretrain.py \
#         --cvNo 0  --use_speech  \
#         --fix_text_encoder --fix_speech_encoder \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_rawwav_2optim_mlmitm.json \
#         --model_config config/uniter-3flow_swav2vec_c4.json \
#         --pretrained_text_checkpoint /data7/emobert/resources/pretrained/bert_base_model.pt \
#         --pretrained_audio_checkpoint /data7/emobert/resources/pretrained/wav2vec_base/wav2vec_base.pt \
#         --train_batch_size 128 --val_batch_size 128 --gradient_accumulation_steps 4 \
#         --learning_rate 5e-5 --lr_sched_type 'linear' \
#         --max_txt_len 30 \
#         --conf_th 0.5 --max_bb 36 --min_bb 10 \
#         --speech_conf_th 1.0 --max_frames 96000 --min_frames 10 \
#         --num_train_steps 100000 --warmup_steps 5000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/flow3-stage1-text12_fix-wav2vec2_fix-cross4Update-nomask_movies_v1v2v3_uniter_mlmitm_lr5e5_notypeemb_train5w

# CUDA_VISIBLE_DEVICES=4,5 horovodrun -np 2 python pretrain.py \
#         --cvNo 0  --use_speech \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_rawwav_2optim_mlmitm.json \
#         --model_config config/uniter-3flow_swav2vec_c4.json \
#         --checkpoint /data7/emobert/exp/pretrain/flow3-stage1-text12_fix-wav2vec2_fix-cross4Update-nomask_movies_v1v2v3_uniter_mlmitm_lr5e5_notypeemb_train5w/ckpt/model_step_100000.pt \
#         --train_batch_size 128 --val_batch_size 128 --gradient_accumulation_steps 4 \
#         --learning_rate 1e-5 --lr_sched_type 'linear' \
#         --max_txt_len 30 \
#         --conf_th 0.5 --max_bb 36 --min_bb 10 \
#         --speech_conf_th 1.0 --max_frames 96000 --min_frames 10 \
#         --num_train_steps 50000 --warmup_steps 0 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/flow3-stage2-basedonstage1_text12_fix-wav2vec2_fix-cross4Update_train10w-nomask_movies_v1v2v3_uniter_mlmitm_lr1e5_notypeemb_train5w

# case2: stage12: text + wav2vec running on gpu7
# CUDA_VISIBLE_DEVICES=6,7 horovodrun -np 2 python pretrain.py \
#         --cvNo 0  --use_speech  \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_rawwav_2optim_mlmitm.json \
#         --model_config config/uniter-3flow_swav2vec_c4.json \
#         --pretrained_text_checkpoint /data7/emobert/resources/pretrained/bert_base_model.pt \
#         --pretrained_audio_checkpoint /data7/emobert/resources/pretrained/wav2vec_base/wav2vec_base.pt \
#         --train_batch_size 256 --val_batch_size 256 --gradient_accumulation_steps 4 \
#         --learning_rate 5e-5 --lr_sched_type 'linear' \
#         --max_txt_len 30 \
#         --conf_th 0.5 --max_bb 36 --min_bb 10 \
#         --speech_conf_th 1.0 --max_frames 96000 --min_frames 10 \
#         --num_train_steps 50000 --warmup_steps 2500 --valid_steps 2500 \
#         --output_dir /data7/emobert/exp/pretrain/flow3-stage12-text12Update-wav2vec2Update-cross4Update-nomask_movies_v1v2v3_uniter_mlmitm_lr5e5_notypeemb_bs1024_train10w