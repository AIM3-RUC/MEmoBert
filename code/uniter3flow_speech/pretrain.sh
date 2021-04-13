export PYTHONPATH=/data7/MEmoBert

# flow3_text12_speech4_cross4, batchsize=30, train 20000 running
# flow3_text12_speech4_cross2, batchsize=30, train 20000 running

CUDA_VISIBLE_DEVICES=5 horovodrun -np 1 python pretrain.py \
        --cvNo 0 --config config/pretrain-movies-v1v2v3-base-2gpu_comparE_2optim_mlmitm.json \
        --model_config config/uniter-3flow_s4c4.json \
        --pretrained_text_checkpoint /data7/MEmoBert/emobert/resources/pretrained/bert_base_model.pt \
        --use_speech \
        --train_batch_size 50 --val_batch_size 50 --gradient_accumulation_steps 8 \
        --learning_rate 5e-5 --lr_sched_type 'linear' \
        --max_txt_len 30 \
        --speech_conf_th 1.0 --max_frames 360 --min_frames 10 \
        --num_train_steps 20000 --warmup_steps 2000 --valid_steps 1000 \
        --output_dir /data7/emobert/exp/pretrain/flow3_text12_speech4_cross4_typeEmb-nomask_movies_v1v2v3_uniter_mlmitm_lr5e5-textbackbone_optimFalse-bs480_faceth0.5

# CUDA_VISIBLE_DEVICES=5 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --config config/pretrain-movies-v1v2v3-base-2gpu_comparE_2optim_mlmitm.json \
#         --model_config config/uniter-3flow_s4c4.json \
#         --pretrained_text_checkpoint /data7/MEmoBert/emobert/resources/pretrained/bert_base_model.pt \
#         --use_speech \
#         --train_batch_size 3 --val_batch_size 3 --gradient_accumulation_steps 8 \
#         --learning_rate 5e-5 --lr_sched_type 'linear' \
#         --use_backbone_optim --backbone_learning_rate 6e-5 \
#         --max_txt_len 30 \
#         --conf_th 1.0 --max_frames 360 --min_frames 10 \
#         --num_train_steps 20000 --warmup_steps 2000 --valid_steps 1000 \
#         --output_dir /data7/emobert/exp/pretrain/flow3_text12_speech4_cross4_typeEmb-nomask_movies_v1v2v3_uniter_mlmitm_lr5e5-textbackbone_optim-bs480_faceth0.5