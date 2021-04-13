export PYTHONPATH=/data7/MEmoBert

# flow3_text12_visual4_cross4, batchsize=30, train 20000 running
# flow3_text12_visual4_cross2, batchsize=30, train 20000 running
# flow3_text12_visual2_cross2, batchsize=30, train 20000 running

# batchsize=180000/(30*2*8) * 30 = 7500
# CUDA_VISIBLE_DEVICES=4,5 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --config config/pretrain-movies-v1v2v3-base-2gpu_rawimg_2optim_mlmitm.json \
#         --model_config config/uniter-3flow_v4c4.json \
#         --pretrained_text_checkpoint /data7/MEmoBert/emobert/resources/pretrained/bert_base_model.pt \
#         --use_visual \
#         --train_batch_size 30 --val_batch_size 30 --gradient_accumulation_steps 8 \
#         --learning_rate 5e-4 --lr_sched_type 'linear' \
#         --max_txt_len 30 --IMG_DIM 112 --image_data_augmentation \
#         --conf_th 0.5 --max_bb 36 \
#         --num_train_steps 20000 --warmup_steps 2000 --valid_steps 1000 \
#         --output_dir /data7/emobert/exp/pretrain/flow3_text12_visual4_cross4_typeEmb-nomask_movies_v1v2v3_uniter_mlmitm_lr5e4-backbone_scratch_optimFalse-bs480_faceth0.5


# CUDA_VISIBLE_DEVICES=6 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --config config/pretrain-movies-v1v2v3-base-2gpu_rawimg_2optim_mlmitm.json \
#         --model_config config/uniter-3flow_v4c4.json \
#         --pretrained_text_checkpoint /data7/MEmoBert/emobert/resources/pretrained/bert_base_model.pt \
#         --use_visual \
#         --train_batch_size 20 --val_batch_size 20 --gradient_accumulation_steps 8 \
#         --learning_rate 5e-5 --lr_sched_type 'linear' \
#         --max_txt_len 30 --IMG_DIM 112 --image_data_augmentation \
#         --use_backbone_optim --backbone_learning_rate 6e-5 \
#         --conf_th 0.5 --max_bb 36 \
#         --num_train_steps 20000 --warmup_steps 2000 --valid_steps 1000 \
#         --output_dir /data7/emobert/exp/pretrain/flow3_text12_visual4_cross4_typeEmb-nomask_movies_v1v2v3_uniter_mlmitm_lr5e4-textbackbone_optim1e4-bs480_faceth0.5