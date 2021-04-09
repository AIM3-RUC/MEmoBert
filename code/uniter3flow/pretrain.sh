export PYTHONPATH=/data7/MEmoBert

# batchsize=180000/(30*2*8) *20 = 7500
CUDA_VISIBLE_DEVICES=7 horovodrun -np 1 python pretrain.py \
        --cvNo 0 --config config/pretrain-movies-v1v2v3-base-2gpu_rawimg_2optim_mlmitm.json \
        --model_config config/uniter-3flow.json \
        --pretrained_text_checkpoint /data7/MEmoBert/emobert/resources/pretrained/bert_base_model.pt \
        --use_visual \
        --train_batch_size 5 --val_batch_size 5 --gradient_accumulation_steps 8 \
        --learning_rate 1e-4 --lr_sched_type 'linear' \
        --max_txt_len 30 --IMG_DIM 112 --image_data_augmentation \
        --conf_th 0.5 --max_bb 36 \
        --num_train_steps 10000 --warmup_steps 1000 --valid_steps 1000 \
        --output_dir /data7/emobert/exp/pretrain/resnet3d_nomask_movies_v1v2v3_uniter_mlmitm_lr1e4-backbone_pretrained_optimFalse-bs480_faceth0.5