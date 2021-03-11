
CUDA_VISIBLE_DEVICES=2,3 python run_mlm.py \
    --model_name_or_path /data7/MEmoBert/emobert/exp/pretrain/only_v1v2v3_txt/bert_base_uncased_1e4_epoch10_100w_bs512/checkpoint-9760/ \
    --train_file /data7/MEmoBert/emobert/exp/pretrain/only_v1v2v3_txt/data/v1v2v3_new_trn.txt \
    --validation_file /data7/MEmoBert/emobert/exp/pretrain/only_v1v2v3_txt/data/v1v2v3_new_val.txt \
    --line_by_line \
    --max_seq_length 30 --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --do_eval \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --num_train_epochs 30 \
    --warmup_steps 0 \
    --report_to 'tensorboard' \
    --evaluation_strategy 'epoch' \
    --save_strategy 'epoch' \
    --lr_scheduler_type 'linear' \
    --output_dir /data7/emobert/exp/pretrain/only_v1v2v3_txt/evaluation_onv1v2v3