
CUDA_VISIBLE_DEVICES=4,5 python run_mlm.py \
    --model_name_or_path bert-base-uncased \
    --train_file /data7/MEmoBert/emobert/exp/pretrain/only_v1v2v3_txt/data/v1v2v3_new_trn.txt \
    --validation_file /data7/MEmoBert/emobert/exp/pretrain/only_v1v2v3_txt/data/v1v2v3_new_val.txt \
    --line_by_line \
    --max_seq_length 30 --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 2 \
    --do_train --do_eval \
    --learning_rate 2e-5 \
    --num_train_epochs 20 \
    --warmup_steps 1000 \
    --report_to 'tensorboard' \
    --evaluation_strategy 'epoch' \
    --save_strategy 'epoch' \
    --lr_scheduler_type 'linear' \
    --output_dir /data7/emobert/exp/pretrain/only_v1v2v3_txt/bert_base_uncased_2e5_epoch20_batch256_wm1000