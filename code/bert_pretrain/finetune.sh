# https://github.com/huggingface/transformers/tree/master/examples/text-classification

# --model_name_or_path  bert-base-uncased \
gpuid=$1
CUDA_VISIBLE_DEVICES=${gpuid} python run_glue.py \
    --model_name_or_path roberta-base \
    --train_file /data7/MEmoBert/emobert/exp/evaluation/MELD/bert_data/train.csv \
    --validation_file /data7/MEmoBert/emobert/exp/evaluation/MELD/bert_data/val.csv \
    --test_file /data7/MEmoBert/emobert/exp/evaluation/MELD/bert_data/test.csv \
    --do_train \
    --do_eval --do_predict \
    --max_seq_length 50 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --num_train_epochs 20 \
    --learning_rate 1e-5 \
    --warmup_steps 50 \
    --weight_decay 0.001 \
    --lr_scheduler_type 'constant' \
    --evaluation_strategy 'epoch' \
    --save_strategy 'epoch' \
    --save_total_limit 5 \
    --output_dir /data7/emobert/exp/evaluation/MELD/results/MeldFinetune-roberta-lr8e6-wd.001-epoch20_max50