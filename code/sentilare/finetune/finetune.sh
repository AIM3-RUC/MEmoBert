export PYTHONPATH=/data7/MEmoBert

gpuid=$1
pretrain_model_dir=/data7/emobert/resources/pretrained/sentilare/pretrain_model/
output_dir=/data7/emobert/exp/finetune/onlytext

### for meld
corpus_name='meld'
corpus_name_L='MELD'
data_dir=/data7/emobert/exp/evaluation/${corpus_name_L}/bert_data/
for lr in 2e-5 5e-5
do
    CUDA_VISIBLE_DEVICES=${gpuid} python run_sent_sentilr_roberta.py \
            --data_dir  ${data_dir} \
            --model_type roberta \
            --model_name_or_path roberta-base \
            --task_name meld \
            --do_train \
            --do_eval \
            --max_seq_length 50 \
            --per_gpu_train_batch_size 32 \
            --per_gpu_eval_batch_size 32 \
            --learning_rate ${lr} \
            --num_train_epochs 3 \
            --logging_steps 100 \
            --save_steps 100 \
            --warmup_steps 100 \
            --eval_all_checkpoints \
            --overwrite_output_dir \
            --seed 42 \
            --output_dir ${output_dir}/${corpus_name}_roberta_base_finetune_lr${lr}_bs32
done

### for msp and iemocap
# corpus_name='msp'
# corpus_name_L='MSP'

# for cvNo in `seq 11 12`;
# do
#     for lr in 2e-5 5e-5; 
#     do
#     bert_data_dir=/data7/emobert/exp/evaluation/${corpus_name_L}/bert_data/${cvNo}
#     CUDA_VISIBLE_DEVICES=${gpuid} python run_cls.py \
#         --model_name_or_path ${pretrain_model_dir}/moviesv1v2v3/bert_base_uncased_2e5/checkpoint-34409/  \
#         --cvNo ${cvNo} \
#         --train_file ${bert_data_dir}/trn_val.csv \
#         --validation_file ${bert_data_dir}/tst.csv \
#         --test_file ${bert_data_dir}/tst.csv \
#         --max_length 50 \
#         --per_device_train_batch_size 32 \
#         --per_device_eval_batch_size 32 \
#         --num_train_epochs 8 \
#         --patience 2 \
#         --learning_rate ${lr} \
#         --lr_scheduler_type 'linear' \
#         --output_dir ${output_dir}/${corpus_name}_taskpretrain_moviesv1v2v3-trnval_bert_base_lr${lr}_bs32_trnval/${cvNo}
#     done
# done