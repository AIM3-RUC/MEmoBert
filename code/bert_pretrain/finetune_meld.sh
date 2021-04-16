# https://github.com/huggingface/transformers/tree/master/examples/text-classification
#注意: meld 数据集名称 test train 而 iemocap msp 数据集 trn tst
#直接finetune的时候 --model_name_or_path bert-base-uncased

source activate transformers
export PYTHONPATH=/data7/MEmoBert

gpuid=$1
pretrain_model_dir=/data7/MEmoBert/emobert/exp/mlm_pretrain/results/
output_dir=/data7/emobert/exp/finetune/onlytext

for cvNo in `seq 1 1`;
do
    for lr in 1e-5 2e-5;
    do
        bert_data_dir=/data7/emobert/exp/evaluation/MELD/bert_data
        CUDA_VISIBLE_DEVICES=${gpuid} python run_cls.py \
            --model_name_or_path ${pretrain_model_dir}/moviesv1v2v3/bert_base_uncased_2e5/checkpoint-34409/  \
            --train_file ${bert_data_dir}/train_val.csv \
            --validation_file ${bert_data_dir}/test.csv \
            --test_file ${bert_data_dir}/test.csv \
            --max_length 50 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --num_train_epochs 8 \
            --patience 2 \
            --learning_rate ${lr} \
            --lr_scheduler_type 'linear' \
            --output_dir ${output_dir}/meld_taskpretain_moviesv1v2v3-trnval_bert_base_lr${lr}_bs32_trnval/${cvNo}
    done
done


# pretrained dir = /data7/MEmoBert/emobert/exp/mlm_pretrain/results/
# ${pretrain_model_dir}/meld/bert_base_uncased_2e5_epoch10_bs64/
# ${pretrain_model_dir}/meld/bert_base_uncased_2e5_epoch10_bs64_trnval/
# ${pretrain_model_dir}/meld/bert_taskpretain_on_moviesv1v2v3_base_uncased_2e5_epoch10_bs64/
# ${pretrain_model_dir}/meld/bert_taskpretain_on_opensub1000w_base_uncased_2e5_epoch10_bs64/

## openpretain
##for opensubtile 1000w
    # ${pretrain_model_dir}/opensub/bert_base_uncased_1000w_linear_lr1e4_warm4k_bs256_acc2_4gpu/checkpoint-93980/
##for opensubtile 500w
    # ${pretrain_model_dir}/opensub/bert_base_uncased_500w_linear_lr1e4_warm4k_bs256_acc2_4gpu/checkpoint-48820/
##for opensubtile 100w
    # ${pretrain_model_dir}/opensub/bert_base_uncased_1e4_epoch10_100w_bs512/checkpoint-9760/ 
##for moviesv1v2v3 20w pretrain
    # ${pretrain_model_dir}/moviesv1v2v3/bert_base_uncased_2e5/checkpoint-34409/