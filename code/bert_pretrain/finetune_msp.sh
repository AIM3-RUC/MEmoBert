# https://github.com/huggingface/transformers/tree/master/examples/text-classification
#注意: meld 数据集名称 test train 而 iemocap msp 数据集 trn tst
#直接finetune的时候 --model_name_or_path bert-base-uncased roberta-base

export PYTHONPATH=/data7/MEmoBert
gpuid=$1
pretrain_model_dir=/data7/MEmoBert/emobert/exp/mlm_pretrain/results
output_dir=/data7/emobert/exp/finetune/onlytext

corpus_name='msp'
corpus_name_L='MSP'

# for cvNo in `seq 1 12`
# do
#     for lr in 2e-5 3e-5 5e-5
#     do
#     bert_data_dir=/data7/emobert/exp/evaluation/${corpus_name_L}/bert_data/${cvNo}
#     CUDA_VISIBLE_DEVICES=${gpuid} python run_cls.py \
#         --corpus_name ${corpus_name} \
#         --model_name_or_path /data7/emobert/exp/text_emo_model/all_3corpus_emo5_bert_base_lr2e-5_bs32_debug/ckpt/epoch-1/  \
#         --restore_pt_checkpoint /data7/emobert/exp/text_emo_model/all_3corpus_emo5_bert_base_lr2e-5_bs32_debug/ckpt/epoch-1/model_step_1.pt \
#         --cvNo ${cvNo} \
#         --train_file ${bert_data_dir}/trn_val.csv \
#         --validation_file ${bert_data_dir}/tst.csv \
#         --test_file ${bert_data_dir}/tst.csv \
#         --max_length 50 \
#         --per_device_train_batch_size 32 \
#         --per_device_eval_batch_size 32 \
#         --num_train_epochs 6 \
#         --patience 2 \
#         --learning_rate ${lr} \
#         --lr_scheduler_type 'linear' \
#         --output_dir ${output_dir}/${corpus_name}_all_3corpus_emo5_bert_base_lr${lr}_bs32_trnval/${cvNo}
#     done
# done

for cvNo in `seq 7 12`
do
    for lr in 2e-5
    do
    bert_data_dir=/data7/emobert/exp/evaluation/${corpus_name_L}/bert_data/${cvNo}
    CUDA_VISIBLE_DEVICES=${gpuid} python run_cls.py \
        --corpus_name ${corpus_name} \
        --model_name_or_path bert-base-uncased  \
        --cvNo ${cvNo} \
        --train_file ${bert_data_dir}/trn_val.csv \
        --validation_file ${bert_data_dir}/tst.csv \
        --test_file ${bert_data_dir}/tst.csv \
        --max_length 50 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --num_train_epochs 6 \
        --patience 2 \
        --learning_rate ${lr} \
        --lr_scheduler_type 'linear' \
        --output_dir ${output_dir}/${corpus_name}_bert_base_lr${lr}_bs32_trnval/${cvNo}
    done
done

# for cvNo in `seq 1 12`
# do
#     for lr in 2e-5 3e-5 5e-5
#     do
#     bert_data_dir=/data7/emobert/exp/evaluation/${corpus_name_L}/bert_data/${cvNo}
#     CUDA_VISIBLE_DEVICES=${gpuid} python run_cls.py \
#         --corpus_name ${corpus_name} \
#         --model_name_or_path /data7/emobert/exp/text_emo_model/bert_movie_model/  \
#         --restore_pt_checkpoint /data7/emobert/exp/text_emo_model/bert_movie_model/model_step_0.pt  \
#         --cvNo ${cvNo} \
#         --train_file ${bert_data_dir}/trn_val.csv \
#         --validation_file ${bert_data_dir}/tst.csv \
#         --test_file ${bert_data_dir}/tst.csv \
#         --max_length 50 \
#         --per_device_train_batch_size 32 \
#         --per_device_eval_batch_size 32 \
#         --num_train_epochs 6 \
#         --patience 2 \
#         --learning_rate ${lr} \
#         --lr_scheduler_type 'linear' \
#         --output_dir ${output_dir}/${corpus_name}_bert_movie_emo5_bert_base_lr${lr}_bs32_trnval/${cvNo}
#     done
# done

# pretrained dir = /data7/MEmoBert/emobert/exp/mlm_pretrain/results/
# ${pretrain_model_dir}/${corpus_name}/${cvNo}/bert_base_uncased_2e5_epoch10_bs64/
# ${pretrain_model_dir}/${corpus_name}/${cvNo}/bert_taskpretain_on_moviesv1v2v3_base_uncased_2e5_epoch10_bs64/
# ${pretrain_model_dir}/${corpus_name}/${cvNo}/bert_taskpretain_on_opensub1000w_base_uncased_2e5_epoch10_bs64/

## supervised-pretrain
# /data7/emobert/exp/text_emo_model/all_5corpus_emo7_bert_base_lr2e-5_bs32/ckpt/
# /data7/emobert/exp/text_emo_model/all_5corpus_emo5_bert_base_lr2e-5_bs32/ckpt/
# /data7/emobert/exp/text_emo_model/all_5corpus_emo7_bert_base_lr2e-5_bs32/ckpt/

## openpretain
##for opensubtile 1000w
    # ${pretrain_model_dir}/opensub/bert_base_uncased_1000w_linear_lr1e4_warm4k_bs256_acc2_4gpu/checkpoint-93980/
##for opensubtile 500w
    # ${pretrain_model_dir}/opensub/bert_base_uncased_500w_linear_lr1e4_warm4k_bs256_acc2_4gpu/checkpoint-48820/
##for opensubtile 100w
    # ${pretrain_model_dir}/opensub/bert_base_uncased_1e4_epoch10_100w_bs512/checkpoint-9760/ 
##for moviesv1v2v3 20w pretrain
    # ${pretrain_model_dir}/moviesv1v2v3/bert_base_uncased_2e5/checkpoint-34409/