# https://github.com/huggingface/transformers/tree/master/examples/text-classification
#注意: meld 数据集名称 test train 而 iemocap msp 数据集 trn tst
#直接finetune的时候 --model_name_or_path bert-base-uncased

export PYTHONPATH=/data7/MEmoBert

gpuid=$1
pretrain_model_dir=/data7/MEmoBert/emobert/exp/mlm_pretrain/results
output_dir=/data7/emobert/exp/finetune/onlytext

corpus_name='iemocap'
corpus_name_L='IEMOCAP'
for cvNo in `seq 8 10`;
do
bert_data_dir=/data7/emobert/exp/evaluation/${corpus_name_L}/bert_data/${cvNo}
CUDA_VISIBLE_DEVICES=${gpuid} python run_cls.py \
    --model_name_or_path  ${pretrain_model_dir}/${corpus_name}/${cvNo}/bert_taskpretain_on_moviesv1v2v3_base_uncased_2e5_epoch50_bs64/ \
    --train_file ${bert_data_dir}/trn.csv \
    --validation_file ${bert_data_dir}/val.csv \
    --test_file ${bert_data_dir}/tst.csv \
    --max_length 50 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 10 \
    --patience 2 \
    --learning_rate 1e-5 \
    --lr_scheduler_type 'linear' \
    --output_dir ${output_dir}/${corpus_name}_baseon_moviesv1v2v3_pretrained_and_taskpretrain_epoch50-lr1e5_bs32_max50/${cvNo}
done

## meld1. meld_lr2e5_warm50_bs64 'acc': 0.64789, 'wuar': 0.64789, 'wf1': 0.62388,
#     --model_name_or_path  bert-base-uncased \
## meld2. meld_basedon_meld_pretrained-lr2e5_warm50_bs64 'acc': 0.6429, 'wuar': 0.6429, 'wf1': 0.63025,
#     --model_name_or_path /data7/MEmoBert/emobert/exp/mlm_pretrain/results/meld/bert_base_uncased_2e5_epoch10_bs64/checkpoint-390
## meld3. meld_basedon_moviesv1v2v3_pretrained-lr2e5_warm50_bs64  'acc':0.65747 , 'wuar':0.65747, 'wf1': 0.63668
#     --model_name_or_path moviesv1v2v3/bert_base_uncased_2e5/checkpoint-34409 
## meld4 meld_basedon_opensub100w_pretrained-lr2e5_warm50_bs64 'acc': 0.64636, 'wuar': 0.64636, 'wf1': 0.6328,
#     --model_name_or_path opensub/bert_base_uncased_1e4_epoch10_100w_bs512/checkpoint-9760/
## meld4 meld_basedon_opensub500w_pretrained-lr2e5_warm50_bs64 
#     --model_name_or_path opensub/bert_base_uncased_500w_linear_lr1e4_warm4k_bs256_acc2_4gpu/checkpoint-48820

## iemocap1. iemocap_lr2e5_warm50_bs64/cv1 
#     --model_name_or_path  bert-base-uncased \
# iemocap cv1=cv2=cv5=cv6=cv9=cv10=170   cv3=cv4=cv7=cv8=180