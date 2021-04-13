# source activate transformers
source activate transformers
export PYTHONPATH=/data7/MEmoBert

result_dir=/data7/MEmoBert/emobert/exp/mlm_pretrain/results

### for meld, 9000 / 64 * 10 = 1400
# corpus_name='meld'
# corpus_name_L='MELD'
# bert_data_dir=/data7/emobert/exp/evaluation/${corpus_name_L}/bert_data/
# CUDA_VISIBLE_DEVICES=4 python run_mlm.py \
#     --model_name_or_path  bert-base-uncased \
#     --do_train --do_eval \
#     --train_file /data7/MEmoBert/emobert/exp/evaluation/${corpus_name_L}/bert_data/train_val.txt \
#     --validation_file /data7/MEmoBert/emobert/exp/evaluation/${corpus_name_L}/bert_data/test.txt \
#     --line_by_line \
#     --max_seq_length 50 \
#     --per_device_train_batch_size 64 \
#     --per_device_eval_batch_size 64 \
#     --gradient_accumulation_steps 2 \
#     --learning_rate 2e-5 \
#     --max_grad_norm 5.0 \
#     --num_train_epochs 10 \
#     --evaluation_strategy 'epoch' \
#     --lr_scheduler_type 'linear' \
#     --load_best_model_at_end true \
#     --overwrite_output_dir true \
#     --output_dir ${result_dir}/${corpus_name}/bert_taskpretain_base_uncased_2e5_epoch10_bs64_trnval

### for msp 
corpus_name='iemocap'
corpus_name_L='IEMOCAP'
for cvNo in `seq 1 10`;
do
bert_data_dir=/data7/emobert/exp/evaluation/${corpus_name_L}/bert_data/${cvNo}
CUDA_VISIBLE_DEVICES=4 python run_mlm.py \
    --model_name_or_path bert-base-uncased \
    --do_train --do_eval \
    --train_file /data7/emobert/exp/evaluation/${corpus_name_L}/bert_data/${cvNo}/trn_val.txt \
    --validation_file /data7/emobert/exp/evaluation/${corpus_name_L}/bert_data/${cvNo}/val.txt \
    --line_by_line \
    --max_seq_length 50 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --max_grad_norm 5.0 \
    --num_train_epochs 10 \
    --evaluation_strategy 'epoch' \
    --lr_scheduler_type 'linear' \
    --load_best_model_at_end true \
    --overwrite_output_dir true \
    --output_dir ${result_dir}/${corpus_name}/${cvNo}/bert_taskpretain_base_uncased_2e5_epoch10_bs64_trnval
done

##for opensubtile 1000w
    # train_file=/data7/MEmoBert/emobert/exp/mlm_pretrain/datasets/OpenSubtitlesV2018/OpenSubtitles.en-zh_cn.en_trn1000w.txt
    # validation_file=/data7/MEmoBert/emobert/exp/mlm_pretrain/datasets/OpenSubtitlesV2018/OpenSubtitles.en-zh_cn.en_val10w.txt
    # model == ${result_dir}/opensub/bert_base_uncased_1000w_linear_lr1e4_warm4k_bs256_acc2_4gpu/checkpoint-93980  eval-loss=1.7316
##for opensubtile 500w
    # train_file=/data7/MEmoBert/emobert/exp/mlm_pretrain/datasets/OpenSubtitlesV2018/OpenSubtitles.en-zh_cn.en_trn500w.txt
    # validation_file=/data7/MEmoBert/emobert/exp/mlm_pretrain/datasets/OpenSubtitlesV2018/OpenSubtitles.en-zh_cn.en_val10w.txt
    # model == ${result_dir}/opensub/bert_base_uncased_500w_linear_lr1e4_warm4k_bs256_acc2_4gpu/checkpoint-48820/  eval-loss=1.78
##for opensubtile 100w
    # train_file=/data7/MEmoBert/emobert/exp/mlm_pretrain/datasets/OpenSubtitlesV2018/OpenSubtitles.en-zh_cn.en_trn100w.txt
    # validation_file=/data7/MEmoBert/emobert/exp/mlm_pretrain/datasets/OpenSubtitlesV2018/OpenSubtitles.en-zh_cn.en_val10w.txt
    # model = ${result_dir}/opensub/bert_base_uncased_1e4_epoch10_100w_bs512/checkpoint-9760/  eval-loss=1.88
##for moviesv1v2v3 20w pretrain
    # train_file=/data7/MEmoBert/emobert/exp/mlm_pretrain/datasets/moviesv123/v1v2v3_new_trn.txt
    # validation_file=/data7/MEmoBert/emobert/exp/mlm_pretrain/datasets/moviesv123/v1v2v3_new_val.txt
    # model = ${result_dir}/moviesv1v2v3/bert_base_uncased_2e5/checkpoint-34409 eval-loss=1.8564

##for meld 1w pretrain
    # train_file=/data7/MEmoBert/emobert/exp/evaluation/MELD/bert_data/train.txt
    # validation_file=/data7/MEmoBert/emobert/exp/evaluation/MELD/bert_data/val.txt
    # model = ${result_dir}/meld/bert_base_uncased_2e5_epoch10_bs64/checkpoint-390  eval-loss=1.7342

##for opensub100w pretrain + meld 1w pretrain
    # train_file=/data7/MEmoBert/emobert/exp/evaluation/MELD/bert_data/train.txt
    # validation_file=/data7/MEmoBert/emobert/exp/evaluation/MELD/bert_data/val.txt
    # model = ${result_dir}/meld/bert_taskpretain_on_opensub1000w_base_uncased_2e5_epoch10_bs64/checkpoint-390  eval-loss=1.528

##for moviesv1v2v3 pretrain + meld 1w pretrain
    # train_file=/data7/MEmoBert/emobert/exp/evaluation/MELD/bert_data/train.txt
    # validation_file=/data7/MEmoBert/emobert/exp/evaluation/MELD/bert_data/val.txt
    # model = ${result_dir}/meld/bert_taskpretain_on_moviesv1v2v3_base_uncased_2e5_epoch10_bs64/checkpoint-390  eval-loss=1.693

##for iemocap 4k pretrain
    # train_file=/data7/MEmoBert/emobert/exp/evaluation/IEMOCAP/bert_data/1/trn.txt
    # validation_file=/data7/MEmoBert/emobert/exp/evaluation/IEMOCAP/bert_data/1/val.txt
    # cv1 model = ${result_dir}/iemocap/1/bert_base_uncased_2e5_epoch10_bs64/checkpoint-153 eval-loss 1.64
    # cv2 model = ${result_dir}/iemocap/2/bert_base_uncased_2e5_epoch10_bs64/checkpoint-170 eval-loss 1.627
    # cv3 model = ${result_dir}/iemocap/3/bert_base_uncased_2e5_epoch10_bs64/checkpoint-162 eval-loss 1.73
    # cv4 model = ${result_dir}/iemocap/4/bert_base_uncased_2e5_epoch10_bs64/checkpoint-162 eval-loss 1.53
    # cv5 model = ${result_dir}/iemocap/5/bert_base_uncased_2e5_epoch10_bs64/checkpoint-153 eval-loss 1.58
    # cv6 model = ${result_dir}/iemocap/6/bert_base_uncased_2e5_epoch10_bs64/checkpoint-153 eval-loss 1.58
    # cv7 model = ${result_dir}/iemocap/7/bert_base_uncased_2e5_epoch10_bs64/checkpoint-180 eval-loss 1.57
    # cv8 model = ${result_dir}/iemocap/8/bert_base_uncased_2e5_epoch10_bs64/checkpoint-180 eval-loss 1.548
    # cv9 model = ${result_dir}/iemocap/9/bert_base_uncased_2e5_epoch10_bs64/checkpoint-153 eval-loss 1.59
    # cv10 model = ${result_dir}/iemocap/10/bert_base_uncased_2e5_epoch10_bs64/checkpoint-153 eval-loss 1.56

##for msp pretrain
#  1-4 7-8:${result_dir}/msp/${cvNo}/bert_base_uncased_2e5_epoch10_bs64/checkpoint-130
#  5-6 9-10:${result_dir}/msp/${cvNo}/bert_base_uncased_2e5_epoch10_bs64/checkpoint-120
#  11 12:${result_dir}/msp/${cvNo}/bert_base_uncased_2e5_epoch10_bs64/checkpoint-110