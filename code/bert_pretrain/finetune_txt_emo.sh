export PYTHONPATH=/data7/MEmoBert

# 在训练的OpenSubtile的预训练模型上，然后用20w的文本情感标注数据进行训练，得到Finetune后的模型，用Val作为验证集合，训练10次应该足够了吧
# 然后确定最好的分类模型，然后用于 MEmobert 的初始化, 早知道就保存为hugging-face的格式了。
# 确定的最终方案是无 OpenSubtile Pretrain 的模型, 
# emo4: Epoch 0: {'total': 12808, 'acc': 0.66052467207995, 'wuar': 0.66052467207995, 'wf1': 0.6823828889234733, 'uwf1': 0.5469848449912948}
# emo5: Epoch 0: {'total': 13343, 'acc': 0.6453571160908341, 'wuar': 0.6453571160908341, 'wf1': 0.6699305583240993, 'uwf1': 0.5117386853066482}
# emo7: Epoch 1: {'total': 13819, 'acc': 0.5862942325783341, 'wuar': 0.5862942325783341, 'wf1': 0.6099784294483714, 'uwf1': 0.4397123516114925}

gpuid=$1
pretrain_model_dir=/data7/emobert/exp/mlm_pretrain/results/opensub/bert_base_uncased_1000w_linear_lr1e4_warm4k_bs256_acc2_4gpu/checkpoint-93980/
output_dir=/data7/emobert/exp/text_emo_model/

# bert_data_dir=/data7/emobert/text_emo_corpus/all_5corpus/emo7_bert_data/
# CUDA_VISIBLE_DEVICES=${gpuid} python run_cls.py \
#     --model_name_or_path ${output_dir}/all_5corpus_emo7_bert_base_lr2e-5_bs32/ckpt/ \
#     --validation_file ${bert_data_dir}/val.csv \
#     --test_file ${bert_data_dir}/train.csv \
#     --validation_pred_path ${bert_data_dir}/validation_pred.npy \
#     --test_pred_path ${bert_data_dir}/test_pred.npy \
#     --max_length 64 \
#     --per_device_eval_batch_size 32 \
#     --output_dir ${output_dir}/all_5corpus_emo7_bert_base_lr2e-5_bs32/

# for lr in 2e-5
# do
#     bert_data_dir=/data7/emobert/text_emo_corpus/all_5corpus/emo7_bert_data/
#     CUDA_VISIBLE_DEVICES=${gpuid} python run_cls.py \
#         --model_name_or_path bert-base-uncased \
#         --train_file ${bert_data_dir}/train.csv \
#         --validation_file ${bert_data_dir}/train.csv \
#         --test_file ${bert_data_dir}/val.csv \
#         --max_length 64 \
#         --per_device_train_batch_size 32 \
#         --per_device_eval_batch_size 32 \
#         --num_train_epochs 5 \
#         --patience 3 \
#         --learning_rate ${lr} \
#         --lr_scheduler_type 'linear' \
#         --output_dir ${output_dir}/all_5corpus_emo7_bert_base_lr${lr}_bs32_debug/
# done

# for lr in 4e-5
# do
#     bert_data_dir=/data7/emobert/text_emo_corpus/all_5corpus/emo7_bert_data/
#     CUDA_VISIBLE_DEVICES=${gpuid} python run_cls.py \
#         --model_name_or_path ${pretrain_model_dir} \
#         --train_file ${bert_data_dir}/train.csv \
#         --validation_file ${bert_data_dir}/val.csv \
#         --test_file ${bert_data_dir}/val.csv \
#         --max_length 64 \
#         --per_device_train_batch_size 32 \
#         --per_device_eval_batch_size 32 \
#         --num_train_epochs 5 \
#         --patience 1 \
#         --learning_rate ${lr} \
#         --lr_scheduler_type 'linear' \
#         --output_dir ${output_dir}/all_5corpus_emo7_opensub1000w_pretrained_bert_lr${lr}_bs32/
# done


# for lr in 2e-5
# do
#     bert_data_dir=/data7/emobert/text_emo_corpus/all_5corpus/emo5_bert_data/
#     CUDA_VISIBLE_DEVICES=${gpuid} python run_cls.py \
#         --model_name_or_path bert-base-uncased \
#         --train_file ${bert_data_dir}/train.csv \
#         --validation_file ${bert_data_dir}/train.csv \
#         --test_file ${bert_data_dir}/val.csv \
#         --max_length 64 \
#         --per_device_train_batch_size 32 \
#         --per_device_eval_batch_size 32 \
#         --num_train_epochs 6 \
#         --patience 3 \
#         --learning_rate ${lr} \
#         --lr_scheduler_type 'linear' \
#         --output_dir ${output_dir}/all_5corpus_emo5_bert_base_lr${lr}_bs32_debug/
# done


# for lr in 2e-5 4e-5
# do
#     bert_data_dir=/data7/emobert/text_emo_corpus/all_5corpus/emo5_bert_data/
#     CUDA_VISIBLE_DEVICES=${gpuid} python run_cls.py \
#         --model_name_or_path  ${pretrain_model_dir} \
#         --train_file ${bert_data_dir}/train.csv \
#         --validation_file ${bert_data_dir}/val.csv \
#         --test_file ${bert_data_dir}/val.csv \
#         --max_length 64 \
#         --per_device_train_batch_size 32 \
#         --per_device_eval_batch_size 32 \
#         --num_train_epochs 5 \
#         --learning_rate ${lr} \
#         --lr_scheduler_type 'linear' \
#         --output_dir ${output_dir}/all_5corpus_emo5_opensub1000w_pretrained_bert_lr${lr}_bs32/
# done

# for lr in 2e-5
# do
#     bert_data_dir=/data7/emobert/text_emo_corpus/all_5corpus/emo4_bert_data/
#     CUDA_VISIBLE_DEVICES=${gpuid} python run_cls.py \
#         --model_name_or_path bert-base-uncased \
#         --train_file ${bert_data_dir}/train.csv \
#         --validation_file ${bert_data_dir}/train.csv \
#         --test_file ${bert_data_dir}/val.csv \
#         --max_length 64 \
#         --per_device_train_batch_size 32 \
#         --per_device_eval_batch_size 32 \
#         --num_train_epochs 6 \
#         --patience 3 \
#         --learning_rate ${lr} \
#         --lr_scheduler_type 'linear' \
#         --output_dir ${output_dir}/all_5corpus_emo4_bert_base_lr${lr}_bs32_debug/
# done