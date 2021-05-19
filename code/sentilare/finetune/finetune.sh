export PYTHONPATH=/data7/MEmoBert

gpuid=$1
pretrain_model_dir=/data7/emobert/resources/pretrained/sentilare/pretrain_model/
output_dir=/data7/emobert/exp/finetune/onlytext

# evaluate on sst, on val in paper is 55.04 and we can get 56.04 perfermance. OK
# corpus_name='sst'
# for lr in 2e-5
# do
#     data_dir=/data7/emobert/resources/pretrained/sentilare/raw_data/sent/sst/
#     CUDA_VISIBLE_DEVICES=${gpuid} python run_sent_sentilr_roberta.py \
#             --cvNo 1 \
#             --data_dir  ${data_dir} \
#             --model_type roberta \
#             --model_name_or_path ${pretrain_model_dir} \
#             --task_name sst \
#             --do_train \
#             --do_eval \
#             --max_seq_length 50 \
#             --per_gpu_train_batch_size 32 \
#             --per_gpu_eval_batch_size 32 \
#             --learning_rate ${lr} \
#             --num_train_epochs 8 \
#             --warmup_steps 100 \
#             --eval_all_checkpoints \
#             --overwrite_output_dir \
#             --seed 42 \
#             --output_dir ${output_dir}/${corpus_name}_roberta_base_finetune_lr${lr}_bs32_m2
# done

# evaluate on imdb, on val in paper is 95.96 and we can get 0.9436 perfermance. max_seq_length=256.
# corpus_name='imdb'
# for lr in 2e-5
# do
#     data_dir=/data7/emobert/resources/pretrained/sentilare/raw_data/sent/${corpus_name}/
#     CUDA_VISIBLE_DEVICES=${gpuid} python run_sent_sentilr_roberta.py \
#             --cvNo 1 \
#             --data_dir  ${data_dir} \
#             --model_type roberta \
#             --model_name_or_path ${pretrain_model_dir} \
#             --task_name imdb \
#             --do_train \
#             --do_eval \
#             --max_seq_length 256 \
#             --per_gpu_train_batch_size 24 \
#             --per_gpu_eval_batch_size 24 \
#             --learning_rate ${lr} \
#             --num_train_epochs 8 \
#             --warmup_steps 100 \
#             --eval_all_checkpoints \
#             --overwrite_output_dir \
#             --seed 42 \
#             --output_dir ${output_dir}/${corpus_name}_roberta_base_finetune_lr${lr}_bs32_m2
# done

### for meld
# corpus_name='meld'
# corpus_name_L='MELD'
# for lr in 2e-5 5e-5
# do
#     data_dir=/data7/emobert/exp/evaluation/${corpus_name_L}/bert_data/
#     CUDA_VISIBLE_DEVICES=${gpuid} python run_sent_sentilr_roberta.py \
#             --cvNo 1 \
#             --data_dir  ${data_dir} \
#             --model_type roberta \
#             --model_name_or_path ${pretrain_model_dir} \
#             --task_name ${corpus_name} \
#             --do_train \
#             --do_eval \
#             --max_seq_length 50 \
#             --per_gpu_train_batch_size 32 \
#             --per_gpu_eval_batch_size 32 \
#             --learning_rate ${lr} \
#             --num_train_epochs 8 \
#             --warmup_steps 100 \
#             --eval_all_checkpoints \
#             --overwrite_output_dir \
#             --seed 42 \
#             --output_dir ${output_dir}/${corpus_name}_roberta_base_finetune_lr${lr}_bs32_m2
# done

### for msp and iemocap
# corpus_name='msp'
# corpus_name_L='MSP'
# for cvNo in `seq 1 12`;
# do
#     for lr in 2e-5 5e-5
#     do
#         data_dir=/data7/emobert/exp/evaluation/${corpus_name_L}/bert_data/${cvNo}
#         CUDA_VISIBLE_DEVICES=${gpuid} python run_sent_sentilr_roberta.py \
#                 --cvNo ${cvNo} \
#                 --data_dir  ${data_dir} \
#                 --model_type roberta \
#                 --model_name_or_path ${pretrain_model_dir} \
#                 --task_name iemocap \
#                 --do_train \
#                 --do_eval \
#                 --max_seq_length 50 \
#                 --per_gpu_train_batch_size 12 \
#                 --per_gpu_eval_batch_size 12 \
#                 --learning_rate ${lr} \
#                 --num_train_epochs 1 \
#                 --warmup_steps 100 \
#                 --eval_all_checkpoints \
#                 --overwrite_output_dir \
#                 --seed 42 \
#                 --output_dir ${output_dir}/${corpus_name}_roberta_base_finetune_lr${lr}_bs32_m2/${cvNo} \
#                 --results_path ${output_dir}/${corpus_name}_roberta_base_finetune_lr${lr}_bs32_m2/results.tsv
#     done
# done

# corpus_name='iemocap'
# corpus_name_L='IEMOCAP'
# for cvNo in `seq 1 10`;
# do
#     for lr in 2e-5 5e-5
#     do
#         data_dir=/data7/emobert/exp/evaluation/${corpus_name_L}/bert_data/${cvNo}
#         CUDA_VISIBLE_DEVICES=${gpuid} python run_sent_sentilr_roberta.py \
#                 --cvNo ${cvNo} \
#                 --data_dir  ${data_dir} \
#                 --model_type roberta \
#                 --model_name_or_path ${pretrain_model_dir} \
#                 --task_name iemocap \
#                 --do_train \
#                 --do_eval \
#                 --max_seq_length 50 \
#                 --per_gpu_train_batch_size 12 \
#                 --per_gpu_eval_batch_size 12 \
#                 --learning_rate ${lr} \
#                 --num_train_epochs 6 \
#                 --warmup_steps 100 \
#                 --eval_all_checkpoints \
#                 --overwrite_output_dir \
#                 --seed 42 \
#                 --output_dir ${output_dir}/${corpus_name}_roberta_base_finetune_lr${lr}_bs32_m2/${cvNo} \
#                 --results_path ${output_dir}/${corpus_name}_roberta_base_finetune_lr${lr}_bs32_m2/results.tsv

#     done
# done