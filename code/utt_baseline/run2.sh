# eg:
# bash run.sh V 0 1
source deactivate base
export PYTHONPATH=/data9/MEmoBert
set -e
gpu=$1

corpus_name=iemocap
#  onlytrn 表示train:val:test的模式，只有训练集合训练，否则 train+val训练，test测试

# 单模态的实验 == pretrained features all 768 dim
# for modality in A
# do
#     for run_idx in 1 2
#     do
#         for cvNo in $(seq 1 10)
#         do
#             python -u run_baseline.py --gpu_id $gpu --modality=$modality \
#                 --dataset_mode ${corpus_name}_pretrained --cvNo ${cvNo} \
#                 --pretained_ft_type utt_baseline  --postfix onlytrn \
#                 --num_threads 0 --run_idx=$run_idx  \
#                 --max_epoch 40 --patience 5 --fix_lr_epoch 20 --warmup_epoch 3 \
#                 --dropout_rate 0.5  --learning_rate 3e-4 --batch_size 32 \
#                 --v_ft_type '' --v_input_size 768 --max_visual_tokens 50  \
#                 --a_ft_type '' --a_input_size 768 --max_acoustic_tokens 64 \
#                 --l_ft_type ''  --l_input_size 768 --max_text_tokens 50 \
#                 --ft_name nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800_nofinetune_onlyspeech \
#                 --l_hidden_size 256 --v_hidden_size 256 --a_hidden_size 256 --mid_fusion_layers '256,128'
#         done
#     done
# done

# # # 多模态的实验
# for modality in LVA
# do
#     for run_idx in 2
#     do
#         for cvNo in $(seq 6 10)
#         do
#             python run_baseline.py --gpu_id $gpu --modality=$modality  \
#                 --dataset_mode ${corpus_name}_pretrained --cvNo ${cvNo} \
#                 --pretained_ft_type utt_baseline  --postfix onlytrn \
#                 --num_threads 0 --run_idx=$run_idx  \
#                 --max_epoch 40 --patience 5 --fix_lr_epoch 20 --warmup_epoch 3 \
#                 --dropout_rate 0.5  --learning_rate 3e-4 --batch_size 64 \
#                 --v_ft_type '' --v_input_size 768 --max_visual_tokens 50  \
#                 --a_ft_type '' --a_input_size 768 --max_acoustic_tokens 64 \
#                 --l_ft_type '' --l_input_size 768 --max_text_tokens 50 \
#                 --ft_name nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800_nofinetune \
#                 --l_hidden_size 256 --v_hidden_size 256 --a_hidden_size 256 --mid_fusion_layers '512,256'
#         done
#     done
# done

# --mid_fusion_layers '256,128'  # 单模态
# --mid_fusion_layers '512,256'  # 多模态