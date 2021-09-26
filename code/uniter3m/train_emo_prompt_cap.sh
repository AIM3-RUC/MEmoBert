
source deactivate base
export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
dropout=0.1
corpus_name='iemocap'
corpus_name_L='IEMOCAP'

## Outline
################# Part1: Explore the ITM task ################################################### 
################# Part2: Explore WWM + Span tasks ################################################### 

################# Part1: Explore WWM + Span tasks ################################################### 
# case1: taskptrain visual + speech + text - itm + wwm + span
for cvNo in $(seq 1 10)
do
        CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
                --cvNo ${cvNo} --n_workers 4 --use_speech --use_visual \
                --seed 4321 \
                --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_mask_iam_prompt.json \
                --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
                --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
                --learning_rate 2e-5 --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
                --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
                --train_batch_size 32 --val_batch_size 32 \
                --num_train_steps 2000 --warmup_steps 100 --valid_steps 100 \
                --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-mask_iam_prompt_diffseed_trnval/${cvNo}
done


# for cvNo in $(seq 1 10)
# do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                 --cvNo ${cvNo} --n_workers 4 --use_speech --use_visual \
#                 --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_mask_heis_prompt.json \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                 --learning_rate 2e-5 --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                 --max_txt_len 100 --IMG_DIM 342 --Speech_DIM 768 \
#                 --train_batch_size 32 --val_batch_size 32 \
#                 --num_train_steps 2000 --warmup_steps 200 --valid_steps 200 \
#                 --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-mask_heis_prompt_trnval/${cvNo}
# done