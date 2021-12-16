source deactivate base

export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
dropout=0.1


################# Part1.1: Directly infer (source IEMOCAP and Target MSP)################################################### 
################# Part2.1: Directly+BERT infer (source IEMOCAP and Target MSP)################################################### 
################# Part3.1: MEmoBERT+Finetune infer (source IEMOCAP and Target MSP )################################################### 
################# Part4.1: MEmoBERT+Prompt infer (source IEMOCAP and Target MSP )################################################### 
################# Part1.2: Directly infer (source MSP and Target IEMOCAP)################################################### 
################# Part2.2: Directly+BERT infer (source  MSP and Target IEMOCAP)################################################### 
################# Part3.2: MEmoBERT+Finetune infer (source MSP and Target IEMOCAP)################################################### 
################# Part4.2: MEmoBERT+Prompt infer (source MSP and Target IEMOCAP)################################################### 

# ################# Part1.1: Directly infer (source IEMOCAP and Target MSP)###################################################
# source_corpus_name='iemocap'
# source_corpus_name_L='IEMOCAP'
# target_corpus_name='msp'
# target_corpus_name_L='MSP'
# for cvNo in $(seq 1 10)
# do
#         inf_batch_size=32
#         checkpoint_dir=/data7/MEmoBert/emobert/exp/evaluation/${source_corpus_name_L}/finetune/
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python infer_emo.py \
#                 --cvNo ${cvNo} --use_text --use_visual --use_speech \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --cls_num 4 \
#                 --config config/infer_test_config/infer-emo-${target_corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint ${checkpoint_dir}/nomask-directTrain-FromScratch-uniter3m_visual_wav2vec_text-lr5e-5_train2000_trnval_seed5678/drop0.1_frozen0_vqa_none/${cvNo}/ckpt/ \
#                 --cls_type vqa --postfix none \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --inf_batch_size ${inf_batch_size} \
#                 --output_dir /data7/emobert/exp/evaluation/${target_corpus_name_L}/inference/nomask-source-nomask-directTrain-FromScratch-uniter3m_visual_wav2vec_text-lr5e-5_train2000_trnval_seed5678
# done

# ################# Part1.2: Directly infer (source MSP and Target IEMOCAP)################################################### 
# source_corpus_name='msp'
# source_corpus_name_L='MSP'
# target_corpus_name='iemocap'
# target_corpus_name_L='IEMOCAP'
# for cvNo in $(seq 1 12)
# do
#         inf_batch_size=32
#         checkpoint_dir=/data7/MEmoBert/emobert/exp/evaluation/${source_corpus_name_L}/finetune/
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python infer_emo.py \
#                 --cvNo ${cvNo} --use_text --use_visual --use_speech \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --cls_num 4 \
#                 --config config/infer_test_config/infer-emo-${target_corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint ${checkpoint_dir}/nomask-directTrain-FromScratch-uniter3m_visual_wav2vec_text-lr5e-5_train2000_trnval_seed5678/drop0.1_frozen0_vqa_none/${cvNo}/ckpt/ \
#                 --cls_type vqa --postfix none \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --inf_batch_size ${inf_batch_size} \
#                 --output_dir /data7/emobert/exp/evaluation/${target_corpus_name_L}/inference/nomask-source-nomask-directTrain-FromScratch-uniter3m_visual_wav2vec_text-lr5e-5_train2000_trnval_seed5678
# done


################# Part2.1: Directly+BERT infer (source IEMOCAP and Target MSP)################################################### 
# source_corpus_name='iemocap'
# source_corpus_name_L='IEMOCAP'
# target_corpus_name='msp'
# target_corpus_name_L='MSP'
# for cvNo in $(seq 1 10)
# do
#         inf_batch_size=32
#         checkpoint_dir=/data7/MEmoBert/emobert/exp/evaluation/${source_corpus_name_L}/finetune/
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python infer_emo.py \
#                 --cvNo ${cvNo} --use_text --use_visual --use_speech \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --cls_num 4 \
#                 --config config/infer_test_config/infer-emo-${target_corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint ${checkpoint_dir}/nomask-directTrain-uniter3m_visual_wav2vec_text-lr5e-5_train2000_run3_trnval/drop0.1_frozen0_vqa_none/${cvNo}/ckpt/ \
#                 --cls_type vqa --postfix none \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --inf_batch_size ${inf_batch_size} \
#                 --output_dir /data7/emobert/exp/evaluation/${target_corpus_name_L}/inference/nomask-source-nomask-directTrain-uniter3m_visual_wav2vec_text-lr5e-5_train2000_run3_trnval
# done

################# Part2.2: Directly+BERT infer (source  MSP and Target IEMOCAP)################################################### 
# source_corpus_name='msp'
# source_corpus_name_L='MSP'
# target_corpus_name='iemocap'
# target_corpus_name_L='IEMOCAP'
# for cvNo in $(seq 1 12)
# do
#         inf_batch_size=32
#         checkpoint_dir=/data7/MEmoBert/emobert/exp/evaluation/${source_corpus_name_L}/finetune/
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python infer_emo.py \
#                 --cvNo ${cvNo} --use_text --use_visual --use_speech \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --cls_num 4 \
#                 --config config/infer_test_config/infer-emo-${target_corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint ${checkpoint_dir}/nomask-directTrain-uniter3m_visual_wav2vec_text-lr5e-5_train2000_run3_trnval/drop0.1_frozen0_vqa_none/${cvNo}/ckpt/ \
#                 --cls_type vqa --postfix none \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --inf_batch_size ${inf_batch_size} \
#                 --output_dir /data7/emobert/exp/evaluation/${target_corpus_name_L}/inference/nomask-source-nomask-directTrain-uniter3m_visual_wav2vec_text-lr5e-5_train2000_run3_trnval
# done


################# Part3.1: MEmoBERT+Finetune infer (source IEMOCAP and Target MSP )################################################### 
# source_corpus_name='iemocap'
# source_corpus_name_L='IEMOCAP'
# target_corpus_name='msp'
# target_corpus_name_L='MSP'
# for cvNo in $(seq 1 10)
# do
#         inf_batch_size=32
#         checkpoint_dir=/data7/MEmoBert/emobert/exp/evaluation/${source_corpus_name_L}/finetune/
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python infer_emo.py \
#                 --cvNo ${cvNo} --use_text --use_visual --use_speech \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --cls_num 4 \
#                 --config config/infer_test_config/infer-emo-${target_corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint ${checkpoint_dir}/nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_span_noitm_train4w-lr5e-5_train1500_run3_trnval/drop0.1_frozen0_vqa_none/${cvNo}/ckpt/ \
#                 --cls_type vqa --postfix none \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --inf_batch_size ${inf_batch_size} \
#                 --output_dir /data7/emobert/exp/evaluation/${target_corpus_name_L}/inference/nomask-source-uniter3m_speechwav2vec_5tasks_wwm_span_noitm_train4w-lr5e-5_train1500_run3_trnval
# done

################# Part3.2: MEmoBERT+Finetune infer (source IEMOCAP and Target MSP )################################################### 
# source_corpus_name='msp'
# source_corpus_name_L='MSP'
# target_corpus_name='iemocap'
# target_corpus_name_L='IEMOCAP'
# for cvNo in $(seq 1 12)
# do
#         inf_batch_size=32
#         checkpoint_dir=/data7/MEmoBert/emobert/exp/evaluation/${source_corpus_name_L}/finetune/
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python infer_emo.py \
#                 --cvNo ${cvNo} --use_text --use_visual --use_speech \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --cls_num 4 \
#                 --config config/infer_test_config/infer-emo-${target_corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint ${checkpoint_dir}/nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_span_noitm_train4w-lr5e-5_train1500_run3_trnval/drop0.1_frozen0_vqa_none/${cvNo}/ckpt/ \
#                 --cls_type vqa --postfix none \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --inf_batch_size ${inf_batch_size} \
#                 --output_dir /data7/emobert/exp/evaluation/${target_corpus_name_L}/inference/nomask-source-nomask-uniter3m_speechwav2vec_5tasks_wwm_span_noitm_train4w-lr5e-5_train1500_run3_trnval
# done


################# Part4.1: MEmoBERT+Prompt infer (source IEMOCAP and Target MSP )###################################################
# 需要重新构建inference代码或者用训练的代码
source_corpus_name='iemocap'
source_corpus_name_L='IEMOCAP'
target_corpus_name='msp'
target_corpus_name_L='MSP'
for cvNo in $(seq 1 1)
do
        inf_batch_size=32
        checkpoint_dir=/data7/MEmoBert/emobert/exp/prompt_pretrain/
        CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python infer_emo.py \
                --cvNo ${cvNo} --use_text --use_visual --use_speech \
                --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
                --cls_num 4 \
                --config config/infer_test_config/infer-emo-${source_corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
                --checkpoint ${checkpoint_dir}/iemocap_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_prompt_lr5e-5_run3_trnval/${cvNo}/ckpt/ \
                --cls_type vqa --postfix none \
                --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
                --inf_batch_size ${inf_batch_size} \
                --output_dir /data7/emobert/exp/evaluation/${target_corpus_name_L}/inference/nomask-source-uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_prompt_lr5e-5_run3_trnval
done

################# Part4.2: MEmoBERT+Prompt infer (source MSP and Target IEMOCAP)################################################### 
# source_corpus_name='msp'
# source_corpus_name_L='MSP'
# target_corpus_name='iemocap'
# target_corpus_name_L='IEMOCAP'
# for cvNo in $(seq 1 12)
# do
#         inf_batch_size=32
#         checkpoint_dir=/data7/MEmoBert/emobert/exp/prompt_pretrain/
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python infer_emo.py \
#                 --cvNo ${cvNo} --use_text --use_visual --use_speech \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --cls_num 4 \
#                 --config config/infer_test_config/infer-emo-${target_corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint ${checkpoint_dir}/msp_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_prompt_lr5e-5_run3_trnval/${cvNo}/ckpt/ \
#                 --cls_type vqa --postfix none \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --inf_batch_size ${inf_batch_size} \
#                 --output_dir /data7/emobert/exp/evaluation/${target_corpus_name_L}/inference/nomask-source-nomask-uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_prompt_lr5e-5_run3_trnval
# done