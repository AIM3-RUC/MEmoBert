source deactivate base

export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
dropout=0.1


################# Part1.1: Directly infer (source IEMOCAP and Target MSP)################################################### 
################# Part2.1: Directly+BERT infer (source IEMOCAP and Target MSP)################################################### 
################# Part3.1: MEmoBERT+Finetune (source IEMOCAP and Target MSP )################################################### 
################# Part4.1: MEmoBERT+Prompt (source IEMOCAP and Target MSP )################################################### 
################# Part1.2: Directly infer (source MSP and Target IEMOCAP)################################################### 
################# Part2.2: Directly+BERT infer (source  MSP and Target IEMOCAP)################################################### 
################# Part3.2: MEmoBERT+Finetune (source MSP and Target IEMOCAP)################################################### 
################# Part4.2: MEmoBERT+Prompt (source MSP and Target IEMOCAP)################################################### 

################# Part1.1: Directly infer (source IEMOCAP and Target MSP)###################################################
source_corpus_name='iemocap'
source_corpus_name_L='IEMOCAP'
target_corpus_name='msp'
target_corpus_name_L='MSP'

for cvNo in $(seq 1 1)
do
        inf_batch_size=32
        frozens=0
        checkpoint_dir=/data7/MEmoBert/emobert/exp/evaluation/${source_corpus_name_L}/finetune/
        CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python infer_emo.py \
                --cvNo ${cvNo} --use_text --use_visual --use_speech \
                --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
                --corpus_name ${corpus_name} --cls_num 4 \
                --config config/infer_test_config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
                --checkpoint ${checkpoint_dir}//drop0.1_frozen0_vqa_none/${cvNo}/ckpt/ \
                --frozen_en_layers ${frozens} --cls_type vqa --postfix none \
                --IMG_DIM 342 --Speech_DIM 768 \
                --inf_batch_size ${inf_batch_size} \
                --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/inference/nomask-
done