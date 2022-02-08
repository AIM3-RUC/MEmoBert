export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
dropout=0.1
corpus_name='iemocap'
corpus_name_L='IEMOCAP'

# case0: inference the whole set and explore the upper-bound
for cvNo in $(seq 1 1)
do
        num_train_steps=1200
        valid_steps=100
        train_batch_size=32
        inf_batch_size=32
        frozens=0
        checkpoint_dir=/data7/MEmoBert/emobert/exp/evaluation/${corpus_name_L}/finetune/
        CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python infer_emo.py \
                --cvNo ${cvNo} --use_text --use_visual \
                --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
                --corpus_name ${corpus_name} --cls_num 4 \
                --config config/infer_test_config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
                --checkpoint ${checkpoint_dir}/nomask-directTrain_upper-uniter3m_visual_text_vstype2-lr5e-5_train2000/drop0.1_frozen0_vqa_none/${cvNo}/ckpt/ \
                --frozen_en_layers ${frozens} --cls_type vqa --postfix none \
                --IMG_DIM 342 --Speech_DIM 768 \
                --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
                --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
                --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/inference/nomask-directTrain_upper-uniter3m_visual_text_valLV
done