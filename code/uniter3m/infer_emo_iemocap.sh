export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
dropout=0.1
corpus_name='iemocap'

# case0: text - finetune on gpu2
for cvNo in $(seq 1 10)
do
num_train_steps=1200
valid_steps=100
train_batch_size=32
inf_batch_size=32
frozens=0
CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
        --cvNo ${cvNo} --use_text \
        --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
        --corpus_name ${corpus_name} --cls_num 4 \
        --config config/infer_test_config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword_long3.json \
        --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_initby5corpus_emo5_text_1tasks_lr5e5_bs512_faceth0.5/ckpt/model_step_20000.pt \
        --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
        --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
        --IMG_DIM 342 --Speech_DIM 768 \
        --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
        --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
        --output_dir /data7/emobert/exp/evaluation/IEMOCAP/finetune/nomask-movies_v1v2v3_uniter3m_text_1tasks_train2w-lr${lr}_train${num_train_steps}_trnval
done