export PYTHONPATH=/data7/MEmoBert
setname=$1
gpu_id=$2

corpus_name='IEMOCAP' # 'MSP-IMPROV' 'IEMOCAP'
corpus_name_small='iemocap' # iemocap 

for cv_no in $(seq 1 12);
do
    CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python extract_fts.py \
        --txt_db "/data7/emobert/exp/evaluation/${corpus_name}/txt_db/${cv_no}/${setname}.db" \
        --img_db "/data7/emobert/exp/evaluation/${corpus_name}/feature/denseface_openface_${corpus_name_small}_mean_std/img_db/${corpus_name_small}" \
        --checkpoint "/data7/emobert/exp/pretrain//ckpt/model_step_100000.pt" \
        --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
        --output_dir /data7/emobert/exp/uniter3m_fts/${corpus_name_small}/nomask_movies_v1_uniter_4tasks_melm0.5emo4_emotypeinput_openface_nofinetune/${cv_no}/${setname} --fp16 \
        --conf_th 0.0 --batch_size 32
done