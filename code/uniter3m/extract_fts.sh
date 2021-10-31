source deactivate base
export PYTHONPATH=/data7/MEmoBert
gpu_id=$1

corpus_name='IEMOCAP' # 'MSP-IMPROV' 'IEMOCAP'
corpus_name_small='iemocap' # iemocap 
for setname in tst val trn
do
    for cvNo in $(seq 1 10)
    do
        CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python extract_fts.py \
            --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
            --txt_db /data7/emobert/exp/evaluation/${corpus_name}/txt_db/${cvNo}/${setname}_wwm_nrcemolex_prompt_mask_iam.db \
            --img_db /data7/emobert/exp/evaluation/${corpus_name}/feature/denseface_openface_iemocap_mean_std_torch/img_db/fc/feat_th0.0_max36_min10 \
            --speech_db /data7/emobert/exp/evaluation/${corpus_name}/feature/wav2vec_db_3mean/feat_th1.0_max64_min10 \
            --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
            --output_dir /data7/emobert/exp/uniter3m_fts/${corpus_name_small}/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800_nofinetune/${cvNo}/${setname} \
            --fp16 --batch_size 100 --use_visual --use_speech 
    done
done