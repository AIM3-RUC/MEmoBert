# conda deactivate
export PYTHONPATH=/root/lrc/MEmoBert
gpu_id=$1

corpus_name='IEMOCAP' # 'MSP'
corpus_name_small='iemocap' # msp  
data_path=/root/lrc/data
exp_name=nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800
checkpoint_dir=/root/lrc/pretrained_model/$exp_name/ckpt

## for a v l input
for setname in tst val trn
do
    for cvNo in $(seq 1 10)
    do
        CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python extract_fts.py \
            --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
            --txt_db $data_path/evaluation/${corpus_name}/txt_db/${cvNo}/${setname}_wwm_nrcemolex_prompt_mask_iam.db \
            --img_db $data_path/evaluation/${corpus_name}/feature/denseface_openface_iemocap_mean_std_torch/img_db/fc/feat_th0.0_max36_min10 \
            --speech_db $data_path/evaluation/${corpus_name}/feature/wav2vec_db_3mean/feat_th1.0_max64_min10 \
            --checkpoint  $checkpoint_dir/model_step_40000.pt \
            --output_dir $data_path/uniter3m_fts/${corpus_name_small}/$exp_name/a/${cvNo}/${setname} \
            --fp16 --batch_size 100 --use_speech
    done
done

# ### for only l/a/v/la/lv/av/lav input
# for setname in tst val trn
# do
#     for cvNo in $(seq 1 12)
#     do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python extract_fts.py \
#             --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#             --txt_db $data_path/evaluation/${corpus_name}/txt_db/${cvNo}/${setname}_wwm_nrcemolex_prompt_mask_iam.db \
#             --img_db $data_path/evaluation/${corpus_name}/feature/denseface_openface_${corpus_name_small}_mean_std_torch/img_db/fc/feat_th0.0_max36_min10 \
#             --speech_db $data_path/evaluation/${corpus_name}/feature/wav2vec_db_3mean/feat_th1.0_max64_min10 \
#             --checkpoint $checkpoint_dir/model_step_40000.pt \
#             --output_dir $data_dir/uniter3m_fts/${corpus_name_small}/$exp_name/${cvNo}/${setname} \
#             --fp16 --batch_size 100 --use_visual --use_speech --use_text
#     done
# done