export PYTHONPATH=/data7/MEmoBert
setname=$1
gpu_id=$2
# for cv_no in $(seq 1 12);
# do
#     CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python extract_fts.py \
#         --txt_db "/data7/emobert/exp/evaluation/MSP-IMPROV/txt_db/${cv_no}/${setname}.db" \
#         --img_db "/data7/emobert/exp/evaluation/MSP-IMPROV/feature/denseface_openface_msp_mean_std/img_db/msp" \
#         --checkpoint "/data7/emobert/exp/pretrain/nomask_movies_v1_uniter_mlm_mrfr_mrckl_3tasks/ckpt/model_step_100000.pt" \
#         --model_config config/uniter-base.json \
#         --output_dir /data7/emobert/exp/mmfts/msp/nomask_movies_v1_uniter_mlm_mrfr_mrckl_3tasks_openface_nofinetune/${cv_no}/${setname} --fp16 \
#         --conf_th 0.0 --batch_size 300
# done

# for finetuned model
for cv_no in $(seq 1 12);
do
    CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python extract_fts.py \
        --txt_db "/data7/emobert/exp/evaluation/MSP-IMPROV/txt_db/${cv_no}/${setname}.db" \
        --img_db "/data7/emobert/exp/evaluation/MSP-IMPROV/feature/denseface_openface_msp_mean_std/img_db/msp" \
        --checkpoint "/data7/emobert/exp/finetune/msp_openface_baseon_nomask_movies_v1_uniter_mlm_mrfr_mrckl_3tasks/drop0.1_frozen0_emocls_none/${cv_no}" \
        --model_config config/uniter-base.json \
        --output_dir /data7/emobert/exp/mmfts/msp/nomask_movies_v1_uniter_mlm_mrfr_mrckl_3tasks_openface_finetuned/${cv_no}/${setname} --fp16 \
        --conf_th 0.0 --batch_size 300
done