export PYTHONPATH=/data7/MEmoBert
# Only can use one gpu
setname=$1
for cv_no in $(seq 1 1);
do
    CUDA_VISIBLE_DEVICES=1 horovodrun -np 1 python extract_fts.py \
        --txt_db "/data7/emobert/exp/evaluation/IEMOCAP/txt_db/${cv_no}/${setname}.db" \
        --img_db "/data7/emobert/exp/evaluation/IEMOCAP/feature/denseface_openface_mean_std_movie_no_mask/img_db/iemocap" \
        --checkpoint "/data7/emobert/exp/pretrain/nomask_movies_v1_uniter_mlm_mrfr_mrckl_3tasks/ckpt/model_step_100000.pt" \
        --model_config config/uniter-base.json \
        --output_dir /data7/emobert/exp/mmfts/iemocap/nomask_movies_v1_uniter_mlm_mrfr_mrckl_3tasks_2tasks_nofinetune/${cv_no}/${setname} --fp16 \
        --conf_th 0.0 --batch_size 400
done