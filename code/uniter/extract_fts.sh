export PYTHONPATH=/data7/MEmoBert
# Only can use one gpu
cv_no=$1
CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python extract_fts.py \
    --txt_db "/data7/emobert/exp/evaluation/IEMOCAP/txt_db/${cv_no}" \
    --img_db "/data7/emobert/img_db/movies_v1" \
    --checkpoint "/data7/emobert/exp/pretrain/nomask_movies_v1_uniter_4tasks/ckpt/model_step_100000.pt" \
    --model_config config/uniter-base.json \
    --output_dir /data7/emobert/exp/mmfts/iemocap/nomask_movies_v1_uniter_4tasks_nofinetune/${cv_no} --fp16 \
    --conf_th 0.0 --batch_size 400