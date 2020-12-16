export PYTHONPATH=/data7/MEmoBert
# Only can use one gpu
CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python extract_fts.py \
    --txt_db "/data7/emobert/txt_db/movies_v1_th0.0_trn_2000.db/" \
    --img_db "/data7/emobert/img_db/movies_v1" \
    --checkpoint "/data7/emobert/exp/pretrain/movies_v1_uniter_4tasks/ckpt/model_step_100000.pt" \
    --model_config config/uniter-base.json \
    --output_dir /data7/emobert/exp/mmfts/test2000 --fp16 \
    --conf_th 0.0 --batch_size 400