CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 horovodrun -np 8 python inf_itm.py \
    --txt_db "/data2/ruc_vl_pretrain/aic/txt_db/itm_partval1000.db" \
    --img_db "/data2/ruc_vl_pretrain/aic/img_db/val/" \
    --checkpoint "/data2/ruc_vl_pretrain/exp/pretrain/aic_cc/ckpt/model_step_81000.pt" \
    --model_config config/uniter-base_zh.json \
    --output_dir /data2/ruc_vl_pretrain/exp/evaluation/retrieval/aic_partval1000/ --fp16 \
    --batch_size 512