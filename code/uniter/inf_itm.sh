CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 horovodrun -np 8 python inf_itm.py \
    --txt_db "/data2/ruc_vl_pretrain/indomain/txt_db/itm_coco_partval.db" \
    --img_db "/data2/ruc_vl_pretrain/indomain/img_db/coco_val2014/" \
    --checkpoint "/data2/ruc_vl_pretrain/exp/pretrain/coco_caption_gen/ckpt/model_step_4000.pt" \
    --model_config config/uniter-base.json \
    --output_dir /data2/ruc_vl_pretrain/exp/evaluation/retrieval/coco_partval_caption/ --fp16 \
    --batch_size 400
