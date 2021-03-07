export PYTHONPATH=/data7/MEmoBert

# CUDA_VISIBLE_DEVICES=3,4 horovodrun -np 2 python pretrain.py \
#         --config config/pretrain-movies-v1v2-base-2gpu_rawimg_2optim_res.json

# CUDA_VISIBLE_DEVICES=1 horovodrun -np 1 python pretrain.py \
#         --config config/pretrain-movies-v1v2-base-2gpu_rawimg_2optim_res.json
#         --output_dir /data7/emobert/exp/pretrain/rawimg_nomask_movies_v1v2_uniter_2tasks_2optim_5e5_1e3_ferinit_res_fix2_reinitlr_fix0 \
#         --checkpoint /data7/MEmoBert/emobert/exp/pretrain/rawimg_nomask_movies_v1v2_uniter_2tasks_2optim_5e5_1e3_ferinit_res_fix2/ckpt/model_step_100000.pt \
#         --checkpoint_step 100000 --is_reinit_lr --warmup_steps 0

# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python pretrain.py \
#         --config config/pretrain-movies-v1v2-base-2gpu_rawimg_2optim_res_onlymlm.json \
#         --output_dir /data7/emobert/exp/pretrain/rawimg_nomask_movies_v1v2_uniter_onlymlm_2optim_5e5_1e3_ferinit_res \
#         --learning_rate 5e-05 --backbone_learning_rate 1e-03

CUDA_VISIBLE_DEVICES=6,7 horovodrun -np 2 python pretrain.py \
        --config config/pretrain-movies-v1v2-base-2gpu_rawimg_2optim_res_onlymlm.json \
        --output_dir /data7/emobert/exp/pretrain/rawimg_nomask_movies_v1v2_uniter_onlymlm_2optim_5e5_1e3_initscratch_res \
        --learning_rate 5e-05 --backbone_learning_rate 1e-03