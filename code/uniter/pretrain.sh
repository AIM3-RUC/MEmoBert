export PYTHONPATH=/data4/MEmoBert

# CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python pretrain.py \
#         --config config/pretrain-movies-v1v2-base-2gpu_4tasks.json \
#         --output_dir /data4/emobert/exp/pretrain/nomask_movies_v1v2_uniter_4tasks_lr5e5_bs1024_faceth0.5 \
#         --learning_rate 5e-05 --gradient_accumulation_steps 2 \
#         --train_batch_size 512 

# CUDA_VISIBLE_DEVICES=4 horovodrun -np 1 python pretrain.py \
#         --config config/pretrain-movies-v1v2-base-2gpu_4tasks_melm.json \
#         --melm_prob 0.5 --model_config config/uniter-base-emoword_multitask.json \
#         --output_dir /data4/emobert/exp/pretrain/nomask_movies_v1v2_uniter_4tasks_lr5e5_bs1024_faceth0.5_mlt-melm5 \
#         --learning_rate 5e-05 --gradient_accumulation_steps 2 \
#         --train_batch_size 512 

# CUDA_VISIBLE_DEVICES=1 horovodrun -np 1 python pretrain.py \
#         --config config/pretrain-movies-v1v2-base-2gpu_3tasks.json \
#         --checkpoint /data4/emobert/resources/pretrained/uniter-base-uncased-init.pt \
#         --output_dir /data4/emobert/exp/pretrain/nomask_movies_v1v2_uniter_3tasks_lr5e5_bs1024_faceth0.5 \
#         --learning_rate 5e-05 --gradient_accumulation_steps 2 \
#         --train_batch_size 512 

CUDA_VISIBLE_DEVICES=5 horovodrun -np 1 python pretrain.py \
        --config config/pretrain-movies-v1v2-base-2gpu_3tasks_melm.json \
        --melm_prob 0.5 --model_config config/uniter-base-emoword_multitask.json \
        --output_dir /data4/emobert/exp/pretrain/nomask_movies_v1v2_uniter_3tasks_lr5e5_bs1024_faceth0.5_mlt-melm5 \
        --learning_rate 5e-05 --gradient_accumulation_steps 2 \
        --train_batch_size 512 

# CUDA_VISIBLE_DEVICES=2 horovodrun -np 1 python pretrain.py \
#         --config config/pretrain-movies-v1v2-base-2gpu_2tasks.json \
#         --checkpoint /data4/emobert/resources/pretrained/uniter-base-uncased-init.pt \
#         --output_dir /data4/emobert/exp/pretrain/nomask_movies_v1v2_uniter_2tasks_lr5e5_bs1024_faceth0.5 \
#         --learning_rate 5e-05 --gradient_accumulation_steps 2 \
#         --train_batch_size 512 
