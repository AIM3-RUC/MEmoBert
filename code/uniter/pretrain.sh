export PYTHONPATH=/data7/MEmoBert

# CUDA_VISIBLE_DEVICES=6 horovodrun -np 1 python pretrain.py \
#         --config config/pretrain-movies-v1-base-2gpu_melm.json

CUDA_VISIBLE_DEVICES=5 horovodrun -np 1 python pretrain.py \
        --config config/pretrain-movies-v1-base-2gpu_mlm_mrfr_mrckl_melm.json

# CUDA_VISIBLE_DEVICES=2,3 horovodrun -np 2 python pretrain.py \
#         --config config/pretrain-movies-v1-base-2gpu_mlm_mrfr.json

# CUDA_VISIBLE_DEVICES=4,5 horovodrun -np 2 python pretrain.py \
#         --config config/pretrain-movies-v1-base-2gpu_mlm_mrckl.json

# CUDA_VISIBLE_DEVICES=6,7, horovodrun -np 2 python pretrain.py \
#         --config config/pretrain-movies-v1-base-2gpu_mlm_mrfr_mrckl.json