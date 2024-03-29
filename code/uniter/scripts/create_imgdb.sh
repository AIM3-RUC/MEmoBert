# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
set -e
# img_dir=/data7/emobert/affectnet_ft_npzs_nomask/movies_v3/fc
# output_dir=/data7/emobert/affectnet_img_db_nomask/movies_v3/fc
# img_dir=/data7/emobert/rawimg_npzs_nomask/movies_v3
# output_dir=/data7/emobert/rawimg_db_nomask/movies_v3
# img_dir=/data7/emobert/ft_trans2_npzs_nomask/movies_v1
# output_dir=/data7/emobert/img_db_nomask/movies_v1_trans2
# img_dir=/data7/emobert/ft_npzs_nomask/voxceleb2_v2/fc
# output_dir=/data7/emobert/img_db_nomask/voxceleb2_v2/fc
img_dir=/data7/emobert/exp/evaluation/IEMOCAP/feature/denseface_affectnet_openface_iemocap_mean_std_torch/ft_npzs/fc
output_dir=/data7/emobert/exp/evaluation/IEMOCAP/feature/denseface_affectnet_openface_iemocap_mean_std_torch/img_db/fc
# img_dir=/data7/emobert/exp/evaluation/MSP/feature/denseface_openface_msp_mean_std_torch/ft_npzs/trans2
# output_dir=/data7/emobert/exp/evaluation/MSP/feature/denseface_openface_msp_mean_std_torch/img_db/trans2
# img_dir=/data7/emobert/exp/evaluation/MELD/feature/denseface_openface_meld_mean_std_torch/ft_npzs/fc
# output_dir=/data7/emobert/exp/evaluation/MELD/feature/denseface_openface_meld_mean_std_torch/img_db/fc
# img_dir=/data7/emobert/exp/evaluation/MELD/feature/denseface_openface_meld_mean_std_torch/raw_img_npzs
# output_dir=/data7/emobert/exp/evaluation/MELD/feature/denseface_openface_meld_mean_std_torch/raw_img_db
# img_dir=/data7/emobert/exp/evaluation/MSP/feature/openface_iemocap_raw_img/raw_img_npzs
# output_dir=/data7/emobert/exp/evaluation/MSP/feature/openface_iemocap_raw_img/raw_img_db
echo "convrting image features ..."
## for downsteam face task conf_th = 0
python convert_imgdir.py --img_dir $img_dir --output $output_dir --conf_th 0.0 --max_bb 36 --nproc 24
## for movies face task conf_th = 0.5
# python convert_imgdir.py --img_dir $img_dir --output $output_dir --conf_th 0.5 --max_bb 64 --nproc 24
echo "done"