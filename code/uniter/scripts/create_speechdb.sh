# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# 语音的处理保持跟
set -e
# img_dir=/data7/emobert/norm_comparE_npzs/movies_v2_5mean
# output_dir=/data7/emobert/norm_comparE_db/movies_v2_5mean
# img_dir=/data7/emobert/wav2vec_feature_npzs/movies_v3_rawwav
# output_dir=/data7/emobert/wav2vec_db/movies_v3_rawwav
# img_dir=/data7/emobert/wav2vec_feature_npzs/movies_v3_cnn_3mean/
# output_dir=/data7/emobert/wav2vec_db/movies_v3_cnn_3mean
# img_dir=/data7/emobert/wav2vec_feature_npzs/voxceleb2_v2_3mean
# output_dir=/data7/emobert/wav2vec_db/voxceleb2_v2_3mean
# img_dir=/data7/emobert/exp/evaluation/IEMOCAP/feature/norm_comparE_npzs_5mean/
# output_dir=/data7/emobert/exp/evaluation/IEMOCAP/feature/norm_comparE_db_5mean/
# img_dir=/data7/emobert/exp/evaluation/IEMOCAP/feature/wav2vec_npzs_3mean/
# output_dir=/data7/emobert/exp/evaluation/IEMOCAP/feature/wav2vec_db_3mean/
# img_dir=/data7/emobert/exp/evaluation/IEMOCAP/feature/wav2vec_cnn_npzs_3mean/
# output_dir=/data7/emobert/exp/evaluation/IEMOCAP/feature/wav2vec_cnn_db_3mean/
# img_dir=/data7/emobert/exp/evaluation/IEMOCAP/feature/wav2vec_rawwav_npzs/
# output_dir=/data7/emobert/exp/evaluation/IEMOCAP/feature/wav2vec_rawwav_db/
# img_dir=/data7/emobert/exp/evaluation/MSP/feature/norm_comparE_npzs_5mean/
# output_dir=/data7/emobert/exp/evaluation/MSP/feature/norm_comparE_db_5mean/
# img_dir=/data7/emobert/exp/evaluation/MSP/feature/wav2vec_npzs_3mean/
# output_dir=/data7/emobert/exp/evaluation/MSP/feature/wav2vec_db_3mean/
# img_dir=/data7/emobert/exp/evaluation/MSP/feature/wav2vec_rawwav_npzs/
# output_dir=/data7/emobert/exp/evaluation/MSP/feature/wav2vec_rawwav_db/
img_dir=/data7/emobert/exp/evaluation/MSP/feature/wav2vec_cnn_npzs_3mean/
output_dir=/data7/emobert/exp/evaluation/MSP/feature/wav2vec_cnn_db_3mean/
# img_dir=/data7/emobert/exp/evaluation/MELD/feature/norm_comparE_npzs_5mean/
# output_dir=/data7/emobert/exp/evaluation/MELD/feature/norm_comparE_db_5mean/
# img_dir=/data7/emobert/exp/evaluation/MELD/feature/wav2vec_asr_npzs_3mean/
# output_dir=/data7/emobert/exp/evaluation/MELD/feature/wav2vec_asr_db_3mean/
# img_dir=/data7/emobert/exp/evaluation/MELD/feature/wav2vec_rawwav_npzs/
# output_dir=/data7/emobert/exp/evaluation/MELD/feature/wav2vec_rawwav_db/
echo "convrting speech features ..."
# if use rawwav, then use the 16000 * 6s = 96000
python convert_imgdir.py --img_dir $img_dir --output $output_dir --conf_th 1.0 --max_bb 64 --nproc 40
echo "done"
