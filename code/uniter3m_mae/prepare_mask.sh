# conda deactivate
set -e
export PYTHONPATH=/data7/MEmoBert

corpus_name='IEMOCAP' # 'MSP'
corpus_name_small='iemocap' # msp  
data_path='/data7/emobert/exp/'

## for a v l input
for mask_ratio in 0.9; do
    for setname in tst
    do
        for cvNo in $(seq 1 10)
        do
            python prepare_mask.py \
                --txt_db $data_path/evaluation/${corpus_name}/txt_db/${cvNo}/${setname}_emowords_sentiword.db \
                --img_db $data_path/evaluation/${corpus_name}/feature/denseface_openface_iemocap_mean_std_torch/img_db/fc/feat_th0.0_max36_min10 \
                --speech_db $data_path/evaluation/${corpus_name}/feature/norm_comparE_db_5mean/feat_th1.0_max64_min10 \
                --use_speech --use_visual --use_text \
                --mask_ratio_text $mask_ratio --mask_ratio_speech $mask_ratio --mask_ratio_visual $mask_ratio \
                --output_dir /root/lrc/data/evaluation/IEMOCAP/mask_asyn_comparE/${cvNo}/${setname}
                # $data_path/evaluation/${corpus_name}/mask_asyn/${cvNo}/${setname}
        done
    done
done

