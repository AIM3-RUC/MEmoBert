# conda deactivate
set -e
export PYTHONPATH=/root/lrc/MEmoBert
gpu_id=$1

corpus_name='IEMOCAP' # 'MSP'
corpus_name_small='iemocap' # msp  
data_path=/root/lrc/data
exp_name=kd-mmd-from-pretrained-v1v2v3-2gpu-lr5e-5 # kd-from-pretrained-v1v2v3-2gpu
checkpoint_dir=/root/lrc/mm_contrast/result/$exp_name
model_step=30000

## for a v l input
for setname in tst val trn
do
    for modality in avl av al vl a v l 
    do
        modality_cmd=""
        if [[ $modality =~ "a" ]];then
            modality_cmd="--use_speech $modality_cmd"
        fi
        if [[ $modality =~ "v" ]];then
            modality_cmd="--use_visual $modality_cmd"
        fi
        if [[ $modality =~ "l" ]];then
            modality_cmd="--use_text $modality_cmd"
        fi
        for cvNo in $(seq 1 10)
            do
                CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python extract_cls_fts.py \
                        --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
                        --txt_db $data_path/evaluation/${corpus_name}/txt_db/${cvNo}/${setname}_wwm_nrcemolex_prompt_mask_iam.db \
                        --img_db $data_path/evaluation/${corpus_name}/feature/denseface_openface_iemocap_mean_std_torch/img_db/fc/feat_th0.0_max36_min10 \
                        --speech_db $data_path/evaluation/${corpus_name}/feature/wav2vec_db_3mean/feat_th1.0_max64_min10 \
                        --checkpoint  $checkpoint_dir/ckpt/model_step_${model_step}.pt \
                        --output_dir $data_path/uniter3m_cls_fts/${corpus_name_small}/$exp_name/$modality/${cvNo}/${setname} \
                        --fp16 --batch_size 100 $modality_cmd
            done
    done
done
