# 下游任务.
source activate vlbert
export PYTHONPATH=/data7/MEmoBert

# for iemocap
# for i in `seq 1 10`; do
#   for setname in val tst trn; do
#     python mk_txtdb_by_names_wwm_nsp.py --input /data7/emobert/exp/evaluation/IEMOCAP/refs/${i}/${setname}_ref.json \
#                     --output /data7/emobert/exp/evaluation/IEMOCAP/txt_db/${i}/${setname}_wwm_nrcemolex_prompt_nsp_itwas.db \
#                     --toker bert-base-uncased  --dataset_name iemocap_${setname}  --use_emo --prompt_type nsp_itwas
#   done
# done

# # # ### for msp
# for i in `seq 1 12`; do
#   for setname in val tst trn; do
#     python mk_txtdb_by_names_wwm_nsp.py --input /data7/emobert/exp/evaluation/MSP/refs/${i}/${setname}_ref.json \
#                     --output /data7/emobert/exp/evaluation/MSP/txt_db/${i}/${setname}_wwm_nrcemolex_prompt_nsp_itwas.db \
#                     --toker bert-base-uncased  --dataset_name msp_${setname} --use_emo --prompt_type nsp_itwas
#   done
# done


# for iemocap
for i in `seq 1 10`; do
  for setname in val tst trn; do
    python mk_txtdb_by_names_wwm.py --input /data7/emobert/exp/evaluation/IEMOCAP/refs/${i}/${setname}_ref.json \
                    --output /data7/emobert/exp/evaluation/IEMOCAP/txt_db/${i}/${setname}_wwm_nrcemolex_prompt_pre_mask_itwas.db \
                    --toker bert-base-uncased  --dataset_name iemocap_${setname}  --use_emo --prompt_type mask_itwas
  done
done

# # # ### for msp
for i in `seq 1 12`; do
  for setname in val tst trn; do
    python mk_txtdb_by_names_wwm.py --input /data7/emobert/exp/evaluation/MSP/refs/${i}/${setname}_ref.json \
                    --output /data7/emobert/exp/evaluation/MSP/txt_db/${i}/${setname}_wwm_nrcemolex_prompt_pre_mask_itwas.db \
                    --toker bert-base-uncased  --dataset_name msp_${setname} --use_emo --prompt_type mask_itwas
  done
done