# 根据人脸特征数据来构建文本数据，一个文本对应一段video.
export PYTHONPATH=/data7/MEmoBert

# # for iemocap
# for i in `seq 2 10`; do
#   for setname in val tst trn; do
#     python mk_txtdb_by_names.py --input /data7/emobert/exp/evaluation/IEMOCAP/refs/${i}/${setname}_ref.json \
#                     --output /data7/emobert/exp/evaluation/IEMOCAP/txt_db/${i}/${setname}.db \
#                     --toker bert-base-uncased  --dataset_name ${setname}
#   done
# done

# for msp
for i in `seq 1 12`; do
  for setname in val tst trn; do
    python mk_txtdb_by_names.py --input /data7/emobert/exp/evaluation/MSP-IMPROV/refs/${i}/${setname}_ref.json \
                    --output /data7/emobert/exp/evaluation/MSP-IMPROV/txt_db/${i}/${setname}.db \
                    --toker bert-base-uncased  --dataset_name ${setname}
  done
done