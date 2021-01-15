## For all code, you must
export PYTHONPATH=/data7/MEmoBert

## 预训练的模型
MLM+ITM+MRF+MRC-KL四个预训练任务:
    /data7/emobert/exp/pretrain/nomask_movies_v1_uniter_4tasks/ckpt/model_step_100000.pt

## 构建对于 LMDB 特征数据库
Step1 将抽取的 Denseface 特征进行 segmentId = movie_name + '_' + segment_index 转化为所有的 npz 文件
    build_lmdb/trans2npz.py
Step2 基于npz数据，构建视觉的 LMDB 数据库
    code/uniter/scripts/create_imgdb.sh
Step3 基本英文的 bert-base-uncased 模型构建 txt_db
    build_lmdb/mk_txtdb_by_faces.py
---Manual Check OK

## 模型转换，采用 bert-base-uncased
code/uniter/scripts/convert_ckpt.py

## Bugs
1. Val 和 Test 的loss不下降的问题

## 修改记录
1. 由于Faces之间也是有顺序的，所以需要进行 name2nbb 简单的取多少个，而是应该根据阈值过滤相应位置的数据, 重写数据获取的代码,
不用，因为构建img-db的时候已经过滤了，所有 img2nbb的个数跟保存的特征是一致的.
2. 同样由于Faces之间是连续的，所以需要图片应该也要有 position 的概念, 加上 position embedding.

## Finetune
1. Finetune 采用两种不同的分类器，一种是uniter的vqa任务的分类器，FC + GELU + LayerNorm + FC， 
第二种是Bert本身的 paraphrase 任务中Finetune任务中分类器 drop+fc,  称为 emocls.
2. Fintune 中更新的层数的设置, uniter 的 下游任务 和 bert 的下游任务 都是全部 finetune 的, 需要调小 batch-szie=64.
3. Finetune 的 emocls 分类器中的 drop=0.1, 即 keep-prob=0.9

## 直接抽取特征
Step1: 对下游任务数据抽取面 Denseface 部表情特征, 用各自任务的均值和方差。-- lrc
Step2: 将抽取的 Denseface 特征进行 segmentId = movie_name + '_' + segment_index 转化为所有的 npz 文件
    build_lmdb/trans2npz.py
Step3: 基于npz数据，构建视觉的 LMDB 数据库, img_db
    code/uniter/scripts/create_imgdb.sh
Step4: 基本英文的 bert-base-uncased 模型文本的 LMDB 特征库，txt_db
    build_lmdb/mk_txtdb_by_faces.py
Step5: 利用预训练好的模型抽取 Uniter 特征
    code/uniter/extract_fts.sh
Step6: 然后利用下游任务的代码进行训练测试
    code/downstream/run_pretrained_ft.sh