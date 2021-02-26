## For all code, you must
export PYTHONPATH=/data7/MEmoBert

## 数据预处理
<!-- run on avec2230 -->
preprocess/process.py

## 预训练的模型
MLM+ITM+MRF+MRC-KL四个预训练任务:
    /data7/emobert/exp/pretrain/nomask_movies_v1_uniter_4tasks/ckpt/model_step_100000.pt

## 构建对于 LMDB 特征数据库
Step1 将抽取的 Denseface 特征进行 segmentId = movie_name + '_' + segment_index 转化为所有的 npz 文件
    build_lmdb/trans2npz.py
Step2 基于npz数据，构建视觉的 LMDB 数据库
    code/uniter/scripts/create_imgdb.sh
Step3 基本英文的 bert-base-uncased 模型构建 txt_db
    build_lmdb/generate_captions.py
    build_lmdb/mk_txtdb_by_faces.py
---Manual Check OK

/data7/emobert/img_db_nomask
movies_v1 tf-version
movies_v1_torch torch-version
movies_v1_trans1 torch-version
movies_v1_trans2 torch-version
movies_v2 torch-version
movies_v2_trans torch-version
movies_v2_trans torch-version

## 模型转换，采用 bert-base-uncased
code/uniter/scripts/convert_ckpt.py


## Bugs
1. Val 和 Test 的loss不下降的问题
2. numpy.savez_compressed 保存的时候，报错。Segmentation fault (core dumped)
    升级numpy没有作用, 换成 numpy.savez() 就可以了
3. 两个优化器之后，训练中出现了
 11%|#1        | 34081/300000 [9:57:53<76:49:37,  1.04s/it]
[1,0]<stdout>:Warning: NaN or Inf found in input tensor.
[1,0]<stdout>:Warning: NaN or Inf found in input tensor.
[1,0]<stdout>:Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 4096.0
[1,1]<stdout>:Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 4096.0
[1,0]<stdout>:Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 4096.0
[1,1]<stdout>:Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 4096.0
https://github.com/NVIDIA/apex/issues/318

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
    build_lmdb/trans2npz_downsteam.py
Step3: 基于npz数据，构建视觉的 LMDB 数据库, img_db
    code/uniter/scripts/create_imgdb.sh
Step4: 基本英文的 bert-base-uncased 模型文本的 LMDB 特征库，txt_db
    build_lmdb/create_txtdb.sh
Step5: 利用预训练好的模型抽取 Uniter 特征
    code/uniter/extract_fts.sh
Step6: 然后利用下游任务的代码进行训练测试
    code/downstream/run_pretrained_ft.sh

## UniterBackbone
step1: 联合人脸视觉的Encoder联合训练，由于数据要传原始的图像，只需要修改imgdb的信息，把feature的信息替换为图像原始信息. --done
    cd /data7/MEmoBert/build_lmdb
    python mk_rawimg_db.py
    然后
    code/uniter/scripts/
    bash convert_imgdir.py

step2: 修改配置文件，采用原始 4tasks + densenet 作为初始化 / from scratch
    face_checkpoint='/data7/MEmoBert/emobert/exp/face_model/densenet100_adam0.001_0.0'
    face_from_scratch=False
    配置文件：code/uniterbackbone/config/uniter-base-backbone.json
step3: 修改dataset和dataloader --OK
step4: 由于加入了backbone的联合训练，所以目前的显存明显不够。修改了：
    1. code/uniterbackbone/data/sampler.py 中的 size_multiple=8 -> 4 
    2. code/denseface/config/conf_fer.py 中的 frozen_dense_blocks=1 或者 2
    3. code/uniterbackbone/config/pretrain-movies-v1-base-2gpu_rawimg.json 中的 batch-size=800
目前的batch-size很小，所以增加迭代次数到 200000 次，看看 itm 的性能是否有提升。

Step5: 进一步优化，backbone 和 cross-transfomer采用不同的优化器和学习率。
Denseface Backbone: SGD + learning rate 1e−2 + weight decay 5e−4
Transformer: AdamW + 5e−4 + weight decay 0.01