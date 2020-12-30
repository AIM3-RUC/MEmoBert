# 新的下游任务的数据处理流程：
Step1: 抽取每个video中每张脸抽取特征 --lrc
Step2: 转化为 npz 文件
    preprocess/iemocap/process_fts.py
    根据某个CV下面的数据进行构建，如果某个segment没有视觉信息，那么填充为0就可以了。
    比如 soft-label = 只包含一帧当前句子的 one-hot, feat=(1,342)
Step3:创建lmdb库
    code/uniter/scripts/create_imgdb.sh
Step4:构建文本的lmdb库
    build_lmdb/mk_txtdb_by_names.py
    顺便把 label 信息也融合到 txt 的 lmdb 库中。
    example['target'] = label

# 数据部分的地址:
/data7/MEmoBert/emobert/exp/evaluation/MSP
    特征地址(with masked都是错误的)
        denseface_openface_mean_std_movie_no_mask\
        denseface_seetaface_mean_std_movie_no_mask\
    img_db(共享)
        denseface_openface_mean_std_movie_no_mask/img_db
    txt_db
        txt_db/${i}/${setname}.db

# Case1: Extract features
code/uniter/extract_fts.sh
保存抽取的特征以及对应的target, 正好读txt_db 就一块保存了

# Case2: Finetune
code/uniter/train_emo.sh