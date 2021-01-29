# vlbert文档
该工程由 https://github.com/ChenRocks/UNITER 修改

## 环境配置
+ 方案1: 这个docker不行 -- Fail
    拉一个uniter镜像下来，
    docker pull chenrocks/uniter
    https://github.com/ChenRocks/UNITER#quick-start
    然后正常的创建docker.  

+ 方案2: -- OK
    按照dockerfile中的信息直接安装
    docker pull pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime
    docker images
    conda create -n vlbert python=3.6
    编辑.condarc
    channels:
        - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
        - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
        - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
    ssl_verify: true

    https://github.com/ChenRocks/UNITER/blob/master/Dockerfile
    + Bug4: ModuleNotFoundError: No module named 'fused_layer_norm_cuda' 
    https://github.com/NVIDIA/apex/issues/156#issuecomment-465301976
    Apex 的错误，重新安装CUDA版本的
    cd /data8/vl_pretrain/apex
    pip uninstall apex & cd apex
    rm -rf build
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
    + + Bug4.1 https://github.com/NVIDIA/apex/issues/802 

+ 方案3: --No Need
    拉一个 horovod 镜像下来
    https://hub.docker.com/r/horovod/horovod/tags
    docker pull horovod/horovod:0.19.3-tf2.1.0-torch-mxnet1.6.0-py3.6-gpu
-----

## 数据准备
由于训练集合的image太大，无法下载，所以自己抽取fastrcnn特征自己构建。
构建image的lmdb数据集的标准的划分方法是按照mscoco集合图片的划分方法。
mscoco集合一共有多少图片：
    train:82783
    val:40504
    test:40775
mscoco 的 public_split 训练集合一共有多少数据:
    trn_names.py 113287 包含val的部分 = 82783 + 30504
    val_names.py 5000
    tst_names.py 5000
itm finetune 集合划分:
    train:82783
    val:5000
    restval: 30504
    test:5000
official pretrain 集合的划分：
    train: 76136 
    restval: 30467
    val: 5000

## 构建图片信息的LMDB --COCO
Step1: 抽取FastRCNN的特征
    cd vlbert/code/fastrcnn_benchmark/inferences
    bash run.sh
    抽取比较慢，可以指定GPUID 以及 startId 和 endId 来多进程跑。
    Bug:
    修改配置文件，指定图片大小，默认是 1333 * 800 
    对于提示 Too Large 的图片修改为 max=667 min=400, 
    如果还有 Too Large 的图片修改为 max=500 min=300 
另外一个比较快速的抽取特征的方法：
    https://github.com/MILVLG/bottom-up-attention.pytorch/blob/master/configs/bua-caffe/extract-bua-caffe-r101.yaml
Step2: 转化为将 hdf5 文件转化为 npz 文件
    cd vlbert/code/vlbert/preprocess
    python trans2lmdb.py 
    注意修改里面的文件路径
Step3: 按照UNITER的数据格式，将npz文件的集合转化为 LMDB 数据库存储格式.
    cd vlbert/code/uniter/scripts/
    bash create_imgdb.sh
    注意修改里面的文件路径

## 对于新的中文数据图片和文本的LMDB -- Update 12.02
Step1: 抽取fastrcnn的特征，保存作为 hdf5 文件
    注意，不同的数据集，要对应不同的命名规范和处理流程
    cd vlbert/code/fastrcnn_benchmark/inferences
    bash run.sh
    对于大的图片，需要二次处理，采用small的配置，如果 small 还不行，可以直接过滤掉
Step2: 将抽取fastrcnn的特征，保存作为 npz 文件
    同样为不同的数据添加不同的数据处理方式
    cd vlbert/preprocess/
    python trans2lmdb.py
Step3: 构建图片特征的 LMDB 库
    cd vlbert/code/uniter/scripts/
    bash create_imgdb.sh
Step4: 对于中文的数据，首先需要处理中文的模型和字典的问题
    将tf转化为torch模型，然后修改 code/vlbert/scripts/convert_ckpt.py 为uniter的命名

Step5: 根据全部的Caption的数据格式，以及整理好的图片LMDB的json文件，获取文本的LMDB库
    cd code/vlbert/preprocess/
    python mk_txtdb_by_imgs.py
    做中文数据集的时候一定要指定tokenizer的类型 ******** --toker bert-base-chinese

## 关于大规模数据测试
存在问题，比如现在3W的AIC数据进行测试，这时候事先计算所有的图片和所有的文本的相似度，
则会存储一个 3w（图片） * 15w（文本）的矩阵，然而内存无法存储这么大的矩阵，那么需要进行优化。
目前的实现方式根据文本的信息，即文本LMDB中的图片Id和文本Id进行构建索引，所以无法直接进行切片的方式将大测试集分成多个小测试集：
self.txt2img = self.txt_db.txt2img
self.img2txts = self.txt_db.img2txts
self.all_img_ids = list(self.img2txts.keys())
如果只是修改img_ids, 可以做分片做text2img的检索，但是无法进行img2text的检索。
方案1一步到位: 需要修改为每个任务，每个样本单独测试，并可以指定测试集合，比如文搜图任务中，那么使用val中所有图片都是候选集合，
或者在其他测试集合上测试。既可以测试速度，也可以在后续搭建demo也可以用到？
方案2小修小补：优化现在的测试方法，也挺麻烦的，但是比方案1肯定要快一些。
1. 同样需要将目前的两个任务一块测试的代码，进行拆分，拆分成文搜图 和 图搜文 两个单独的测试任务。
2. 对于每个任务，都需要进行切片操作，比如首先进行文搜图的任务，将aic3w测试集合切分成6份, step=5000. 
这样可以三个测试集合的结果进行加权平均就可以了。
3. 保存文搜图任务的结果，然后图搜文任务中直接进行结果的读取即可。

## 注意事项
+ Bug1: pretrain 以及 finetune 中的 output_dir 判断是否存在部分删除 --fix
+ Bug2: 必须要.git 文件才行，佛了
+ Bug3: batch_size 要大于seq_len * 8, 在data/sampler.py:41 不知道为啥要这么设计, 目前增大batch_size可以避免错误, 看看后续有没有什么影响
+ Bug4: 需要手动建立/data3/zjm/vlbert/exp/pretrain/cocoval/ckpt/文件夹 --fix
+ Bug5: 目前vocab-size是 28996 ？ 但是官方的TF版本的bert和Torch版本的Bert都是 30552 个token。
    Bert Base Cased = 28996 而 Bert Base uncased = 30552
    /data1/pretrained_torch_berts/cased_L-12_H-768_A-12/
+ Bug6: vocab  中的 unused1 ～ unused1000 表示啥意思？
    unused1 表示预留字符 可以用来添加自己的新词，添加自己的新词？
    https://github.com/google-research/bert/issues/396
    比如定义 [Start] = 1, [End] = 2 写到 const.py 文件中。
+ Bug7: 报错, 由于数据全0的时候就会报这么错误，就会导致空指针。
    [1,3]<stderr>:    batch.record_stream(torch.cuda.current_stream())
    [1,3]<stderr>:RuntimeError: invalid device pointer: %p0
    检查数据，双流的时候由于itm需要构建负例，需要的数据处理方式不同，而当时修改代码的时候没有注意，导致的错误。
+ Bug8: 报错可能是数据index的溢出导致的
    RuntimeError: Creating MTGP constants failed
    RuntimeError: cuda runtime error (59) : device-side assert triggered at 
    原因是 model.py 中 gather_index 的错误。
    print(torch.cat([txt_emb, img_emb], dim=1).shape) # [400, 71, 768]
    print(torch.max(gather_index)) # 83
    index 超了合并后的embedding的最大值，超过上限了。
    问题就本来是tl, 是错的，修改成了 nbb 也是错的，应该是最大文本长度。
    gather_index = get_gather_index(txt_lens, [nbb]*len(txt_ids),
                                len(txt_ids), max(txt_lens), out_size)

## 修改记录
将双流模型中的 type embeddings 去掉了。
anwen hu 2020/11/22
model/pretrain.py: add forward_caption()
model/layer.py: revise UniterModel.forward(), add judgement of the dim of attention_mask
add model/caption.py 
vlbert/pretrain.py: add build_caption_dataset()

-------
## 项目代码介绍
1. config/  : 配置文件
    任务的配置文件，包括路径设置、任务指定、训练参数配置/
        -- 根据卡的数目修改配置文件, 主要修改batch-size参数，1024/GPU，根据显存占用情况调整.
    网络参数配置是: uniter-base.json
2. data/    : dataloader和构建各个任务的dataset实现
    增加caption任务需要增加新的dataset的实现
3. model/   : 网络结构
    model.py
    pretrain.py 整体的pretrain阶段的流程
4. optim/   : 优化器和学习率调整策略等
5. preprocess/  :   数据预处理
    特征提取、原始数据预处理等
6. scripts/ : 数据下载等
    LMDB特征格式转化、数据下载等
7. utils/   : 工具包
8. pretrain.py 预训练入口文件 (具体使用方法见原始工程readme, README.md)
9. inf_itm.py  retrieval 任务测试 (具体使用方法见原始工程readme)
10. train_itm_hard_negatives.py  ms coco retrieval任务 (具体使用方法见原始工程readme)

## 使用方法
1. 注意, 单流网络和双流网络需要在config文件中添加
```
{   ...
    "model_mode": "two_flow", // two_flow 表示双流, one_flow 表示单流
    ...
}
```
模型config文件 (uniter-base.json) 和实验配置config文件 (pretrain***) 都需要添加.
2. 训练双流网络:
CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 python pretrain.py \
        --config config/pretrain-coco-base-4gpu-2flow.json
3. 训练单流网络:
CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 python pretrain.py \
         --config config/pretrain-coco-base-4gpu.json
4. 特殊字符的标记:
/data8/vlbert/indomain/txt_db/pretrain_coco_train.db/meta.json


## lmdb 读取
>>> import lmdb
>>> env = lmdb.open('./')
>>> txn = env.begin(buffers=True)
>>> import msgpack
>>> from lz4.frame import compress, decompress
>>> msgpack.loads(decompress(txn.get('0'.encode('utf-8'))), raw=False)

## 微调并测试 MS COCO的 Retrival 任务
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 horovodrun -np 6 python train_itm_hard_negatives.py \
        --config config/train-itm-coco-base-8gpu-hn.json
得到 vlbert/exp/finetune/coco_retrieval/ckpt/model_step_5000_final.pt

对于 Caption 数据来说， 图：文 = 1:5
图检索文: 1: len(val_imgs) * 5, coco: 1:25000
文检索图: 1: len(val_imgs), coco: 1:5000

## "max_txt_len" in config
通过设置max_txt_len， 会过滤掉txt_db中文本长度超过max_txt_len的样例。
(max_txt_len=-1表示不进行过滤)
例如，在xyb的预训练中，对于非caption任务，max_txt_len=180, 可以保留>90%的样本进行训练，但在
测试时，max_txt_len=-1，表示所有样本参与评测。