## For all code, you must
export PYTHONPATH=/data7/MEmoBert

## 数据预处理
<!-- run on avec2230 -->
preprocess/process.py
ffmpeg -ss 00:50:30:329 -t 0:0:1.30 -i /data7/emobert/resources/raw_movies/No0111.Antwone.Fisher.mp4 -c:v libx264 -c:a aac -strict experimental -b:a 180k output.mp4
数据检查，检查文本，语音 以及视觉能否对应: ---问题不大。
----sample1 (mkv)---- ok
video_clips/No0002.The.Godfather/3.mp4 --4s多
audio_clips/No0002.The.Godfather/3.wav  --4.80s
text: I gave her freedom but I taught her never to dishonour her family 
time: 4.7s
----sample2 (mkv)---- ok
video_clips/No0002.The.Godfather/72.mp4 
audio_clips/No0002.The.Godfather/72.wav  --00:00:01.69
text: Don Barzini
time: 1.6s
----sample3 (mkv)---- 语音跟跟字幕不匹配，实际发音是 this is， 应该是下一句内容.
video_clips/No0079.The.Kings.Speech/5.mp4 
audio_clips/No0079.The.Kings.Speech/5.wav  --00:00:01.47
text: Good afternoon.
time: 1.5s
----sample4 (mkv)---- 语音跟跟字幕不匹配，实际发音是"the BBC National Programme and Empire Services." 少了this is.
video_clips/No0079.The.Kings.Speech/6.mp4 
audio_clips/No0079.The.Kings.Speech/6.wav  --00:00:03.20
text: This is the BBC National Programme and Empire Services.
time: 3.2s
----sample5 (mkv)----这个是正确的，说明字幕以及时间大体是对的，可能会有点小误差.
video_clips/No0079.The.Kings.Speech/4.mp4 --
audio_clips/No0079.The.Kings.Speech/4.wav  --00:00:01.34
text: Time to go.
time: 1.3s
----sample6 (mp4)----ok
video_clips/No0088.Legally.Blonde.2.mp4/134.mp4 --
audio_clips/No0088.Legally.Blonde.2.mp4/134.wav --00:00:01.28
text: right on the entry field.
time: 1.4s
----sample7 (mp4)----ok 视频是高清的5.5G，所以切的比较慢
video_clips/No0266_downton_abbey_s06e01/538.mp4 --ok
audio_clips/No0266_downton_abbey_s06e01/538.wav 
text: So this isn't something Lord Merton has persuaded you into?.
time: 3.2s

[Bug]之前的moives-v1的过滤后的有效片段是 53% 共81000条，现在是40.6只有6000条。
需要验证修正后的切割方法是否正确。 --- 通过下面的验证，目前修正后的应该是正确的。
由于之前切割不准确，基本多切了一段，所以时长都增加了，也就增加了 activate-spk 的可能性，这里 vad 跟 face 匹配的时候算的是绝对值，只要有匹配的就矮子里面拔将军。
No0030.About.Time 共 1972 片段， 但是 ActiveSpk 只有 343 条，0.17 的通过率.
原来的切割视频的方法是 ActiveSpk 只有 542 条. grep -vwf  has_active_spk.txt has_active_spk.txt.eror 但是有 263条不同的。随机检查几条进行验证。
Error-video_clips/No0030.About.Time.Error/4.mp4: 647K, 切的不准导致多了前一句; 0，人物没有说话，是镜头外的人在说话。由于多了一句，所以前面有几帧刚好跟vad-match的，所以得分稍微高一点。
Now-video_clips/No0030.About.Time/4.mp4: 129K; None, 人物没有说话，是镜头外的人在说话。
text: There was something solid about her.
Error-video_clips/No0030.About.Time.Error/1960.mp4: 647K; 明显的也是切割错误。
Now-video_clips/No0030.About.Time/1960.mp4: 84k;
text: That's fine.


## 下游任务的数据预处理以及特征抽取流程
MELD EmoList = {0: 'neutral', 1:'surprise', 2: 'fear', 3: 'sadness', 4: 'joy', 5: 'disgust', 6: 'anger'}
    MELD 特征数据: /data7/emobert/exp/evaluation/MELD/feature/denseface_openface_meld_mean_std_torch/img_db/
    MELD 原始图片特征数据，112*112: --pending
    MELD 的文本数据:
    /data7/emobert/exp/evaluation/MELD/txt_db/train_emowords_emotype.db
    /data7/emobert/exp/evaluation/MELD/txt_db/val_emowords_emotype.db
    /data7/emobert/exp/evaluation/MELD/txt_db/test_emowords_emotype.db
    MELD 语音特征数据: --pending

IEMOCAP EmoList = []: 

MSP EmoList = []: 

## 下游任务的原始图片数据的预处理
if dataset_name == 'iemocap':
        mean, std = 131.0754, 47.858177
elif dataset_name == 'msp':
    mean, std = 96.3801, 53.615868
elif dataset_name == 'meld':
    mean, std = 67.61417, 37.89171
else:
    print('the dataset name is error {}'.format(dataset_name))

## 存储位置的优化
不要将所有的小文件存储在data7上, 否则data7小文件太多，导致特别卡.
由于目前frames中间数据太多并且没有用，所以直接删除就可以。
删除进程
kill -s 9 pid
删除文件
mkdir /data1/blank/
rsync --delete-before -d /data1/blank/ *
文件拷贝 -av 支持断点续传
rsync -av --progress --bwlimit=50000  ./No0001.The.Shawshank.Redemption  /data8/emobert/data_nomask_new/frames/

## 初始化预训练的模型
1. pt format, bert-base-uncased, /data7/MEmoBert/emobert/resources/pretrained
2. bin format, pretrained_on_opensub1000w, /data7/emobert/exp/mlm_pretrain/results/opensub/bert_base_uncased_1000w_linear_lr1e4_warm4k_bs256_acc2_4gpu/checkpoint-93980  eval-loss=1.7316
3. bin format, pretrained_on_movies_v1v2v3,  /data7/emobert/exp/mlm_pretrain/results/moviesv1v2v3/bert_base_uncased_2e5/checkpoint-34409 eval-loss=1.8564

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

## tf模型转换torch，采用 bert-base-uncased
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
4. 一个数据集输入的时候是正常的，但是当输入是两个是数据集的时候，会报错。
忘记修改batch-size=120,而是用的10240, 所以batch都是空的。

5. Assertion `srcIndex < srcSelectDimSize` failed.
[1,0]<stdout>:torch.Size([120, 21])
[1,0]<stdout>:torch.Size([120, 25])
[1,0]<stdout>:torch.Size([120, 19])
[1,0]<stdout>:torch.Size([120, 2642]) # 突然蹦出个这么大的数据来。
验证的时候没有去掉长度大于max-len的句子。

6. 当将face的threshold=0.5之后，会有空的数据，就会报错 :RuntimeError: invalid device pointer: %p0
判断是文本和图像都没有导致的错误还是只是单纯的图像没有导致的错误。然后添加相应的处理机制～
[1,0]<stdout>:[Debug empty] txt 9 img 0 的时候会报错，添加异常处理机制～
/data4/MEmoBert/code/uniter/data/emocls.py


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
    bash extract_features.sh 需要配置 split-num 和 给定的gpu信息
Step2: 将抽取的 Denseface 特征进行 segmentId = movie_name + '_' + segment_index 转化为所有的 npz 文件
    build_lmdb/trans2npz_downsteam.py
Step3: 基于npz数据，构建视觉的 LMDB 数据库, img_db
    code/uniter/scripts/create_imgdb.sh
Step4: 基本英文的 bert-base-uncased 模型文本的 LMDB 特征库，txt_db
    build_lmdb/create_txtdb.sh
--- Analyse: 
    cd preprocess/analyse/
    python analyse_filter_strategies.py

Step5: 利用预训练好的模型抽取 Uniter 特征
    code/uniter/extract_fts.sh
Step6: 然后利用下游任务的代码进行训练测试
    code/downstream/run_pretrained_ft.sh

Step7: 抽取语音的 ComparE 特征, 10ms/frame, 130dim.
preprocess/extract_features.py

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

Step5: 进一步优化，backbone 和 cross-transfomer 采用不同的优化器和学习率。
Denseface Backbone: AdamW + learning rate 1e−3 + weight decay 5e−4
Transformer: AdamW + 5e−5 + weight decay 0.01

## UniterBackbone的联合的中断后继续训练.
两个模型的加载是分开的，中断训练的话，学习率的初始化、模型restore是分开的。 
一个最简单的解决方案就是模型加载还是分开加载，在face-backbone加载的时候过滤掉没有用的就好。
如果face-backbone初次使用，那么加载预训练好的 face-exp 模型 或者 lip-reading 模型。
如果是断点继续训练，那么加载的是联合训练的后的模型。

## UniterBackbone的联合的预训练任务问题.
学习视觉任务有三种方式，分别是 MRCKL，MRFR，NCE-Loss. 
由于联合训练之后 MRCKL，MRFR，NCE-LOSS 就都无法使用。
目前考虑的是去掉 ITM 任务，那么就只剩 MLM 和 MELM 任务了。
Frame Order method.

## Uniter3Flow 的联合训练
这种训练方式相比之前UNiter的训练需要更多的训练时间。
1. 多头结构时的CLS和SEP token 要如何设计？
目前的方法是CLS和SEP都加在文本上，type-embeeding 去掉了，然后其他的保持不变。
方案2:做法是去掉文本模态的 [Sep] Token, 每个模态输出之后加上type-embedding, 有type-embedding可以不加[Sep].
使模型任务尽可能的简单。前面的encoder就好好进行特征学习，后面的 cross-encoder 进行任务学习。
2. 文本+face 已经跑通了. -- evaluating

3. 文本+speech -- going
采用抽取好的ComparE的特征，采用的数据处理类型, 跟feature的Img保持一致即可。

4. 文本+speech+face --pending


参考HERO的做法:
Cross-encoder model, may including 2~4 transformer layers.
用于接收三个模态的 encoder 的输出，然后做 Concat + attention-mask(音频需要有降采样，Attention Mask全1)进行拼接.
整理的实现参考HERO的实现，HERO的实现也是基于Uniter的框架，所以参考比较方便一些。
f_config 对应 cross-transformer, 6层，但是采用 robota base 中的6层作为初始化, 1 3 5 7 9 11 层的参数进行初始化。type_vocab_size=1.
c_config 对应 temporal-transformer, 3层，type_vocab_size=2.
首先 文本 input_ids, 不需要加 SEP, wordembeeding + positionembedding + typeembedding(1)
https://github.com/linjieli222/HERO/blob/master/model/embed.py#L12
然后 视觉 input_ids, 不需要加SEP, transformed_im + position_embeddings + type_embeddings(1)
https://github.com/linjieli222/HERO/blob/faaf15d6ccc3aa4accd24643d77d75699e9d7fae/model/encoder.py#L247
文本和视觉的type都是1有点奇怪。
然后经过 Cross-Transformer 输入是 [CLS] + input_ids/img_fts + [SEP] + img_fts/input_ids, HERO 采用的是 concat[img_emb,txt_emb]. Img在前面。
在 Cross-Transformer 层上只有一个MLM的预训练任务。输入是 img + txt 来预测 masked token.
最后过 Temporal-Transformer 的输入是: 直接 Cross-Transformer 的输出，如果需要mask/shuffle的话，则是在 Cross-Transformer 之后做.
https://github.com/linjieli222/HERO/blob/f938515424b5f3249fc1d2e7f0373f64112a6529/model/encoder.py#L287
关于视觉和语音模态是否要加 CLS Token, 可以自定义token进行分类。
https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py#L88

FOM 任务是如何做的？？--Pending
'''

## Uniter3m 的联合训练
语音特征，每5帧取一个平均, 语音部分最大长度是 64. 
语音相关的任务只保留itm.
--done
视觉的特征任务中加入语音的信息, 目的是做多模态，所以三个模态的信息要一块出现比较好。
--done

图像特征和语音特征之间需不需要加 [SEP] 标签？--pending
语音特征要不要添加预训练任务,  尝试加 speech-itm 的任务？

## 训练策略
1. 目前的txt=30, img=64, max-token=10240, 得到的batch-size大约100～120左右。
共有20w训练数据，200,000/100 = 2000 iters. 2000 * 20 = 4w steps..

