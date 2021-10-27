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
    MELD 语音特征数据: --down


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

## Uniter3Flow 的联合训练
这种训练方式相比之前UNiter的训练需要更多的训练时间。
1. 多流结构的 CLS 和 SEP token 要如何设计？
每个分支去掉 CLS-token, 语音和视觉分支的输出可以加 type-embeeding 区分不同的模态.
2. 关于单模态的部分，可以采用第一个位置进行finetune.
3. 对于每个branch的更新设置单独的参数，多阶段训练.
4. 先以文本和语音两个模态作为baseline.
cross-encoder 的层数目前采用 2 / 4 层.

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
'''

## Uniter3m 的联合训练
语音特征，每3帧取一个平均, 语音部分最大长度是 64. 
语音相关的任务只保留itm.
--done
视觉的特征任务中加入语音的信息, 目的是做多模态，所以三个模态的信息要一块出现比较好。
--done
图像特征和语音特征之间需不需要加 [SEP] 标签？ 没有加，但是不同的type.
语音特征要不要添加预训练任务,  尝试加 speech-itm 的任务, 加了没啥用，越加效果越差。

## Uniter3m + SentiLARE 论文中两个预训练任务
Early Fusion of Emo LARE. 没有句子级别的情感分类层, 
Late Supervised 在EF的基础上加了一个句子分类层,
目前一个batch里面既有EF又有LS, 但是格式是一样的，所以二者几乎是一样的。
当输入的utt-category是unknown的时候类别是什么呢？ 此时的label全是-1，所以没有loss.
所以就这个任务其实可以控制 task-ratio 这个参数进行控制 EF 和 LS 的比例。

## 训练策略
1. 目前的txt=30, img=64, max-token=10240, 得到的batch-size大约100～120左右。
共有20w训练数据，200,000/100 = 2000 iters. 2000 * 20 = 4w steps..

## 高质量的情感样本筛选 --Done
在EmoClsTask中，只把那些质量比较高的数据，并且保持类别均衡，拿来出做EmoCls。
挑选策略，如果之后一个词，那么去掉；如果类别是Neutral，那么要求概率大约 pneu=80%； 如果类别是其他的情感类别，那么概率大于 pohter=40% 才可以。
以上两个阈值，可以根据最终的数据分布进行确定。 pohter=30% 35% 40% pneu=70% 75% 80%
/data7/MEmoBert/build_lmdb/modify_text_db.py
BertMovies的数据都比较平衡，但是不符合常理啊，数据集中怎么可能分布那么均衡呢？


## UNIMO 单模态训练 --Done
可以同时利用单模态，或者任意模态的组合进行训练，--use_visual --use_speech 用来初始化模型。
而构建db的时候不能根据 --use_visual 来进行判断，而是config中每个db模态信息是否存在.

1. 可以直接加入有标注的文本数据单独做 MLM and EmoCls 的任务。--Done
2. 比如利用 opensubtitle 的数据进行文本的训练, MLM 情感分类模型 和 EmoCLs 情感分类.
整理基于 opensubtitle 500w 的文本数据，构建均衡的文本分类模型
需要确认模型，采用几分类的模型，目前先选择 5分类的。 
最后确认模型选择综合几个测试集合的结果，最终选择的模型是:
/data7/emobert/exp/text_emo_model/all_3corpus_emo5_bert_base_lr2e-5_bs32_debug/ckpt/epoch-1
将1000w的数据分成4份，并只用长度大于等于3的：
text_filepath = '/data7/emobert/exp/mlm_pretrain/datasets/OpenSubtitlesV2018/OpenSubtitles.en-zh_cn.en_trn1000w.txt'
/data7/emobert/exp/mlm_pretrain/datasets/OpenSubtitlesV2018/opensub1000w_p1.csv
/data7/emobert/exp/mlm_pretrain/datasets/OpenSubtitlesV2018/opensub1000w_p2.csv
/data7/emobert/exp/mlm_pretrain/datasets/OpenSubtitlesV2018/opensub1000w_p3.csv
/data7/emobert/exp/mlm_pretrain/datasets/OpenSubtitlesV2018/opensub1000w_p4.csv
分别构建txtdb，然后用模型提取weak-label. 
预训练的时候的EmoCls单独做一个数据集合（只用质量高的数据做）neu 概率大于 80% 其他类别的概率大于 40%.

### 可以提升模型对模态适应能力，加入单独的单模态和双模态的结果，测试不同模态组合情感下的结果 --Done
加入单独的 A V L AV AL LV 的训练 --Done
taskname: 
    mspansrfrnotext
    mspanrfrnotext
    mspanrcklnotext

## 重新加入EmoWords的训练
https://d3smihljt9218e.cloudfront.net/lecture/26027/slideshow/513b2b4eb1663f1d965efce4a34d97b6.pdf
ACL 2021 paper entitled: "eMLM: A New Pre-training Objective for Emotion Related Tasks".
采用的词典是： http://saifmohammad.com/WebPages/lexicons.html
[anger, joy, sadness, surprise, fear, disgust] 大约共 6000 词
首先 emo-word 是 50% 的概率进行 mask.
然后 nonemo-word 的概率是 max(句子长度*15% - emoword个数*50%, 0) /  (句子长度 - emoword个数)
比如句子长度为10个word, 情感词是2个，那么 max(10*15% - 2*50%, 0) /  (句子长度 - emoword个数) = 0.5 / 8 = 0.0625概率。


## 更强的文本情感分类模型
/data6/lrc/EmotionXED/combined 5分类。
BertMovie模型的结果是 69.4%  其中有200多数据是重复的，另外里面还有类别6的，重新评测一下。
BertMovie模型评测根据 preprocess/get_text_weak_label.py
on /data6/lrc/EmotionXED/combined/val_e5.tsv Or tst_e5.tsv
验证集合: 'acc': 0.6721714687280393, 'wuar': 0.6721714687280393, 'wf1': 0.6743000025873979, 'uwf1': 0.6270778807276054}
测试集合: 'acc': 0.6788807461692206, 'wuar': 0.6788807461692206, 'wf1': 0.6830066081593439, 'uwf1': 0.6211400386303344}
on /data7/emobert/text_emo_corpus/all_3corpus/emo5_bert_data/
验证集合: 'acc': 0.6033959975742874, 'wuar': 0.6033959975742874, 'wf1': 0.644581201529892, 'uwf1': 0.42027357887138084,
训练集合: 'acc': 0.6565843167530605, 'wuar': 0.6565843167530605, 'wf1': 0.6608707380083497, 'uwf1': 0.619811166882872,


采用其他三个数据集和之前EmotionXED的训练集合合并，然后进行评测，相当于训练的时候只增加了另外三个数据集的数据。
模型: all_3corpus_emo5_bert_base_lr2e-5_bs32_debug/ckpt/epoch-1
on /data6/lrc/EmotionXED/combined/val_e5.tsv Or tst_e5.tsv
验证集合:  'acc': 0.6598735066760365, 'wuar': 0.6598735066760365, 'wf1': 0.645875864949761, 'uwf1': 0.5875428120722952
测试集合:  'acc': 0.6743837441705529, 'wuar': 0.6743837441705529, 'wf1': 0.6633638279338312, 'uwf1': 0.5947985825668611
on /data7/emobert/text_emo_corpus/all_3corpus/emo5_bert_data/
验证集合: Epoch 1: {'total': 11543, 'acc': 0.7538768084553409, 'wuar': 0.7538768084553409, 'wf1': 0.7669220015312301, 'uwf1': 0.5655239452375367}
训练集合: Epoch 1: {'total': 72536, 'acc': 0.8292296239108856, 'wuar': 0.8292296239108856, 'wf1': 0.8279270478979138, 'uwf1': 0.7923723758424639}

模型: all_5corpus_emo5_bert_base_lr2e-5_bs32_debug/ckpt/epoch-4
on /data6/lrc/EmotionXED/combined/val_e5.tsv Or tst_e5.tsv
验证集合: {'acc': 0.8323963457484188, 'wuar': 0.8323963457484188, 'wf1': 0.8285462423426776, 'uwf1': 0.8029856423823029}
测试集合: {'acc': 0.7836442371752165, 'wuar': 0.7836442371752165, 'wf1': 0.7812776363501978, 'uwf1': 0.7441933520355383}
on /data7/emobert/text_emo_corpus/all_3corpus/emo5_bert_data/
验证集合: Epoch 0: 'acc': 0.6325911808022178, 'wuar': 0.6325911808022178, 'wf1': 0.6610797416100563, 'uwf1': 0.49062888564098844
训练集合: Epoch 0: 'acc': 0.7212970111392963, 'wuar': 0.7212970111392963, 'wf1': 0.7205095503713067, 'uwf1': 0.6772995803609284,


## 在论文MOCKINGJAY中提到语音用 sinusoidal 编码 --Done
sinusoidal positional encoding instead of learnable positional embeddings because acoustic features can be arbitrarily long with high variance.
在三模态的baseline上进行测试。
本文保持原有的编码方式不变，语音和人脸采用位置 Sinusoid 编码方式。 试试吧，感觉用处不大。

## 语音和视觉的mask机制修改为中间的 Span 的Mask. --Done
由于视觉和音频信息的local smoothness特性，需要改成 Span Mask.
参考: https://github.com/andi611/Mockingjay-Speech-Representation/blob/9377bf2585c020b4d217b35f0d27963eb45274ef/utility/mam.py#L92
code/uniter/data/mrm.py _get_consecutive_img_mask()
语音方面，这里采用的 transformer的输出，而不是 cnn 部分的输出，这里采用 span 的作用可能不大，先试试吧。

这里有两种策略：
1. Mockingjay 中采用跟 Bert 一样的策略，15% 的概率选中，然后80%的概率遮蔽并预测，10%的概率随机替换，其他的输入保持不变，即没有Mask操作，
为了验证时刻全序列输入，避免Train和Infer的Gap问题。
2. 很简单的一种方式, 在原来的mrfr的基础，只是将随机帧的Mask修改为中间 span frames 的 mask.

还是直接一步到位，采用方案1吧。 视觉和语音都可以直接用。

## 将Word的Mask机制，修改为 Whole Word Mask 的方法 --Done
生成数据，[word-tokens, word-tokens]。
build_lmdb/create_txtdb_wwm.sh
mlm_wwm pretrain-task

## FrameOrderTask 也该加进来了 --Done
vfom & sfom,  pretrain-task
参考HERO的做法:
Cross-encoder model, may including 2~4 transformer layers.
用于接收三个模态的 encoder 的输出，然后做 Concat + attention-mask(音频需要有降采样，Attention Mask全1)进行拼接.
整理的实现参考HERO的实现，HERO的实现也是基于Uniter的框架，所以参考比较方便一些。
f_config 对应 cross-transformer, 6层，但是采用 robota base 中的6层作为初始化, 1 3 5 7 9 11 层的参数进行初始化。type_vocab_size=1.
首先是视觉的信息经过跨模态的特征编码层，进行编码。
    self.f_encoder = CrossModalTrm(config.f_config, vfeat_dim, max_frm_seq_len)
    CrossModalTrm 中包含 compute_img_embedding 并且经过了一个 CrossModalTrm 的编码
然后经过进行shuffle, 然后送入 c_encoder, 然后预测真实的序列的Index.
而在Uniter结构要做的话，没有单独的模态Encoder, 只能在输入Uniter之前进行shuffle.
FOM中任务的理解:
output_order = shuffled_order: [4, 7, 2, 3, 8, 5, 6, 0, 1]
按照shuffle order将输入的特征序列按照以上的顺序重新组合，所以此时0号位置应该是放time=4的特征，1号位置是time=7的特征
那 output_target 和 output_order 的对应关系是什么？
time=0的特征在7号位置上，time=1的特征在8号位置上。 预测真实的时刻的特征所在的位置。
output_target: [7, 8, -1, -1, 0, -1, -1, 1, 4]


## 语音和视觉采用更多的特征 --Done
wav2vec2.0 CNN 的输出。 以及 densenet2.0 中间层的输出。


## Explore different ITM tasks
目前的ITM任务是以文本最强的模态作为anchor, 然后随机选其他样本的 视觉和语音 作为负例。
1. 存在的问题是视觉和语音作为一个整体出现的，
2. 还是相当于跨模态的对比，没有用到融合/协同的方法。

### Method1: 分解构建负面例子的时候替换其中的一个或者多个模态是构建负面例子. --Done
Case1: 随机替换其中的语音模态
Case2: 随机替换其中的视觉模态
Case3: 随机替换其中的语音和视觉模态
Case4: 随机替换其中的文本模态 -- 也就是文本和语音不同跟Case3是一样的。
code/uniter3m/data/onemodalnegitm.py
neg_sample_p = 0.5 
neg_sample_p = 0.9

### Method2: Contrastive+HardNegative . -- Done
目前的做法是 每个正面例子构建150个负例(由batch-szie大小决定)
itm_neg_samples = 150



## Contrastive Learning For multimodal Fusion --Going
参考构建更加challenging负例进行
P4Contrast: Contrastive Learning with Pairs of Point-Pixel Pairs for RGB-D Scene Understanding.
Contrastive Multimodal Fusion with TupleInfoNCE. 2021
目前的ITM任务是 Cross-modality 的预训练任务，只要用于模态检索。
构建Contrastive Learning, 参考并对比 hard-negative itm.

## <分析1> 加入 EmoCls 任务会带来性能的下降？（EmoCls任务探究 Sheet）
根据实验结果进行总结，在什么情况下文本的EmoCls会带来提升？ 在什么情况下会带来下降？
---- 感觉EmoCls任务和某些预训练任务会产生冲突。
在LV LA 两个模态的相关实验中，加入 EmoCls 任务一般都会比不加要好一点。
但是在 LVA 三模态的实验中，加入 EmoCls 很多情况下效果反而变差。
Note: 问题的原因可能不在EmoCls，可能在三个模态融合的时候时候还存在其他的问题？ 见<分析3>

## <分析2.1> 在纯文本端的加强，对于多个模态的影响？（单模态输入加强 Sheet） --Done

## <分析2.2> 在纯语音端的加强（单模态输入加强 Sheet）--Done
将 VoxCeleb2 的语音和文本输入加入训练.

## <分析3> 三个模态和两个模态同时使用需要注意什么，很多两个模态work的结论，放到三个模态的情况下反而不work.  --Going
一些其他的方法在 LA LV 中work，但是在 AVL 中就不work.
在 AVL 三个模态的情况下, 构建 LA 和 LV 模态输入模型的情况.


## <分析4> 多模态效果的Upper-Bound 在哪？ -- Done.
将测试集合加入训练，然后看看在测试集合上的效果。 基本都能到100%.
添加 inference-emo 阶段的代码，加载模型直接测试，不用再训练了，也方便进行half-set或者部分缺失模态的测评。
训练的时候同时使用 trn + val + tst 进行训练，为啥差异会这么大，数据集合并部分代码有问题？[Bug]
果然是，训练的时候只用到了训练集合的数据，并没有用到验证集的数据。
配置文件中的必须要 assert len(opts.train_txt_dbs) == len(opts.train_img_dbs) == len(opts.train_speech_dbs)
否则会zip函数会根据长度短的列表进行对齐，会导致数据缺失.
训练的时候必须要 max_txt_len，否则最大长度就是30，损失好多数据呢 [Bug], 目前都修改了最大长度是 120.

## <分析4.1> 纯文本的Upper-Bound 在哪？ -- Done.
将测试集合加入训练集合训练，并进行测试，看看结果, 结果也很好，都是epoch=4/6的结果，当epoch=0的时候结果70不到。
IEMOCAP: UA=0.95155
MSP: UA=0.97278

## <分析5> ITM 任务到底需要不需要？ --Done
经过一些实验发现，目前的ITM任务去掉，效果反而更好。但是应该还有改进的空间。

## <分析6> 采用prompt的mask机制尝试一下 --Done
[Bug] 老版本的 pytorch_pretrained_bert.BertTokenizer 不能获取 [MASK] 整个词。
toker = AutoTokenizer.from_pretrained('/data2/zjm/tools/LMs/bert_base_en')
toker2 = BertTokenizer.from_pretrained('/data2/zjm/tools/LMs/bert_base_en')
>>> toker2.tokenize('I am [MASK].')
['i', 'am', '[', 'mask', ']', '.']
>>> toker.tokenize('I am [MASK].')
['i', 'am', '[MASK]', '.']
但是如果手动把句号分开就可以了
>>> toker2.tokenize('I am [MASK] .')
['i', 'am', '[MASK]', '.']
目前的结果来看，比正常finetune结果基本一致
https://github.com/JinmingZhao/prompt_demos
尝试不同的seed, 结果会有一点提升，跟baseline的结果差不多。

采用跨模态的这种Mask预测，输入的正常的文本，mask操作再dataset中操作.
[CLS] text1 + i am [MASK] [SEP] v----  a---- 
[CLS] text1 + i am [MASK] [SEP] v---- 
[CLS] text1 + i am [MASK] [SEP] a---- 
[CLS] i am [MASK] [SEP] v----  a---- 
[CLS] i am [MASK] [SEP] v---- 
[CLS] i am [MASK] [SEP] a---- 
[CLS] i am [MASK] [SEP] l---- 

设计多种不同的预先训练任务:
l_mask_va -- promptmask
l_mask_v -- promptmask
l_mask_a -- promptmask
l_mask -- promptmask
mask_v -- cmpromptmask (cross-modality)
mask_a -- cmpromptmask (cross-modality)
mask_va -- cmpromptmask (cross-modality)
测试场景分别对应上面的7种测试场景，符合模态缺失的场景 -- 明天先把这个实现一下.  --对于整体的lva的识别有帮助，但是对于模态缺失的场景没啥帮助。
/data7/MEmoBert/code/uniter3m/config/downstream/pretrain-task-iemocap-base-2gpu_cm_mask_prompt.json

目前全场景 /data7/emobert/exp/prompt_pretrain/iemocap_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_prompt_lr5e-5_trnval
结果UAR=81.14%，比目前pretrain+finetune的结果UAR=80.79% 好一些， 结果还不错～
msp_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_prompt_lr5e-5_trnval
结果UAR=72.70%，比目前pretrain+finetune的结果 UAR=71.10% 好一些， 结果还不错～


## <分析7> 基于NSP的机制也可以，可以把 prompt 放前面 -- Discard
采用类似itm的做法，把放在第二个位置，这样比较方便
[CLS] text1 [SEP] v----  a---- 
对比采用itm预训练任何和不采用itm预训练任务的下游任务
[CLS] text1 [SEP] prompt [SEP] v----  a---- 
[CLS] 0     [SEP] 0      [SEP] 1----  2----    token type.
目前的结果来看，比正常finetune结果低一些～


## <分析8> 探索模态缺失的场景 -- Going
不同的模态制定不同的 template，比如
方案1, 将固定的模板放在前面 VS 放在后面的结果:
    [CLS] i am [MASK] + text1 [SEP] v----  a---- 
    [CLS] i am [MASK] + text1 [SEP] v---- 
    [CLS] i am [MASK] + text1 [SEP] a---- 
    [CLS] i am [MASK] + text1 [SEP]
    [CLS] i am [MASK] [SEP] v----  a---- 
    [CLS] i am [MASK] [SEP] v---- 
    [CLS] i am [MASK] [SEP] a---- 
方案2, 换用一个更自然的模版:
    [CLS] I feel [MASK] through text1 [SEP] v----  a---- 
    [CLS] I feel [MASK] through text1 [SEP] v----
    [CLS] I feel [MASK] through text1 [SEP] a----
    [CLS] I feel [MASK] through text1 [SEP]
    [CLS] I feel [MASK] through [SEP] v----  a---- 
    [CLS] I feel [MASK] through [SEP] v----
    [CLS] I feel [MASK] through [SEP] a---- 

方案3.1, 给不同的 condition 设置不同的标志, S V T:
    [CLS] T V S: I feel [MASK] through text1 [SEP] v----  a---- 
    [CLS] T V:, I feel [MASK] through text1 [SEP] v----
    [CLS] T S: I feel [MASK] through text1 [SEP] a----
    [CLS] T: I feel [MASK] through text1 [SEP]
    [CLS] V S: I feel [MASK] through [SEP] v----  a---- 
    [CLS] V: I feel [MASK] through [SEP] v----
    [CLS] S: I feel [MASK] through [SEP] a---- 
方案3.2, 换用一个分组的任务提示方式:
    [CLS] from three modalities, I feel [MASK] through text1 [SEP] v----  a---- 
    [CLS] from two modalities I feel [MASK] through text1 [SEP] v----
    [CLS] from two modalities I feel [MASK] through text1 [SEP] a----
    [CLS] from one modalities I feel [MASK] through text1 [SEP]
    [CLS] from two modalities I feel [MASK] through [SEP] v----  a---- 
    [CLS] from one modalities I feel [MASK] through [SEP] v----
    [CLS] from one modalities I feel [MASK] through [SEP] a---- 
方案3.3, 换用一个更自然详细的任务提示方式:
    [CLS] from text, visual and speech modalities, I feel [MASK] through text1 [SEP] v----  a---- 
    [CLS] from text and visual modalities, I feel [MASK] through text1 [SEP] v----
    [CLS] from text and speech modalities, I feel [MASK] through text1 [SEP] a----
    [CLS] from text modalities, I feel [MASK] through text1 [SEP]
    [CLS] from visual and speech modalities, I feel [MASK] through [SEP] v----  a---- 
    [CLS] from visual modalities, I feel [MASK] through [SEP] v----
    [CLS] from speech modalities, I feel [MASK] through [SEP] a---- 


方案4. 采用 soft-prompt 的方式, 采用 S 个 unused-embedding 来作为 soft-prompt.


方案5. 采用 soft-prompt 的初始化采用 情感词来做。 




## <分析9> 预训练的模型固定住，在下游任务进行测试 -- Going


## <分析10> 跨数据集的实验 -- Going
IEMOCAP和MSP两个数据集交叉验证。