## Target 
1. 验证是否是视觉信息的影响，本实验利用Pretrained Bert-base 模型进行初始化，在目前的训练数据集上进行MLM预训练任务的预训练。
验证在20w左右的数据规模下进行预训练，能否达到的 masked token 的准确率是多少？

2. 验证下游任务纯文本的性能上限，目前多模态的下游任务的性能跟单纯的文本的性能持平(lrc), 这里系统的验证一下，三个下游任务单纯用文本进行Finetune的性能是多少？ 从而验证目前的多模态融合策略是否有效。

## 具体实现 
pip install transformers
pip install datasets
采用torch版本的 https://github.com/huggingface/transformers
由于我们只需要进行MLM任务的预训练，不需要 next-sentence-prediction的任务，
数据准备只需要一行一句话，不用空行分割表示来自不同document.
即相当于进行 Language Model 的训练，可以参考
https://github.com/huggingface/transformers/tree/master/examples/language-modeling#robertabertdistilbert-and-masked-language-modeling

数据格式要求：
每行一句话即可


## Bugs
1. 关于多卡训练
https://github.com/huggingface/transformers/issues/5488
https://blog.csdn.net/qq_39894692/article/details/114370889

python -m torch.distributed.launch \
    --nproc_per_node num-gpus run_mlm.py
2. 手动安装，否则 mlm 无法运行
https://github.com/huggingface/transformers/tree/master/examples#important-note

## 实验结果
 --max_seq_length 30 --per_device_train_batch_size 64 \
--learning_rate 2e-5 \
--weight_decay 0.01 \
--num_train_epochs 20 \
--warmup_steps 2000 \
--lr_scheduler_type 'linear' \
只调整 learning_rate: 
1e-4; 很快过拟合，15epoch. eval-loss 在 2.0 左右, train-loss:1.5479. 
5e-5; 很快过拟合，15epoch, eval-loss 在 1.9 左右, train-loss: 1.6234
2e-5, eval-loss 在 1.8 左右, train-loss: 1.7728

请教张老板之后，经验是 增大batch-szie，调整学习率和epoch(epoch总数影响训练中的学习率). 
调整策略：去掉 weight-decay. warmup=1000. 控制变量，先分析。
采用2e-5的学习率，调整训练epoch=10. batch-size=64, acc-step=1.
 --- 没有过拟合, eval-loss 在 1.87 左右, train-loss: 1.92, eval-ppl=6.7419 持续下降，都下降的很慢。
采用2e-5的学习率，调整训练epoch=15. batch-size=64, acc-step=1.
 --- 没有过拟合, eval-loss 在 1.90 左右, train-loss: 1.82 (后期学习率太小，train都不降了), eval-ppl= 6.78027 持续下降，都下降的很慢。
采用2e-5的学习率，调整训练epoch=20, batch-size=128, acc-step=2.
 --- 没有过拟合, eval-loss 在 1.90 震荡, train-loss: 1.8789 , eval-ppl=6.509 持续下降，都下降的很慢。

--- 综上，第一学习率后期会变的很小，需要调整学习率策略。第二 batch-size 增大作用不大
采用2e-4的学习率，调整训练epoch=10. batch-size=128, acc-step=2.


## 扩展，可以考虑用相关的电影字幕的数据集进行Pretrain，然后用当前的数据集进行 Pretrain 进行测试。
https://github.com/huggingface/transformers/issues/10474

1. cornell_movie_dialog 对话数据集
https://huggingface.co/datasets/cornell_movie_dialog
220,579 conversational exchanges between 10,292 pairs of movie characters
involves 9,035 characters from 617 movies
in total 304,713 utterances

2. open_subtitles 电影字幕的对话数据集，可以采用其中英文的字幕数据
https://huggingface.co/datasets/open_subtitles
dataset = load_dataset("open_subtitles", lang1="fi", lang2="hi")

具体下载：
1.1 解析的英文字幕，包含每个词的起止时间。ls -lR|grep "^-"|wc -l  递归的统计文件的个数。
http://opus.lingfil.uu.se/download.php?f=OpenSubtitles/en.tar.gz 
大约包含各种十几种类型的, 2000部电影。 统计一下多少句话？

1.2 直接手动下载
https://opus.nlpl.eu/OpenSubtitles.php
随机下载几个 en-xx 语音的数据，只利用英文的部分即可。
https://opus.nlpl.eu/download.php/?f=OpenSubtitles/v2018/moses/en-es.txt.zip # 
https://opus.nlpl.eu/download.php/?f=OpenSubtitles/v2018/moses/en-fr.txt.zip #
https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/en-zh_cn.txt.zip # 1120,3286.

/data7/MEmoBert/emobert/exp/pretrain/only_v1v2v3_txt/data/OpenSubtitlesV2018/OpenSubtitles.en-zh_cn.en.txt   
--Use this.

增加为100w的数据进行先训练，然后再当前数据集上进行 domain pretrain.
--train_file OpenSubtitles.en-zh_cn.en_trn100w.txt
--validation_file OpenSubtitles.en-zh_cn.en_val10w.txt