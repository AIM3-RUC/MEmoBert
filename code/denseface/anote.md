## Step1
重写tensorflow版本的denseface，并在FER+上进行训练
FER+ 数据地址：
/data3/zjm/dataset/ferplus/npy_data  是处理好的灰度图输入 train/val/test 以及 对应的target.
fer_idx_to_class = ['neu', 'hap', 'sur', 'sad', 'ang', 'dis', 'fea', 'con']
densenet100_adam0.001_0.0 epoch-43
[Val] result WA: 0.8285 UAR 0.6795 F1 0.6969
[Tst] result WA: 0.8229 UAR 0.6450 F1 0.6716

## Step2
加入更多的数据进行训练
/data3/zjm/dataset/SFEW_2.0
/data3/zjm/dataset/ExpW

## Step3
实现更多的模型，比如 VggFace 以及 ResNet
VggNet:
vggnet_adam0.0001_0.25 epoch-68 --Use this.
[Val] result WA: 0.8103 UAR 0.6175 F1 0.6508
[Tst] result WA: 0.8030 UAR 0.6110 F1 0.6430
vggnet_adam0.001_0.1 epoch-47
[Val] result WA: 0.7714 UAR 0.4903 F1 0.5266
[Tst] result WA: 0.7603 UAR 0.4763 F1 0.5084
vggnet_adam0.001_0.25 epoch-26
[Val] result WA: 0.7419 UAR 0.4436 F1 0.4679
[Tst] result WA: 0.7341 UAR 0.4513 F1 0.4680
vggnet_adam0.0001_0.5
[Val] result WA: 0.7985 UAR 0.6078 F1 0.6299
[Tst] result WA: 0.7990 UAR 0.6111 F1 0.6367
vggnetsmall_adam0.0001_0.25. epoch-84
[Val] result WA: 0.7677 UAR 0.5350 F1 0.5589
[Tst] result WA: 0.7467 UAR 0.5181 F1 0.5411

ResNet, 仿照vggnet 和 densenet 的经验，resnet一般dropout=0:
训练不太稳定，需要加大 reduce_half_lr_epoch=40, 想要训练稳定可以采用线性下降的方式。
resnet18_adam_warmup_run2_adam0.0005_0.0
Loading best model found on val set: epoch-68
[Val] result WA: 0.8128 UAR 0.6448 F1 0.6725
[Tst] result WA: 0.8109 UAR 0.6363 F1 0.6609

resnet18_newrun1_adam0.0005_0.0
Loading best model found on val set: epoch-61
[Val] result WA: 0.8165 UAR 0.6492 F1 0.6814
[Tst] result WA: 0.8041 UAR 0.6187 F1 0.6489

resnet18_newrun2_adam0.0005_0.0
Loading best model found on val set: epoch-34
[Val] result WA: 0.7983 UAR 0.6232 F1 0.6502
[Tst] result WA: 0.7839 UAR 0.5944 F1 0.6134
resnet18_adam0.0001_0.0
Loading best model found on val set: epoch-44  
[Val] result WA: 0.8120 UAR 0.6616 F1 0.6794
[Tst] result WA: 0.8123 UAR 0.6475 F1 0.6696
resnet18_adam0.0001_0.0 run1
Loading best model found on val set: epoch-41
[Val] result WA: 0.8081 UAR 0.6523 F1 0.6928
[Tst] result WA: 0.8004 UAR 0.6239 F1 0.6539
resnet18_adam0.0001_0.0 run2
Loading best model found on val set: epoch-47
[Val] result WA: 0.7999 UAR 0.6200 F1 0.6349
[Tst] result WA: 0.7984 UAR 0.6050 F1 0.6363
resnet18_adam0.0001_0.1
Loading best model found on val set: epoch-16
[Val] result WA: 0.7887 UAR 0.5652 F1 0.5974
[Tst] result WA: 0.7668 UAR 0.5431 F1 0.5721
resnet18_adam0.0001_0.25
Loading best model found on val set: epoch-9
[Val] result WA: 0.7579 UAR 0.5339 F1 0.5533
[Tst] result WA: 0.7387 UAR 0.4993 F1 0.5191
resnet34_adam0.0001_0.0
Loading best model found on val set: epoch-34
[Val] result WA: 0.7929 UAR 0.6710 F1 0.6690
[Tst] result WA: 0.7805 UAR 0.6464 F1 0.6391
resnet34_adam0.0001_0.1
Loading best model found on val set: epoch-24
[Val] result WA: 0.7921 UAR 0.6158 F1 0.6361
[Tst] result WA: 0.7890 UAR 0.5856 F1 0.5996
resnet34_adam0.001_0.0 Or Use this one
Loading best model found on val set: epoch-41
[Val] result WA: 0.8218 UAR 0.6539 F1 0.6866
[Tst] result WA: 0.8209 UAR 0.6282 F1 0.6587