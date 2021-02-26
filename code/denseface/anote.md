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

ResNet, 仿照vggnet 和 densenet 的经验，resnet一般dropout比较小:
resnet18_adam0.0001_0.0

resnet18_adam0.0001_0.1

resnet18_adam0.0001_0.25

resnet34_adam0.0001_0.0

resnet34_adam0.0001_0.1

resnet34_adam0.001_0.0