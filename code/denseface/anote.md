## Step1 --doing
重写tensorflow版本的denseface，并在FER+上进行训练
FER+ 数据地址：
/data3/zjm/dataset/ferplus/npy_data  是处理好的灰度图输入 train/val/test 以及 对应的target.

fer_idx_to_class = ['neu', 'hap', 'sur', 'sad', 'ang', 'dis', 'fea', 'con']

## Step2
加入更多的数据进行训练
/data3/zjm/dataset/SFEW_2.0
/data3/zjm/dataset/ExpW