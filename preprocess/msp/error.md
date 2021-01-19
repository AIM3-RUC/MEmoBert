## Bug 
MSP 数据处理的 Bug 真的无语。

int2name 是按照 10:1:1来划分的, 但是 视觉 feature是按照 11:1 来划分的，
val_int2name == val.h5 == tst.h5
tst_int2name + trn_int2name == trn.h5 = 3608
所以对于tst集合和trn集合 int2name 和 h5文件中的 keys 肯定会匹配不上。

但是对于文本信息，是按照 10:1:1来划分的，连11:1都没有做！
root@emobert2220:/data7/MEmoBert/emobert/exp/evaluation/MSP-IMPROV/feature/bert_large/1#
>>> import h5py
>>> a = h5py.File('val.h5')
>>> len(a)
211
>>> a = h5py.File('tst.h5')
>>> len(a)
215
>>> a = h5py.File('trn.h5')
>>> len(a)
3393