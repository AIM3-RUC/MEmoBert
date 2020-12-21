## For all code, you must
export PYTHONPATH=/data7/MEmoBert

## 构建LMDB 特征数据库
Step1 将抽取的Denseface特征进行 segmentId = movie_name + '_' + segment_index 转化为所有的 npz 文件
    build_lmdb/trans2npz.py
Step2 基于npz数据，构建视觉的 LMDB 数据库
    code/uniter/scripts/create_imgdb.sh
Step3 基本英文的 bert-base-uncased 模型构建 txt_db
    build_lmdb/mk_txtdb_by_faces.py
---Manual Check OK 
## 模型转换，采用 bert-base-uncased
code/uniter/scripts/convert_ckpt.py

## Bugs


## 修改记录
1. 由于Faces之间也是有顺序的，所以需要进行 name2nbb 简单的取多少个，而是应该根据阈值过滤相应位置的数据, 重写数据获取的代码,
不用，因为构建img-db的时候已经过滤了，所有 img2nbb的个数跟保存的特征是一致的.

2. 同样由于Faces之间是连续的，所以需要图片应该也要有 position 的概念, 加上 position embedding.


## 挂载金山云
KPFS
下载软件并安装
wget https://ks3-cn-beijing.ksyun.com/ksyun-resource/project/kpfs/console/kpfs-client-1.3.1-1.el7.centos.x86_64.rpm
rpm -ivh kpfs-client-1.3.1-1.el7.centos.x86_64.rpm
创建挂载目录
mkdir /mnt/kpfs
挂载
kpfs -ip 172.31.252.24 -fs kpfs-cq57fmum -dir /mnt/kpfs


自动化挂载
vim /etc/kpfs/mount.cfg  将下面这行写入文件
-ip 172.31.252.24 -fs kpfs-cq57fmum -dir /mnt/kpfs

重启服务
systemctl restart kpfs-client.service