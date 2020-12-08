## For all code, you must
export PYTHONPATH=/data7/MEmoBert

## 构建LMDB 特征数据库
Step1 将抽取的Denseface特征进行 segmentId = movie_name + '_' + segment_index 转化为所有的 npz 文件
    build_lmdb/trans2npz.py
Step2 基于npz数据，构建视觉的 LMDB 数据库
    code/uniter/scripts/create_imgdb.sh


## bug1 
有四部电影的数据是空的
## bug2
目前特征数据总数是 89314 /data7/emobert/ft_npzs/movie110_v1
但是 active_spk 中统计的结果只有 88137 条记录。preprocess/data/analyse/fileter_details.txt
## bug3
如果想获取一个视频中的脸是否连续，每一张脸对应的index有没有？
## bug4
目前ffmped对于每张脸的置信度有多少？ 比如有的脸特别模糊或者只能看到脸的一部分，这种的脸要不要根据阈值过滤？