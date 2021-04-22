https://webvpn.ruc.edu.cn/ 然后进行跳转到下面的地址
env: py35
pip install jupyter
网页版访问:
    在宿主机修改docker的端口映射（运行一次）
    docker inspect docker name | grep IPAddress
    sudo iptables -t nat -A  DOCKER -p tcp --dport 2259 -j DNAT --to-destination 172.17.0.3:8080
    在docker内启动服务
    jupyter notebook --ip=0.0.0.0 --port=8080 --no-browser --allow-root
    http://202.112.113.78:2259/?token=b463012c52ce7d5f4e71769a7ccf76194a064a41b9b7552e

    最后直接访问。
    http://202.112.113.78:2259/notebooks/code/bert_pretrain/visualize.ipynb

VSCODE + jupyter notebook
    在 extention 中安装 jupyter 和 jupyter notebook 即可
    如何启动？
    ctrl + shift + P 
    然后输入 Jupyter: Create New Blank Notebook 即创建新的 ipynb 文件

关于 Uniter 的 Visualization.
Uniter 模型的可视化不能直接用 BertModel, 因为需要处理其他模态的数据，所以先提取修改Uniter的代码，
返回对应的 Attention 的值，目前的 Uniter 代码没有返回 Attention的接口。
layers 12 and each layer with shape torch.Size([1, 12, 11, 11])
然后用 bertviz 的 show_head_view 或者 show_model_view 进行可视化。