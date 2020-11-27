import os
import subprocess
import torch.multiprocessing as multiprocessing
from functools import partial

def _init_model(device_num, model, model_kwargs):
    try:
        return model(device=device_num, **model_kwargs)
    except RuntimeError:
        return None


def init_model_on_gpus(model, model_kwargs, num, avaible_gpus=None, memory_threshold=0):
    ''' 在GPU上生成n个预训练模型用于抽取特征
    Parameters:
    ----------------------------
        model: 需要在GPU上初始化的模型, 需要接收device参数表示使用的gpu(只支持单gpu), model_kwargs是初始化模型所需要的其他参数, 以dict形式传入
        num: 生成多少个模型,
        avaible_gpus: 可用gpu列表
        memory_threshold: 仅选取剩余显存 > # 的gpu, 
    
    Return:
    -----------------------------
        model_instances->list, target_gpu->list
    '''
    assert num > 0, 'Model nums must be larger than 0'
    init_model_func = partial(_init_model, model=model, model_kwargs=model_kwargs)
    p = subprocess.Popen('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    err = p.stderr.read()
    if err:
        raise RuntimeError(err)

    gpu_infos = p.stdout.readlines()
    gpu_mem_remain = [(i, x.decode().split(':')[-1].strip()) for i, x in enumerate(gpu_infos)]
    if avaible_gpus:
        assert isinstance(avaible_gpus, (tuple, list)) and any([isinstance(x, int) for x in avaible_gpus]), \
                'avaible_gpus should be a list or tuple of integers, got:{}'.format(avaible_gpus)
        gpu_mem_remain = [gpu_mem_remain[x] for x in avaible_gpus]
    
    gpu_mem_remain = [(i, int(x.split()[0])) for i, x in gpu_mem_remain if int(x.split()[0]) >= memory_threshold]
    gpu_mem_remain = sorted(gpu_mem_remain, key=lambda  x: x[1], reverse=True)
    gpu_id = [x[0] for x in gpu_mem_remain]
    ret, ret_gpu_id = [], []

    if len(gpu_id) == 0:
        print('warning: No gpu satisfies the requirements')
        return ret, ret_gpu_id
    
    _num = min(num, len(gpu_id))

    if _num > 3:
        pool = multiprocessing.Pool(_num)
        _ret = pool.map(init_model_func, gpu_id[:_num])
        pool.close()
        pool.join()
    else:
        _ret = [init_model_func(device_num=x) for x in gpu_id[:_num]]
    
    _succ_gpu_id = [gpu_id[i] for i in range(len(_ret)) if _ret[i]]
    _ret = list(filter(lambda x: x, _ret))
    _num_succ = len(_ret)
    ret += _ret
    ret_gpu_id += _succ_gpu_id
    if _num_succ < num and len(_ret):
        print("model init on", [x for x in _succ_gpu_id])
        _ret, _ret_gpu_id = init_model_on_gpus(model, model_kwargs, num-_num, avaible_gpus=avaible_gpus, \
            memory_threshold=memory_threshold)
        ret += _ret
        ret_gpu_id += _ret_gpu_id
    
    return ret, ret_gpu_id

if __name__ == "__main__":
    import sys, time
    sys.path.append('../')
    from tasks.text import BertExtractor
    start = time.time()
    models = init_model_on_gpus(BertExtractor, {}, 2, avaible_gpus=None, memory_threshold=2000)
    end = time.time()
    print(models)
    print(end-start)
