import os
import torch.multiprocessing as multiprocessing
from tqdm import tqdm
from functools import reduce
# from base_scheduler import BaseScheduler

def simple_run(func, args, cache, i):
    ans = list(map(lambda x: func(x), args))
    cache[i] = ans

def simple_processer(funcs, args, num):
    print(funcs)
    input()
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    ans = []
    sample_len = len(args) / num
    assert len(funcs) == num
    args_each = [args[int(sample_len*i):int(sample_len*(i+1))] for i in range(num)]
    with multiprocessing.Manager() as MG:
        sub_process = []
        cache = MG.dict()
        for i in range(num):
            # p = multiprocessing.Process(target=simple_run, args=(funcs[i], args_each[i], cache, i))
            p = multiprocessing.spawn(simple_run, args=(funcs[i], args_each[i], cache, i), nprocs=1, join=True, daemon=False)
            sub_process.append(p)
        
        # for p in sub_process:
        #     p.spawn()
        #     # p.join()

        for p in sub_process:
            p.join()
        
        ans = sorted(dict(cache).items(), key=lambda x: x[0])
        print(ans)
        input()
        ans = list(reduce(lambda x, y: x+y ,map(lambda x: x[1], ans)))

    return ans

if __name__ == '__main__':
    def func(x):
        return x*x
    
    import time
    start = time.time()
    funcs = [func for x in range(4)]
    ans = simple_processer(funcs, range(100), 4)
    end = time.time()
    print(end - start)
    print(ans)
    