import os
import time

def f0():
    time.sleep(1)
    return 1, 2

def f1(x, y):
    time.sleep(1)
    return x+y, x*x + y*y

def f3(x):
    time.sleep(1)
    return x+2

def f4(x):
    time.sleep(1)
    return x+4

def f5(x, y, z):
    time.sleep(1)
    return x+y+z

def f6(x):
    time.sleep(1)
    return x-3

def calc_indegree(graph):
    ans = {}
    for key, value in graph.items():
        if key in passed_node:
            continue
        for v in value:
            v = v[1]
            if ans.get(v):
                ans[v].append(key)
            else: 
                ans[v] = [key]

    for key in graph.keys():
        if key not in ans.keys() and key not in passed_node:
            ans[key] = []
    
    return ans

def find_zero_degree(indegree_graph):
    ans = []
    for key, value in indegree_graph.items():
        if len(value) == 0:
            ans.append(key)
    return ans

def update_graph(decline_node):
    passed_node.append(decline_node)

def get_input_param(func, graph, shared_memory):
    ans = []
    for key, value in graph.items():
        for v in value:
            if v[1] == func:
                ans.append(shared_memory[key+'_'+str(v[0])])
    return ans

def run_func(func):
    param = get_input_param(func, graph, shared_memory)
    print(func)
    print("found param:", param)
    ans = globals()[func](*param)
    print(ans)
    if not isinstance(ans, tuple):
        ans = (ans,)
    for i in range(len(ans)):
        shared_memory[func+'_'+str(i)] = ans[i]

    passed_node.append(func)

    print(shared_memory)
    print()

if __name__ == '__main__':
    import multiprocessing
    with multiprocessing.Manager() as MG:
        graph = {
            'f0': [(0, 'f1', 0), (1, 'f1', 1), (1, 'f6', 1)],
            'f1': [(0, 'f3', 0), (1, 'f4', 0)],
            'f3': [(0, 'f5', 0)],
            'f4': [(0, 'f5', 1)],
            'f6': [(0, 'f5', 2)],
            'f5': []
        }
        passed_node = MG.list()
        shared_memory = MG.dict()
        start = time.time()
        while len(passed_node) < len(graph.keys()):
            pool = multiprocessing.Pool(2)
            indegree_graph = calc_indegree(graph)
            print(indegree_graph)
            process = []
            conduct_functions = find_zero_degree(indegree_graph)
            for func in conduct_functions: 
                pool.apply_async(run_func, (func,))

            pool.close()
            pool.join()
            # for p in process:
            #     p.join()
            
        
        end = time.time()
        print("time:", end-start)
        
        print(shared_memory)
    