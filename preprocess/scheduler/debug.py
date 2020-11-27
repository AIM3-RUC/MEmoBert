from functools import partial

# def partial(fuc, *part_args):
#     def wrapper(*extra_args):
#         args = list(part_args)
#         print("args:", args)
#         print("extra_args:", extra_args)
#         args += extra_args
#         return fuc(*args)
#     return wrapper

def add(x,y):
    return x+y

if __name__ == '__main__':
    add_one = partial(add, 1, 1)
    ans = add_one(3)
    print(ans)
