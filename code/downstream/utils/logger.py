import time
import os
import logging

def get_logger(path, suffix):
    cur_time = time.strftime('%Y-%m-%d-%H.%M.%S', time.localtime(time.time()))
    logger = logging.getLogger(cur_time)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(os.path.join(path, f"{suffix}_{cur_time}.log"))
    handler.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # handler.setFormatter(formatter)

    # console = logging.StreamHandler()
    # console.setLevel(logging.INFO)

    logger.addHandler(handler)
    # logger.addHandler(console)
    return logger