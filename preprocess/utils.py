import os

def get_basename(path):
    basename = os.path.basename(path)
    if os.path.isfile(path):
        basename = basename[:basename.rfind('.')]
    return basename

def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
