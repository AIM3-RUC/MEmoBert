import os
from os.path import join
import shutil

root_dir = '/mnt/data2/videos'
mov_dirs = os.listdir(root_dir)
print('total {} movies'.format(len(mov_dirs)))
for mov_dir in mov_dirs:
    if os.path.isfile(join(root_dir,mov_dir)):
        continue
    filepaths = os.listdir(join(root_dir,mov_dir))
    if len(filepaths) != 2:
        print("check this {} {}".format(mov_dir, len(filepaths)))
        continue
    print("current {}".format(mov_dir))
    No = mov_dir.split('_')[0]
    No = 'No0{}'.format(No)
    for filepath in filepaths:
        postfix = filepath.split('.')[-1]
        print(join(root_dir, mov_dir, '{}_{}.{}'.format(No, 'None', postfix)))
        shutil.move(join(root_dir, mov_dir, filepath), join(root_dir, mov_dir, '{}_{}.{}'.format(No, 'None', postfix)))
        shutil.move(join(root_dir, mov_dir, '{}_{}.{}'.format(No, 'None', postfix)), root_dir)