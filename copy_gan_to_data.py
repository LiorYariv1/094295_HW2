import os
import shutil

copy_to =  '/home/student/HW2/data/train'
root = '/home/student/HW2/Images'
for folder in os.listdir(root):
    path_src = os.path.join(root,folder)
    path_dest = os.path.join(copy_to,folder)
    destination = shutil.copytree(path_src, path_dest,dirs_exist_ok=True)
    print(destination)

