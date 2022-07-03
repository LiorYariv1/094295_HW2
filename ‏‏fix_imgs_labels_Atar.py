import pandas as pd
import os
from pathlib import Path
import warnings
import shutil



def rearrange(dir, img_name, op, exact):
    file = os.path.join(dir, img_name)
    print(f"{exact} {file}  > {op}")
    counter = 0
    if op == "lete":
        if (exact==1):
            if os.path.isfile(file):
                os.remove(file)
                counter+=1
            else:
                warnings.warn(f"couldn't find file: {file} *$*$*$*$*$*$*$*$*$*$*$*$*$*$*$*$*$$$")
        else:
            for p in os.listdir(dir):
                if img_name in p:
                    file = os.path.join(dir, p)
                    os.remove(file)
                    counter += 1
        print(f"{counter} images were deleted ({file})")
    else:
        # if not Path(op).exists:
        #     print(f"not Path(op).exists -  {op}")
        #     os.mkdir(op)
        for p in os.listdir(dir):
            if img_name in p:
                file = os.path.join(dir, p)
                try:
                    shutil.move(file, op)
                    counter += 1
                except Exception as e:
                    print(e)
        print(f"{counter} images were moved from {dir} to {op}")



if __name__ == "__main__":
    print('starting')
    # shutil.copytree('/home/student/HW2/data_orig', '/home/student/HW2/data_noaug')
    df = pd.read_csv('new_fixes.csv')
    # os.mkdir('data_noaug/maybe')
    for _, row in df.iterrows():
        # print(row['path'])
        # print(f"row['path'][2:] {row['path'][2:]}")
        # print(f"row['path'][2:].replace('data','data_noaug') {row['path'][2:].replace('data','data_noaug')}")
        # print(row['dest'])
        # print(f"row['dest'][2:].replace('data','data_noaug') {row['dest'][2:].replace('data','data_noaug')}")
        # print(row['path'][2:],"\n", row['img'],"\n", row['dest'][2:],"\n", row['exact']"\n")
        rearrange(row['path'][2:], row['img'], row['dest'][2:], row['exact'])
    #
    #print(os.listdir('data/val/x'))
    print("DONE")
    """
    type(row['path']): <class 'str'>
    type(row['img']): <class 'str'>
    type(row['dest']): <class 'str'>
    type(row['exact']): <class 'int'>    
    """

