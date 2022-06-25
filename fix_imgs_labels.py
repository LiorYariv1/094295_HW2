import pandas as pd
import os
from pathlib import Path
import warnings
import shutil



def rearrange(dir, img_name, op):
    file = os.path.join(dir, img_name)
    print(file)
    if op == "delete":
        if os.path.isfile(file):
            os.remove(file)
        else:
            warnings.warn(f"couldn't find file: {file}")
    else:
        if not Path(op).exists:
            os.mkdir(op)
        shutil.copy(file, op)


if __name__ == "__main__":
    print('starting')
    # shutil.copytree('/home/student/HW2/data_orig', '/home/student/HW2/data_noaug')
    df = pd.read_csv('fixes.csv')
    os.mkdir('data_noaug/maybe')
    for _, row in df.iterrows():
        rearrange(row['path'][2:].replace('data','data_noaug'), row['img'],row['dest'][2:].replace('data','data_noaug'))