import os
import shutil

if __name__=='__main__':
    train_dir_tmp = os.path.join("data", "train_tmp")
    shutil.rmtree(train_dir_tmp)
    print('deleted')
