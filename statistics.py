import os
import matplotlib.pyplot as plt

def hist_num_sapmles(dir):
    samples = []
    for dir_name in os.listdir(dir):
        num_dir = len(os.listdir(f'{dir}/{dir_name}'))
        print(f'number of samples in directory {dir_name}: {num_dir}')
        samples.append(num_dir)
        fig = plt.figure()
    plt.bar(range(1,11),samples)
    plt.suptitle(dir.split('/')[0])
    plt.show()


hist_num_sapmles('data_orig/train')
hist_num_sapmles('data_noaug/train')
hist_num_sapmles('data/train')
