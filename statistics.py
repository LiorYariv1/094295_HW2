import os
import matplotlib.pyplot as plt
import numpy as np
def hist_num_sapmles(dir):
    samples = []
    total = 0
    LETTERS = ['I','II','III','IV','V','VI','VII','VIII','IX','X']
    letters = ['i','ii','iii','iv','v','vi','vii','viii','ix','x']
    for letter in letters:
        num_dir = len(os.listdir(f'{dir}/{letter}'))
        print(f'number of samples in directory {letter}: {num_dir}')
        total+=num_dir
        samples.append(num_dir)
        fig = plt.figure()
    plt.bar(LETTERS,samples)

    plt.suptitle('Number Of Images Per Letter In The Given Train Set')
    plt.axhline(total/10, color='#59C9D9')
    plt.annotate('Mean',(9,207), fontsize=9)
    plt.axhline(np.median(samples), color='#62DFBB')
    plt.annotate('Median',(9,np.median(samples)+0.2),fontsize=9)
    plt.show()
    print('total number of images: ', total)



hist_num_sapmles('data_orig/train')
# hist_num_sapmles('data_noaug/train')
# hist_num_sapmles('data/train')
