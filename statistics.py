import os
import matplotlib.pyplot as plt
import numpy as np
def hist_num_sapmles(dir):
    samples = []
    total = 0
    LETTERS = ['I','II','III','IV','V','VI','VII','VIII','IX','X']
    letters = ['i','ii','iii','iv','v','vi','vii','viii','ix','x']
    for letter in letters:
        images_list = os.listdir(f'{dir}/{letter}')
        images_list = [x for x in images_list if 'img' not in x and 'shift' not in x and 'flip' not in x]
        num_dir = len(images_list)
        print(f'number of samples in directory {letter}: {num_dir}')
        total+=num_dir
        samples.append(num_dir)
        fig = plt.figure()
    plt.bar(LETTERS,samples)

    plt.suptitle('Number Of Images Per Letter In The Cleaned Train Set')
    plt.axhline(np.mean(samples), color='#59C9D9')
    plt.annotate('Mean',(9,np.mean(samples)+0.35), fontsize=9)
    plt.axhline(np.median(samples), color='#62DFBB')
    plt.annotate('Median',(9,np.median(samples)+0.5),fontsize=9)
    plt.show()
    print('total number of images: ', total)
    print('Mean: ',total/10)
    print('Median: ',np.median(samples))
    print('Min: ', np.min(samples))
    print('Max: ', np.max(samples))

def hist_multiple():
    out_samples =[]
    in_samples = []
    total_out = 0
    total_in = 0
    LETTERS = ['I','II','III','IV','V','VI','VII','VIII','IX','X']
    letters = ['i','ii','iii','iv','v','vi','vii','viii','ix','x']
    in_dir = 'data/train'
    out_dir = 'data_orig/train'
    for letter in letters:
        print('*'*10, f'Letter: {letter}','*'*10)
        images_list = os.listdir(f'{in_dir}/{letter}')
        images_list = [x for x in images_list if 'img' not in x and 'shift' not in x and 'flip' not in x]
        num_dir_in = len(images_list)
        num_dir_out = len(os.listdir(f'{out_dir}/{letter}'))
        print(f'number of samples in directory Before: {num_dir_out}')
        print(f'number of samples in directory After: {num_dir_in}')
        total_in+=num_dir_in
        total_out +=num_dir_out
        in_samples.append(num_dir_in)
        out_samples.append(num_dir_out)
        fig = plt.figure()

    plt.bar(LETTERS,out_samples, width=0.9, align='center',color = '#3280C4', alpha=0.3)# the widest bar encompasses the other two
    plt.bar(LETTERS,in_samples, width=0.8)  # the widest bar encompasses the other two
    plt.suptitle('Number Of Images Per Letter In The Cleaned Train Set')
    plt.axhline(np.mean(in_samples), color='#59C9D9')
    plt.annotate('Mean',(9,np.mean(in_samples)+0.35), fontsize=9)
    plt.axhline(np.median(in_samples), color='#62DFBB')
    plt.annotate('Median',(9,np.median(in_samples)+0.5),fontsize=9)
    plt.show()
    print('total number of images: ', total_in)
    print('Mean: ',total_in/10)
    print('Median: ',np.median(in_samples))
    print('Min: ', np.min(in_samples))
    print('Max: ', np.max(in_samples))
    out_samples =[]
    in_samples = []
    total_out = 0
    total_in = 0
    LETTERS = ['I','II','III','IV','V','VI','VII','VIII','IX','X']
    letters = ['i','ii','iii','iv','v','vi','vii','viii','ix','x']
    in_dir = 'data/train'
    out_dir = 'data_orig/train'
    for letter in letters:
        print('*'*10, f'Letter: {letter}','*'*10)
        images_list = os.listdir(f'{in_dir}/{letter}')
        images_list = [x for x in images_list if 'img' not in x and 'shift' not in x and 'flip' not in x]
        num_dir_in = len(images_list)
        num_dir_out = len(os.listdir(f'{out_dir}/{letter}'))
        print(f'number of samples in directory Before: {num_dir_out}')
        print(f'number of samples in directory After: {num_dir_in}')
        total_in+=num_dir_in
        total_out +=num_dir_out
        in_samples.append(num_dir_in)
        out_samples.append(num_dir_out)
        fig = plt.figure()

    plt.bar(LETTERS,out_samples, width=0.9, align='center',color = '#3280C4', alpha=0.6)# the widest bar encompasses the other two
    plt.bar(LETTERS,in_samples, width=0.8)  # the widest bar encompasses the other two
    plt.suptitle('Number Of Images Per Letter In The Cleaned Train Set')
    plt.axhline(np.mean(in_samples), color='#59C9D9')
    plt.annotate('Mean',(9,np.mean(in_samples)+0.35), fontsize=9)
    plt.axhline(np.median(in_samples), color='#62DFBB')
    plt.annotate('Median',(9,np.median(in_samples)+0.5),fontsize=9)
    plt.show()
    print('total number of images: ', total_in)
    print('Mean: ',total_in/10)
    print('Median: ',np.median(in_samples))
    print('Min: ', np.min(in_samples))
    print('Max: ', np.max(in_samples))

# hist_num_sapmles('data_orig/train')
# hist_num_sapmles('data_noaug/train')
# hist_num_sapmles('data/train')
