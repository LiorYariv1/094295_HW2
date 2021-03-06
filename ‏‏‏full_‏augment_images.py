import os
import torchvision
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian
import matplotlib.pyplot as plt



Roman_to_Numeric = {'i':1,'ii':2,'iii':3,'iv':4,'v':5,'vi':6,'x':10}
Numeric_to_Roman = {v:k for k,v in Roman_to_Numeric.items()}


def augment_digit(cur_digit, path, files_list):
    dest = cur_digit
    source_path = os.path.join(path,cur_digit)
    dest_path = os.path.join(path,dest)
    for image_name in files_list:
        if '.png' in image_name:
            image_path = os.path.join(source_path,image_name)
            im = Image.open(image_path) # reading the image using its path

            # define limits variables
            translate_random_var = [5, 25]
            angle_random_var = [-10, 10]
            scale_random_var = [0.91, 1]
            shear_random_var = [-5, 5]

            # apply operations
            for m1, m2 in [(1,1),(1,-1),(-1,1),(-1,-1)]:

                up_high = max(m1 * translate_random_var[0], m1 * translate_random_var[1])
                up_low = min(m1 * translate_random_var[0], m1 * translate_random_var[1])
                up = np.random.uniform(low=up_low, high=up_high, size=(1)).astype(float)[0]
                left_high = max(m2 * translate_random_var[0], m2 * translate_random_var[1])
                left_low = min(m2 * translate_random_var[0], m2 * translate_random_var[1])
                left = np.random.uniform(low=left_low, high=left_high, size=(1)).astype(float)[0]
                scl = np.random.uniform(low=scale_random_var[0], high=scale_random_var[1], size=(1)).astype(float)[0]
                ang = np.random.uniform(low=angle_random_var[0], high=angle_random_var[1], size=(1)).astype(float)[0]
                shr = np.random.uniform(low=shear_random_var[0], high=shear_random_var[1], size=(1)).astype(float)[0]

                newim = torchvision.transforms.functional.affine(im, angle=ang, translate=[up, left], scale=scl,
                                                                   shear=[shr], fillcolor=255)
                newim.save(os.path.join(dest_path, image_name.split('.png')[0] + f'_shift_{m1}{m2}_3.png'))


if __name__ == "__main__":
    DIGITS = ['i','ii','iii','iv','v','vi','x', 'vii', 'viii', 'ix']
    print("start building lists")
    LETTERS = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'x', 'ix']
    letters_info = {}
    train_path = 'data/train'
    for letter in LETTERS:
        print(letter)
        letter_info = {'num_orig': 0, 'orig': [], 'flipped': [], 'gan': [], 'shift': []}
        all_images = os.listdir(f'{train_path}/{letter}')
        for image in all_images:
            if 'img' in image:
                letter_info['gan'].append(image)
            elif 'shift' in image:
                letter_info['shift'].append(image)
            elif 'flipped' in image:
                letter_info['flipped'].append(image)
            elif '.png' in image:
                letter_info['orig'].append(image)
        letter_info['num_orig'] = len(letter_info['orig'])
        letters_info[letter] = letter_info
    print("lists done")

    path = 'data/train'
    for digit in LETTERS:
        print('shifting digit ',digit)
        files_list = letters_info[digit]['orig']+letters_info[digit]['flipped']
        augment_digit(digit,path,files_list)

