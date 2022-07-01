import os
import torchvision
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
from PIL import Image

# importing all the required libraries
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


def augment_digit(cur_digit, path):
    dest = cur_digit
    source_path = os.path.join(path,cur_digit)
    dest_path = os.path.join(path,dest)
    for image_name in os.listdir(source_path):
        if '.png' in image_name:
            image_path = os.path.join(source_path,image_name)
            # image = Image.open(image_path)
            image = io.imread(image_path) # reading the image using its path

            # print('Rotated Image')
            # apply rotate operation
            max_rotate = 10
            ang = np.random.uniform(low=-max_rotate, high=max_rotate, size=(1)).astype(float)[0]
            # print(f'ang {ang}')
            rotated = rotate(image, angle=ang, mode='wrap')

            # apply shift operation
            for m1, m2 in [(1,1),(1,-1),(-1,1),(-1,-1)]:
                max_shift = 3.6
                up_high = max(0, m1*max_shift)
                up_low = min(0, m1*max_shift)
                up = np.random.uniform(low=up_low, high=up_high, size=(1)).astype(float)[0]
                left_high = max(0, m2*max_shift)
                left_low = min(0, m2*max_shift)
                left = np.random.uniform(low=left_low, high=left_high, size=(1)).astype(float)[0]
                # print(f'up {up},  left {left}')
                transform = AffineTransform(translation=(up, left))
                wrapShift = warp(rotated, transform, mode='wrap')
                plt.imsave(os.path.join(dest_path, image_name.split('.png')[0] + f'_shift_{str(m1),str(m2)}.png'), wrapShift)


if __name__ == "__main__":
    DIGITS = ['i','ii','iii','iv','v','vi','x', 'vii', 'viii', 'ix']
    path = 'data/train'
    for digit in DIGITS:
        print('shifting digit ',digit)
        augment_digit(digit,path)

