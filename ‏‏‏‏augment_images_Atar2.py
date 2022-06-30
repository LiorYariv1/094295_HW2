import os
import torchvision
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# # importing all the required libraries
# import warnings
# warnings.filterwarnings('ignore')
# import numpy as np
# import skimage.io as io
# from skimage.transform import rotate, AffineTransform, warp
# from skimage.util import random_noise
# from skimage.filters import gaussian
# import matplotlib.pyplot as plt



Roman_to_Numeric = {'i':1,'ii':2,'iii':3,'iv':4,'v':5,'vi':6,'x':10}
Numeric_to_Roman = {v:k for k,v in Roman_to_Numeric.items()}


def augment_digit(cur_digit, path):
    dest = cur_digit if cur_digit not in ['iv', 'vi'] else 'iv' if cur_digit == 'vi' else 'vi'
    source_path = os.path.join(path,cur_digit)
    dest_path = os.path.join('atarTry',dest)
    for image_name in os.listdir(source_path):
        if '.png' in image_name:
            image_path = os.path.join(source_path,image_name)
            image = Image.open(image_path)

            # apply rotate operation
            max_rotate = 10
            ang = np.random.uniform(low=-max_rotate, high=max_rotate, size=(1)).astype(float)[0]
            # print(f'ang {ang}')
            # rotated = rotate(image, angle=ang, mode='wrap')
            rotated = torchvision.transforms.functional.rotate(image, ang)

            # apply shift operation
            max_shift = 3.5
            up = np.random.uniform(low=-max_shift, high=max_shift, size=(1)).astype(float)[0]
            left = np.random.uniform(low=-max_shift, high=max_shift, size=(1)).astype(float)[0]
            # print(f'up {up},  left {left}')
            # transform = AffineTransform(translation=(up, left))
            # wrapShift = warp(rotated, transform, mode='wrap')

            # flipped = torchvision.transforms.functional.hflip(image)
            wrapShift.save(os.path.join(dest_path,image_name.split('.png')[0]+'_shift.png'))
            plt.imsave(os.path.join(dest_path,image_name.split('.png')[0]+'_shift.png'), wrapShift)


if __name__ == "__main__":
    DIGITS = ['i','ii','iii','iv','v','vi','x']
    path = 'data/train'
    for digit in DIGITS:
        print('shifting digit ',digit)
        augment_digit(digit,path)

