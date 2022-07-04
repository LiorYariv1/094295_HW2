import os
import torchvision
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
from PIL import Image


Roman_to_Numeric = {'i':1,'ii':2,'iii':3,'iv':4,'v':5,'vi':6,'x':10}
Numeric_to_Roman = {v:k for k,v in Roman_to_Numeric.items()}


def augment_digit(cur_digit, path):
    dest = cur_digit if cur_digit not in ['iv', 'vi'] else 'iv' if cur_digit == 'vi' else 'vi'
    source_path = os.path.join(path,cur_digit)
    dest_path = os.path.join(path,dest)
    for image_name in os.listdir(source_path):
        if '.png' in image_name:
            image_path = os.path.join(source_path,image_name)
            image = Image.open(image_path)
            flipped = torchvision.transforms.functional.hflip(image)
            flipped.save(os.path.join(dest_path,image_name.split('.png')[0]+'_flipped.png'))


if __name__ == "__main__":
    DIGITS = ['i','ii','iii','iv','v','vi','x']
    path = 'data/train'
    for digit in DIGITS:
        print('flipping digit ',digit)
        augment_digit(digit,path)

