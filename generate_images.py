import numpy as np
import os
import torch
import tqdm
import cs236781.plot as plot
from PIL import Image
from torchvision.utils import save_image


LETTERS = ['i','ii','iii','iv','v','vi','vii','viii','x','ix']
best_z_dim = [112,128,256,112,112,256,112,256,112,112]
root= '/home/student/HW2/checkpoints/'
models_paths = {112:f'{root}gan',128:f'{root}gan_2',256:f'{root}gan_z_dim_256'}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for i,letter in enumerate(LETTERS):
    if letter in ['i','ii','iii']:
        continue
    print('-'*10, 'running letter', letter, '*'*10)
    model_directory = f'{models_paths[best_z_dim[i]]}/{letter}'
    all_models = os.listdir(model_directory)
    if letter =='i':
        all_models = [x.split('.')[0] for x in all_models]
        all_models = np.array([int(x[1:]) for x in all_models if 'i' in x])
        best_model = f'i{np.max(all_models)}'
    else:
        all_models = np.array([int(x.split('.')[0]) for x in all_models])
        best_model = np.max(all_models)
    number_of_images = 1000
    best_model_path = os.path.join(model_directory,f'{best_model}.pt')
    print(f'*** Loading final checkpoint file from epoch {best_model_path}')
    gen = torch.load(best_model_path, map_location=device)
    samples = gen.sample(number_of_images, with_grad=False)

    images_path = f'Final_images/{letter}'
    os.makedirs(images_path)
    for i,sample in enumerate(samples):
        save_image(sample, f'{images_path}/img{i}.png')
