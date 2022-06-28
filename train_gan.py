import unittest
import os
import sys
sys.path.append('/home/student/HW2')
import pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
import gan
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from answers import part3_gan_hyperparams
from gan import train_batch, save_checkpoint
import IPython.display
import tqdm
import cs236781.plot as plot
from PIL import Image
from torchvision.utils import save_image


# Optimizer
def create_optimizer(model_params, opt_params):
    opt_params = opt_params.copy()
    optimizer_type = opt_params['type']
    opt_params.pop('type')
    return optim.__dict__[optimizer_type](model_params, **opt_params)



# Loss
def dsc_loss_fn(y_data, y_generated):
    return gan.discriminator_loss_fn(y_data, y_generated, hp['data_label'], hp['label_noise'])


def gen_loss_fn(y_generated):
    return gan.generator_loss_fn(y_generated, hp['data_label'])
#%%
class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = [x for x in os.listdir(main_dir) if '.png' in x]

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image,1

#%%
LETTER = 'ix'
dataset_dir = f'/home/student/HW2/data/train/{LETTER}'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
tf = T.Compose([
    T.Resize((64, 64)),
    # PIL.Image -> torch.Tensor
    T.ToTensor(),
    # Dynamic range [0,1] -> [-1, 1]
    T.Normalize(mean=(.5,.5,.5), std=(.5,.5,.5)),
])
# ds_gwb = ImageFolder(os.path.dirname(dataset_dir), tf)
ds_gwb = CustomDataSet(dataset_dir, tf)
x0, y0 = ds_gwb[0]
#%%

torch.manual_seed(42)

# Hyperparams
hp = part3_gan_hyperparams()
batch_size = hp['batch_size']
z_dim = hp['z_dim']


# Data
dl_train = DataLoader(ds_gwb, batch_size, shuffle=True)
im_size = ds_gwb[0][0].shape

# Model
dsc = gan.Discriminator(im_size).to(device)
gen = gan.Generator(z_dim, featuremap_size=4).to(device)
dsc_optimizer = create_optimizer(dsc.parameters(), hp['discriminator_optimizer'])
gen_optimizer = create_optimizer(gen.parameters(), hp['generator_optimizer'])

#%%

# Training
checkpoint_file = f'checkpoints/gan/{LETTER}/'
os.makedirs(checkpoint_file,exist_ok=True)
checkpoint_file_final = f'{checkpoint_file}_final'
if os.path.isfile(f'{checkpoint_file}.pt'):
    os.remove(f'{checkpoint_file}.pt')

num_epochs = 100
try:
    dsc_avg_losses, gen_avg_losses = [], []
    for epoch_idx in range(num_epochs):
        # We'll accumulate batch losses and show an average once per epoch.
        dsc_losses, gen_losses = [], []
        print(f'--- EPOCH {epoch_idx + 1}/{num_epochs} ---')

        with tqdm.tqdm(total=len(dl_train.batch_sampler), file=sys.stdout) as pbar:
            for batch_idx, (x_data, _) in enumerate(dl_train):
                x_data = x_data.to(device)
                dsc_loss, gen_loss = train_batch(
                    dsc, gen,
                    dsc_loss_fn, gen_loss_fn,
                    dsc_optimizer, gen_optimizer,
                    x_data)
                dsc_losses.append(dsc_loss)
                gen_losses.append(gen_loss)
                pbar.update()

        dsc_avg_losses.append(np.mean(dsc_losses))
        gen_avg_losses.append(np.mean(gen_losses))
        print(f'Discriminator loss: {dsc_avg_losses[-1]}')
        print(f'Generator loss:     {gen_avg_losses[-1]}')

        if save_checkpoint(gen, dsc_avg_losses, gen_avg_losses, checkpoint_file+str(epoch_idx)):
            best_model = epoch_idx
            print(f'Saved checkpoint.')

        samples = gen.sample(5, with_grad=False)
        fig, _ = plot.tensors_as_images(samples.cpu(), figsize=(6, 2))
        fig.suptitle(f'batch number {epoch_idx}')
        plt.show()
except KeyboardInterrupt as e:
    print('\n *** Training interrupted by user')
# %%

## CODE TO GENERATE AND SAVE IMAGES from best model

number_of_images = 100
print(f'*** Loading final checkpoint file from epoch {best_model}')
gen = torch.load(f'{checkpoint_file}{best_model}.pt', map_location=device, )
samples = gen.sample(number_of_images, with_grad=False)

images_path = f'Images/{LETTER}'
os.makedirs(images_path)
for i,sample in enumerate(samples):
    save_image(sample, f'{images_path}/img{i}.png')
