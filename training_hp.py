import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import wandb
import torchvision
from torchvision import models, transforms, datasets
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import time
import copy
from tqdm import tqdm
import random
import logging
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import shutil


def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune_name', default="first")

    parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online', type=str)
    parser.add_argument('--project', default="Lab2", type=str)

    parser.add_argument('--gan_pct', default=0.33, type=float)
    parser.add_argument('--train_c_size', default=18, type=int)
    # parser.add_argument('--shift_pct', default=0.33, type=float)

    args = parser.parse_args()
    return args

# ======================================================
# ======================================================
# ======================================================
# ======================================================

# You are not allowed to change anything in this file.
# This file is meant only for training and saving the model.
# You may use it for basic inspection of the model performance.

# ======================================================
# ======================================================
# ======================================================
# ======================================================



# class ImageDataSet(Dataset):
#     def __init__(self, images_files, transform=None):
#         self.image_files =images_files
#         self.transform = transform
#         self.class_to_idx={'i': 0, 'ii': 1, 'iii': 2, 'iv': 3, 'ix': 4, 'v': 5, 'vi': 6, 'vii': 7, 'viii': 8, 'x': 9}
#         self.classes = ['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x']
#     def __len__(self):
#         # Here, we need to return the number of samples in this dataset.
#         return len(self.image_files)
#
#     def __getitem__(self, index):
#         file_path = self.image_files[index]
#         image = Image.open(file_path)
#         image = image.convert('RGB')
#         transformed_image = self.transform(image)
#         label = self.class_to_idx[file_path.split("/")[-2]]
#         return transformed_image, label


def imshow(inp, title=None):
    """Imshow for Tensors."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(15, 15))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=100):
    """Responsible for running the training and validation phases for the requested model."""
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    loss_dict = {'train': [], 'val': []}
    acc_dict = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                # print("################################################")
                # print(f'running_loss={running_loss}')
                # print(f'inputs.size(0)={inputs.size(0)}')
                # print(f'loss.item()={loss.item()}')
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            acc_dict[phase].append(epoch_acc.item())
            loss_dict[phase].append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            results = {f'{phase}_loss':epoch_loss,f'{phase}_accuracy':epoch_acc,f'{phase}_epoch':epoch}
            wandb.log(results)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    wandb.log({f'best_val_acc': best_acc,'time':f'{time_elapsed // 60}:{time_elapsed % 60}'})  # log best results

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, loss_dict, acc_dict


if __name__=='__main__':
    # Training hyperparameters

    print("Your working directory is: ", os.getcwd())
    logger = logging.getLogger(__name__)

    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LR = 0.001

    # Paths to your train and val directories
    train_dir = os.path.join("data", "train")
    val_dir = os.path.join("data", "val")
    train_dir_tmp = os.path.join("data", "train_tmp")

    args = parsing()
    # random.seed(42)
    # args.shift_pct = 1-args.gan_pct - args.flip_pct
    # logger.info(args)
    with open('letters_info.pkl', 'rb') as file:
        letter_info = pickle.load(file)


    wandb.init(project=args.project, entity="labteam",mode=args.wandb_mode, ) #logging to wandb
    wandb.config.update(args)
    train_c_size = args.train_c_size*50
    train_images = []
    all_images = []
    images_by_letter ={}
    for letter in letter_info.keys():
        original_size = letter_info[letter]['num_orig'] + len(letter_info[letter]['flipped']) #orig+flipped size
        total_letter_list = (letter_info[letter]['orig']+letter_info[letter]['flipped']).copy()
        gan_num = int((train_c_size-original_size)*args.gan_pct)
        gan_num = min(gan_num, len(letter_info[letter]['gan']))
        gan_sample = random.sample(letter_info[letter]['gan'], gan_num)
        # flip_num = int((train_c_size-original_size)*args.flipped_pct)
        # flip_sample = random.sample(letter_info['flipped'], flip_num)
        shift_sample = random.sample(letter_info[letter]['shift'], train_c_size-gan_num-original_size)
        total_letter_list = total_letter_list+gan_sample+shift_sample
        all_images += total_letter_list
        images_by_letter[letter] = total_letter_list
        # total_letter_list = []
        # for t in types:
        #     total_letter_list += letter_info[letter][t]#+letter_info[letter]['shift']
        letter_dir = f'{train_dir}/{letter}'
        images_by_letter[letter] = total_letter_list
        total_letter_list = [f'{letter_dir}/{x}' for x in total_letter_list]
        letter_path = f'{train_dir_tmp}/{letter}'
        os.makedirs(letter_path, exist_ok=True)
        for image in total_letter_list:
            shutil.copy(image,letter_path)
        train_images += total_letter_list
        print(f'number of total images for letter {letter}:',len(total_letter_list))
        print(f'number of orig+flipped images for letter {letter}:',original_size)
        print(f'number of gan images for letter {letter}:',gan_num)
        print(f'number of shift images for letter {letter}:',train_c_size-gan_num-original_size)


    with open(f'{wandb.run.dir}/images_by_letter', 'wb') as file:
         pickle.dump(images_by_letter,file)

    # Resize the samples and transform them into tensors
    data_transforms = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])

    torch.manual_seed(0)

    train_dataset = datasets.ImageFolder(train_dir_tmp, data_transforms)
    val_dataset = datasets.ImageFolder(val_dir, data_transforms)

    class_names = train_dataset.classes
    print("The classes are: ", class_names)

    # Dataloaders initialization
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # for i in range(3):
    #     inputs, classes = next(iter(train_dataloader))
    #     out = torchvision.utils.make_grid(inputs)
    #     imshow(out, title=[class_names[x] for x in classes])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    NUM_CLASSES = len(class_names)

    # Use a prebuilt pytorch's ResNet50 model
    model_ft = models.resnet50(pretrained=False)

    # Fit the last layer for our specific task
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.Adam(model_ft.parameters(), lr=LR)

    # Decay LR by a factor of 0.1 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

    # Train the model
    model_ft, loss_dict, acc_dict = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders,
                                                dataset_sizes, num_epochs=NUM_EPOCHS)

    # Save the trained model
    torch.save(model_ft.state_dict(), f'{wandb.run.dir}/trained_model.pt')
    wandb.finish()
    shutil.rmtree(train_dir_tmp)
    # Basic visualizations of the model performance
    fig = plt.figure(figsize=(20, 10))
    plt.title("Train - Validation Loss")
    plt.plot(loss_dict['train'], label='train')
    plt.plot(loss_dict['val'], label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend(loc='best')
    plt.savefig('train_val_loss_plot.png')

    fig = plt.figure(figsize=(20, 10))
    plt.title("Train - Validation ACC")
    plt.plot(acc_dict['train'], label='train')
    plt.plot(acc_dict['val'], label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('ACC', fontsize=12)
    plt.legend(loc='best')
    plt.savefig('train_val_acc_plot.png')