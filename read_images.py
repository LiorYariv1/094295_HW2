import os
import pickle
import shutil
## This code reads the pickle of the final images, copy them into the final directory and plots histogram of the composition
import matplotlib.pyplot as plt
import pandas as pd

###
"""
This script reads the best images list and copy it into a directory for submission.
"""
###


with open('/home/student/HW2/wandb/run-20220703_210200-sv9aryh2/files/images_by_letter', 'rb') as file:
    images_by_letter_best_run = pickle.load(file)
#%%
train_dir = os.path.join("data_data", "train")
val_dir = os.path.join("data_data", "val")
final_train_dir = os.path.join("data", "train")
final_val_dir = os.path.join("data", "val")

for letter,images_list in images_by_letter_best_run.items():
    letter_path = f'{final_train_dir}/{letter}'
    os.makedirs(letter_path, exist_ok=True)
    letter_dir = f'{train_dir}/{letter}'
    for image in images_list:
        shutil.copy(f'{letter_dir}/{image}', letter_path)

shutil.copytree(val_dir, final_val_dir)
