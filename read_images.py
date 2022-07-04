import os
import pickle
import shutil
#%%
with open('/home/student/HW2/wandb/run-20220703_210200-sv9aryh2/files/images_by_letter', 'rb') as file:
    images_by_letter_best_run = pickle.load(file)
#%%
train_dir = os.path.join("data", "train")
val_dir = os.path.join("data", "val")
final_train_dir = '/home/student/HW2/data_final/data/train'
for letter,images_list in images_by_letter_best_run.items():
    letter_path = f'{final_train_dir}/{letter}'
    os.makedirs(letter_path, exist_ok=True)
    letter_dir = f'{train_dir}/{letter}'
    images_list = [f'{letter_dir}/{x}' for x in images_list]
    for image in images_list:
        shutil.copy(image, letter_path)
final_val_dir =  '/home/student/HW2/data_final/data/val'
shutil.copytree(val_dir, final_val_dir)