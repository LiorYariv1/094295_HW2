import os
import pickle
import shutil
#%%
with open('/home/student/HW2/wandb/run-20220703_042738-fc7q80r7/files/images_by_letter', 'rb') as file:
    images_by_letter_best_run = pickle.load(file)
#%%
train_dir = os.path.join("data", "train")
final_train_dir = '/home/student/HW2/data/train_final_try'
for letter,images_list in images_by_letter_best_run.items():
    letter_path = f'{final_train_dir}/{letter}'
    os.makedirs(letter_path, exist_ok=True)
    letter_dir = f'{train_dir}/{letter}'
    images_list = [f'{letter_dir}/{x}' for x in images_list]
    for image in images_list:
        shutil.copy(image, letter_path)