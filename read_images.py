import os
import pickle
import shutil
## This code reads the pickle of the final images, copy them into the final directory and plots histogram of the composition
import matplotlib.pyplot as plt
import pandas as pd
with open('/home/student/HW2/wandb/run-20220703_210200-sv9aryh2/files/images_by_letter', 'rb') as file:
    images_by_letter_best_run = pickle.load(file)
#%%
train_dir = os.path.join("data", "train")
val_dir = os.path.join("data", "val")
final_train_dir = '/home/student/HW2/data_final/data/train'
letters = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII','VIII','IX','X']
GAN = [0]*10
shift = [0]*10
flipped = [0]*10
orig = [0]*10
for letter,images_list in images_by_letter_best_run.items():
    letter_num = letters.index(str.upper(letter))
    letter_path = f'{final_train_dir}/{letter}'
    # os.makedirs(letter_path, exist_ok=True)
    letter_dir = f'{train_dir}/{letter}'
    # images_list = [f'{letter_dir}/{x}' for x in images_list]
    for image in images_list:
        # shutil.copy(f'{letter_dir}/{image}', letter_path)
        if 'img' in image:
            GAN[letter_num]+=1
        elif 'shift' in image:
            shift[letter_num]+=1
        elif 'flip' in image:
            if letter=='iv':
                print(image)
            flipped[letter_num]+=1
        elif '.png' in image:
            orig[letter_num]+=1

final_val_dir =  '/home/student/HW2/data_final/data/val'
# shutil.copytree(val_dir, final_val_dir)
flipped[8]=0
shift[8]+=1
df = pd.DataFrame({'Letters':letters, 'Original':orig,'Flip':flipped,'Augmentation':shift,'GAN':GAN})
ax = df.plot(x='Letters', kind='bar', stacked=True, rot=1, width=0.7, color=['#3280C4', '#FF9900', '#00b33c', '#7B0099'], title='Copmosition of Current Train Set')
handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[::-1], labels[::-1], title='Line', loc='upper left')
ax.legend(handles[::-1], labels[::-1],bbox_to_anchor=(1.35,1), loc='upper right')
plt.show()