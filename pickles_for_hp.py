import pickle
import os


LETTERS = ['i','ii','iii','iv','v','vi','vii','viii','x','ix']
letters_info = {}
train_path = 'data/train'
for letter in LETTERS:
    letter_info = {'num_orig': 0, 'orig': [], 'flipped': [], 'gan': [], 'shift': []}
    all_images = os.listdir(f'{train_path}/{letter}')
    for image in all_images:
        if 'img' in image:
            letter_info['gan'].append(image)
        elif 'shift' in image:
            letter_info['shift'].append(image)
        elif 'flipped' in image:
            letter_info['flipped'].append(image)
        elif '.png' in image:
            letter_info['orig'].append(image)
    letter_info['num_orig'] = len(letter_info['orig'])
    letters_info[letter]=letter_info
#%%

with open('letters_info.pkl','wb') as file:
    pickle.dump(letters_info,file)
