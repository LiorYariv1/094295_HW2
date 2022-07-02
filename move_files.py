import os
import shutil

# os.rename("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
# os.replace("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
# shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")



Roman_to_Numeric = {'i':1,'ii':2,'iii':3,'iv':4,'v':5,'vi':6,'x':10}
Numeric_to_Roman = {v:k for k,v in Roman_to_Numeric.items()}



if __name__ == "__main__":
    DIGITS = ['i','ii','iii','iv','v','vi','x', 'vii', 'viii', 'ix']
    # path = 'data/train'
    # for digit in DIGITS:
    #     print('shifting digit ',digit)
    #     augment_digit(digit,path)

    print("start building lists")
    LETTERS = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'x', 'ix']
    letters_info = {}
    train_path = 'data/train'
    for letter in LETTERS:
        if letter!='vii':
            continue
        print(letter)
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
        letters_info[letter] = letter_info
    print("lists done")

    path = 'atarTry'
    for digit in LETTERS:
        if digit!='vii':
            continue
        print('moving digit ',digit)
        files_list = letters_info[digit]['shift']
        dest = digit
        source_path = os.path.join(path, digit)
        dest_path = os.path.join('oldfiles', dest)
        for image_name in files_list:
            image_path = os.path.join(source_path, image_name)
            dest_path = os.path.join(dest_path, image_name)
            shutil.move(f"{image_path}.foo", f"{dest_path}.foo")


