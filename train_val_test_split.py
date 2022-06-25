import itertools
import os

import numpy as np

import random
from pathlib import Path
import warnings

DATA_PATH = './data_noaug' ##SHOULD BE ./data
ROMAN_LETTERS = ['i','ii','iii','iv','v','vi','vii','viii','ix','x']
def get_picture_dict():
    picture_dict = {roman_let: [] for roman_let in ROMAN_LETTERS}
    for roman_let in ROMAN_LETTERS:
        for mid_dir in ['train', 'maybe']:
            path = os.path.join(DATA_PATH, mid_dir, roman_let)
            if os.path.isdir(path):
                picture_dict[roman_let] += [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    return picture_dict


def move_picture(old_path, set_name, roman_let):
    if 'maybe' in old_path and set_name == 'train':
        dest_dir = os.path.join(DATA_PATH, 'maybe', roman_let)
    else:
        dest_dir = os.path.join(DATA_PATH, set_name, roman_let)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, os.path.basename(old_path))
    if os.path.isfile(old_path):
        os.rename(old_path, dest_path)
    else:
        warnings.warn(f"couldn't find file: {old_path}")


def split_train_val_test(val_ratio=0.2, test_ratio=0.1):
    old_n = count_n_pics()
    picture_dict = get_picture_dict()
    for roman_let, pics in picture_dict.items():
        random.shuffle(pics)
        test_idx = int(np.ceil(len(pics) * test_ratio))
        val_idx = int(np.ceil(len(pics) * val_ratio + test_idx))
        test, val, train = np.split(pics, [test_idx, val_idx])
        for set_name, set_pics in zip(['test', 'val', 'train'], [test, val, train]):
            for old_path in set_pics:
                move_picture(old_path, set_name, roman_let)
    assert_split(old_n)
                
                
def get_set_pic_dict(sets=('maybe', 'test', 'train', 'val')):
    set_pic_dict = {set_name: [] for set_name in sets}
    for set_name in sets:
        for roman_let in ROMAN_LETTERS:
            path = os.path.join(DATA_PATH, set_name, roman_let)
            if os.path.isdir(path):
                set_pic_dict[set_name] += [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        set_pic_dict[set_name] = set(set_pic_dict[set_name])
    return set_pic_dict


def assert_split_is_disjoint(set_pic_dict, sets=('maybe', 'test', 'train', 'val')):
    for set1, set2 in itertools.combinations(sets, 2):
        assert set_pic_dict[set1].isdisjoint(set_pic_dict[set2]), f"{set1} and {set2} are not disjoint"


def assert_no_picture_was_lost(old_n):
    new_n = count_n_pics()
    assert new_n == old_n, f"There were {old_n} pics before split, but after split there were {new_n}"


def assert_split(old_n, sets=('maybe', 'test', 'train', 'val')):
    set_pic_dict = get_set_pic_dict(sets)
    assert_split_is_disjoint(set_pic_dict, sets)
    if old_n is not None:
        assert_no_picture_was_lost(old_n)


def count_n_pics():
    n_pics = 0
    sets = os.listdir(DATA_PATH)
    for set_name in sets:
        for roman_let in ROMAN_LETTERS:
            path = os.path.join(DATA_PATH, set_name, roman_let)
            if os.path.isdir(path):
                n_pics += len([os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    return n_pics


if __name__ == '__main__':
    print('Start')
    split_train_val_test()
    print('end')

