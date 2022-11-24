import numpy as np
import cv2
import os


train_split = 0.8
valid_split = 0.1
test_split = 0.1

full_folder = './dataset/full/'
train_folder = './dataset/train_set/'
valid_folder = './dataset/valid_set/'
test_folder = './dataset/test_set/'

dirs = os.listdir(full_folder)
total_imgs = len(dirs)

num_train_images = int(total_imgs * train_split)
num_valid_images = int(total_imgs * valid_split)
num_test_images = int(total_imgs * test_split)


train_dirs = dirs[:num_train_images]
valid_dirs = dirs[num_train_images:num_train_images+num_valid_images]
test_dirs = dirs[num_train_images+num_valid_images:]

# move train dirs to train folder
for dir in train_dirs:
    full_dir = full_folder + dir
    train_dir = train_folder + dir
    os.rename(full_dir, train_dir)

#move valid dirs to valid folder
for dir in valid_dirs:
    full_dir = full_folder + dir
    valid_dir = valid_folder + dir
    os.rename(full_dir, valid_dir)

#move test dirs to test folder
for dir in test_dirs:
    full_dir = full_folder + dir
    test_dir = test_folder + dir
    os.rename(full_dir, test_dir)