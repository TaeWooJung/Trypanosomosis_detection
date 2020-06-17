import numpy as np
import cv2
import os

vid_name = 'vid05'
DIR = 'trypanosome_data/new_annotation/{}'.format(vid_name)
SAVE_DIR = 'trypanosome_data/binary_mask/{}'.format(vid_name)

img_names = os.listdir(DIR)
rm_arr = cv2.imread('black2.png')

for img_name in img_names:
    print('Converting {} into binary image...'.format(img_name))
    img_arr = cv2.imread('{}/{}'.format(DIR, img_name))
    # select all pixels that are not white
    mask = (img_arr != [255., 255., 255.]).all(axis=2)
    # convert all non-white pixels to black
    img_arr[mask] = [0, 0, 0]
    # remove noise at the bottom right corner
    img_arr[1024-rm_arr.shape[0]:1024, 1360-rm_arr.shape[1]:1360] = rm_arr
    # make sure save image as png
    cv2.imwrite('{}/{}.png'.format(SAVE_DIR, img_name[:-4]), img_arr)

print('Done!')




