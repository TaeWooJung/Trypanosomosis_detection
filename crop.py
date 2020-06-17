import os
import cv2
import numpy as np
import shutil
import random


def is_positive(crop_bin_arr, p=0.02):
    # count number of white pixels
    white_count = np.count_nonzero(crop_bin_arr)
    height, width, _ = crop_bin_arr.shape
    total = height * width

    # if white pixels are at least 2%, consider the image as positive crop
    if white_count/total >= p:
        return True

    return False


is_crop = False

x = 'set5/valid'
DIR_img = 'Preprocessed/{}'.format(x)
DIR_binary = 'trypanosome_data/binary_mask'
SAVE_DIR = 'crop_6/{}'.format(x)

# name the list of videos to generate crop images
vid_names = ['vid01', 'vid02', 'vid03', 'vid04', 'vid05', 'vid09']
vid_dict = {}

# set the size of crop image
height, width = 224, 224

if is_crop:

    # make a grid
    max_y = 1024
    max_x = 1360

    step_y = 112
    step_x = 112

    split_y = [0+i*step_y for i in range(max_y) if 0+i*step_y+height <= max_y]
    split_x = [0+i*step_x for i in range(max_x) if 0+i*step_x+width <= max_x]

    crop_points = []

    for y in split_y:
        for x in split_x:
            crop_points.append((y, x))

    print('number of crop images = {}\n'.format(len(crop_points)))

    for vid_name in vid_names:
        img_names = [i for i in os.listdir('{}'.format(DIR_img)) if vid_name in i]
        vid_dict[vid_name] = img_names

    for vid_name in vid_dict:

        # make necessary directories
        if not os.path.isdir(SAVE_DIR):
            os.makedirs(SAVE_DIR)
            os.makedirs('{}/pos'.format(SAVE_DIR))
            os.makedirs('{}/neg'.format(SAVE_DIR))
            os.makedirs('{}/bin'.format(SAVE_DIR))

        for img_name in vid_dict[vid_name]:

            print('Cropping {}...'.format(img_name))

            img_arr = cv2.imread('{}/{}'.format(DIR_img, img_name))
            bin_arr = cv2.imread('{}/{}/{}'.format(DIR_binary, vid_name, img_name))

            n = 1   # counter
            for crop_point in crop_points:

                starty, startx = crop_point
                crop_bin_arr = bin_arr[starty:starty + height, startx:startx + width]
                crop_img_arr = img_arr[starty:starty + height, startx:startx + width]

                # If ROI >= 0.2, patch is labeled as positive patch
                if is_positive(crop_bin_arr):
                    cv2.imwrite('{0}/pos/{1}_{2}.png'.format(SAVE_DIR, img_name[:-4], n), crop_img_arr)

                # If ROI < 0.2, patch is labeled as negative patch
                else:
                    cv2.imwrite('{0}/neg/{1}_{2}.png'.format(SAVE_DIR, img_name[:-4], n), crop_img_arr)

                cv2.imwrite('{0}/bin/{1}_{2}.png'.format(SAVE_DIR, img_name[:-4], n), crop_bin_arr)
                n += 1

    print('Finished cropping...!')


# info about crop_5_test
# vid01 - 13%
# vid02 - 7.75%
# vid03 - 5%
# vid04 - 20%
# vid05 - 13.5%
# vid09 - 5.5%

# p <= 0.02 considered as negative patches

p = 0.05

vid_names = ['vid01', 'vid02', 'vid03', 'vid04', 'vid05', 'vid09']
sharpen = True

p_vids = {'vid01': 0.08,
          'vid02': 0.0585,
          'vid03': 0.045,
          'vid04': 0.1465,
          'vid05': 0.125,
          'vid09': 0.047}

'''
# ROI thresholds for T1, T2, T3, T4, T5
T1
p_vids = {'vid01': 0.055,
          'vid02': 0.036,
          'vid03': 0.028,
          'vid04': 0.093,
          'vid05': 0.094,
          'vid09': 0.024}
          
T2
p_vids = {'vid01': 0.06,
          'vid02': 0.04,
          'vid03': 0.032,
          'vid04': 0.103,
          'vid05': 0.1,
          'vid09': 0.03}
          
T3
p_vids = {'vid01': 0.065,
          'vid02': 0.045,
          'vid03': 0.036,
          'vid04': 0.113,
          'vid05': 0.1075,
          'vid09': 0.036}

T4
p_vids = {'vid01': 0.0713,
          'vid02': 0.051,
          'vid03': 0.04,
          'vid04': 0.125,
          'vid05': 0.115,
          'vid09': 0.04}

T5
p_vids = {'vid01': 0.08,
          'vid02': 0.0585,
          'vid03': 0.045,
          'vid04': 0.1465,
          'vid05': 0.125,
          'vid09': 0.047}
'''

x = 'valid'
set = 'set1'

for vid_name in vid_names:

    print('Thresholding {}...'.format(vid_name))

    p = 0.05
    POS_DIR = 'crop_6/{}/{}/pos'.format(set, x)
    NEG_DIR = 'crop_6/{}/{}/neg'.format(set, x)
    POS_SAVE = 'trypanosome_data/experiment_6/{}/{}/pos'.format(set, x)
    NEG_SAVE = 'trypanosome_data/experiment_6/{}/{}/neg'.format(set, x)

    if not os.path.exists('trypanosome_data/experiment_6/{}'.format(set)):
        os.mkdir('trypanosome_data/experiment_6/{}'.format(set))

    if not os.path.exists('trypanosome_data/experiment_6/{}/{}'.format(set, x)):
        os.mkdir('trypanosome_data/experiment_6/{}/{}'.format(set, x))
        os.mkdir(POS_SAVE)
        os.mkdir(NEG_SAVE)

    pos_img_names = [i for i in os.listdir(POS_DIR) if vid_name in i]
    neg_img_names = [i for i in os.listdir(NEG_DIR) if vid_name in i]

    final_pos_img = list()

    for img_name in pos_img_names:

        BIN_DIR = 'crop_6/bin/{}'.format(img_name)
        bin_crop = cv2.imread(BIN_DIR)

        if is_positive(bin_crop, p):
            final_pos_img.append(img_name)

    # make balance dataset
    if len(final_pos_img) < len(neg_img_names):
        random.shuffle(neg_img_names)
        neg_img_names = neg_img_names[:len(final_pos_img)]

    else:
        random.shuffle(final_pos_img)
        final_pos_img = final_pos_img[:len(neg_img_names)]

    for i in range(len(final_pos_img)):
        POS_IMG_DIR = '{}/{}'.format(POS_DIR, final_pos_img[i])
        NEG_IMG_DIR = '{}/{}'.format(NEG_DIR, neg_img_names[i])
        shutil.copy(POS_IMG_DIR, POS_SAVE)
        shutil.copy(NEG_IMG_DIR, NEG_SAVE)

print('Done!')


