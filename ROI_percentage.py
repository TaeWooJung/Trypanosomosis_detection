import numpy as np
import cv2
import os

def is_positive(crop_bin_arr):
    # count number of white pixels
    white_count = np.count_nonzero(crop_bin_arr)
    height, width, _ = crop_bin_arr.shape
    total = height * width

    return white_count/total

DIR_NEG = 'trypanosome_data/crop_5/original/'
VIDS = ['vid01', 'vid02', 'vid03', 'vid04', 'vid05', 'vid06']

for vid in VIDS:
    VID_DIR = os.path.join(DIR_NEG, '{}/bin'.format(vid))
    img_names = os.listdir(VID_DIR)
    ROI = list()

    for img in img_names:
        IMG_DIR = os.path.join(VID_DIR, img)
        crop_bin_arr = cv2.imread(IMG_DIR)
        ROI.append(is_positive(crop_bin_arr)*100)

    print(ROI)


