import cv2
import numpy as np
import os

DIR = 'trypanosome_data/experiment_5'

sets = ['set1', 'set2', 'set3', 'set4', 'set5']
type = ['train', 'valid', 'test']
Class = ['pos', 'neg']

sharpen_filter = np.array([[-1, -1, -1, -1, -1],
                             [-1, 2, 2, 2, -1],
                             [-1, 2, 8, 2, -1],
                             [-1, 2, 2, 2, -1],
                             [-1, -1, -1, -1, -1]
                             ])/8.0


for set in sets:

    print("Sharpening {}...".format(set))

    SAVE_DIR = 'Sharpened/{}'.format(set)

    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    for x in type:

        x_IMG_DIR = '{}/{}/{}'.format(DIR, set, x)
        x_SAVE_DIR = '{}/{}'.format(SAVE_DIR, x)

        if not os.path.exists(x_SAVE_DIR):
            os.mkdir(x_SAVE_DIR)
            os.mkdir('{}/{}'.format(x_SAVE_DIR, Class[0]))
            os.mkdir('{}/{}'.format(x_SAVE_DIR, Class[1]))

        for c in Class:

            c_IMG_DIR = '{}/{}'.format(x_IMG_DIR, c)
            c_SAVE_DIR = '{}/{}'.format(x_SAVE_DIR, c)
            img_names = os.listdir(c_IMG_DIR)

            for img_name in img_names:
                img = cv2.imread('{}/{}'.format(c_IMG_DIR, img_name))
                sharpened = cv2.filter2D(img, -1, sharpen_filter)
                cv2.imwrite('{}/{}'.format(c_SAVE_DIR, img_name), sharpened)


# cv2.imwrite('sharpen_1.png', sharpened)
# cv2.imwrite('sharpen_3.png', sharpened)
# cv2.waitKey()
