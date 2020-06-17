import os
import json
import urllib.request as ur
import shutil
import cv2

def json2annotated(data):

    # Read the video from the specified path
    with open(data) as json_file:
        raw_json = json.load(json_file)

    output_dir = data[:len(data) - 5]

    try:
        # Create an appropriately named folder
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # If failure to create the file should occur, send out an error message
    except OSError:
        print('Error: was not able to create file')

    for image in raw_json:

        try:
            # get info from raw_json file
            raw_url = image["Labeled Data"]
            image_id = image["External ID"][:-4]
            mask_info = image["Label"]["objects"]

            # create temporary dir
            if not os.path.exists("annotated/temp"):
                os.makedirs("annotated/temp")

            # get raw image
            raw_dir = "annotated/temp/raw_{}.png".format(image_id)
            ur.urlretrieve(raw_url, raw_dir)

            # empty coordinates
            sum_contours = list()

            # # If there is more than one annotation in an image
            # if len(mask_info) > 1:
            #     # overlay images
            counter = 1

            for sub_image in mask_info:
                # get masked image
                mask_dir = "annotated/temp/{}_{}.png".format(image_id, counter)
                mask_url = sub_image["instanceURI"]
                ur.urlretrieve(mask_url, mask_dir)

                # get contour coordinates of mask image
                img = cv2.imread(mask_dir)
                imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(imgray, 127, 255, 0)
                contours, hierachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # save contour coordinates
                sum_contours += contours
                counter += 1

            # overlay contour line on the raw image
            raw = cv2.imread(raw_dir)
            output_im = cv2.drawContours(raw, sum_contours, -1, (0, 255, 0), 2)
            cv2.imwrite("{}/{}.png".format(output_dir, image_id), output_im)
            print("Generating annotated image {}...".format(image_id))

        except IndexError:
            continue

    # remove directory 'temp'
    shutil.rmtree("annotated/temp")
    # cv2.imshow('image', output_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()