import sys
import os
import pathlib
import math
import cv2 as cv
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import operator
import matplotlib.pyplot as plt

# pre-processing, convert to grayscale, adjust brightness (if average brightness is less than 0.4, increase brightness; if average brightness is greater than 0.6, reduce brightness). Resize the image to TWO different sizes: 200*200 and 50*50 and save them
def pre_process(folder):

    for image_name in os.listdir(folder):
        prefix = "pre_processed\\"
        image = cv.imread(folder + "\\" + image_name)
        
        if image is None:
            print ('Error opening image!')
            return -1

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray = cv.resize(gray, (200, 200))
        gray = np.array(gray, dtype=np.float32)
        gray = gray / 255.0
        avg = np.average(gray)
        if avg < 0.4:
            gray = gray * 1.5
        elif avg > 0.6:
            gray = gray * 0.5
        gray = np.array(gray * 255, dtype=np.uint8)
        cv.imwrite(prefix + "200x200_" + image_name, gray)
        gray = cv.resize(gray, (50, 50))
        cv.imwrite(prefix + "50x50_" + image_name, gray)

        # check
        # print(image_name + " is done")

    return 0


def main(argv):
    # folder path
    path = str(pathlib.Path(__file__).parent.resolve())
    suffix = "\\ProjData\\Test\\bedroom"
    folder = path + suffix
    print(folder)

    # make folder for pre-processed images
    if not os.path.exists(path + "\\pre_processed"):
        os.makedirs(path + "\\pre_processed")

    # pre-processing
    pre_process(folder)

    # new folder path
    suffix = "\\pre_processed"
    folder = path + suffix

    print(folder)
    # extract SIFT features
    # extract_sift(folder)

    # sift test
    # img_array = np.load('sift_features\\50x50_image_0003.jpg.npy')
    # plt.imshow(img_array, cmap='gray')
    # plt.show() 

    # extract histogram features
    # extract_histogram(folder)
    # extract_histogram_v2(folder)

    print("is done")
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])