import sys
import os
import pathlib
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import shutil

# pre-processing, convert to grayscale, adjust brightness (if average brightness is less than 0.4, increase brightness; if average brightness is greater than 0.6, reduce brightness). Resize the image to TWO different sizes: 200*200 and 50*50 and save them
def pre_process(folder):
    # print(folder)
    last_two_dirs = os.path.split(os.path.split(folder)[0])[1] + '\\' + os.path.split(folder)[1]
    # print(last_two_dirs)
    for image_name in os.listdir(folder):
        
        prefix = "pre_processed\\" + last_two_dirs
        # print(prefix)
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
    return 0


# Extract SIFT features on ALL training images and save the data
def extract_sift():
    
    # create sift_features folder
    if not os.path.exists("sift_features"):
        os.makedirs("sift_features")

    if not os.path.exists("sift_features\\Train\\reduced_to_50"):
        os.makedirs("sift_features\\Train\\reduced_to_50")

    if not os.path.exists("sift_features\\Train\\reduced_to_200"):
        os.makedirs("sift_features\\Train\\reduced_to_200")

    if not os.path.exists("sift_features\\Test\\reduced_to_50"):
        os.makedirs("sift_features\\Test\\reduced_to_50")

    if not os.path.exists("sift_features\\Test\\reduced_to_200"):
        os.makedirs("sift_features\\Test\\reduced_to_200")


    # sift desc
    if not os.path.exists("sift_descriptors"):
        os.makedirs("sift_descriptors")

    if not os.path.exists("sift_descriptors\\Train\\reduced_to_50"):
        os.makedirs("sift_descriptors\\Train\\reduced_to_50")

    if not os.path.exists("sift_descriptors\\Train\\reduced_to_200"):
        os.makedirs("sift_descriptors\\Train\\reduced_to_200")

    if not os.path.exists("sift_descriptors\\Test\\reduced_to_50"):
        os.makedirs("sift_descriptors\\Test\\reduced_to_50")

    if not os.path.exists("sift_descriptors\\Test\\reduced_to_200"):
        os.makedirs("sift_descriptors\\Test\\reduced_to_200")

    # extract sift features on training images

    for image_name in os.listdir("pre_processed\\Train\\reduced_to_50"):
        image = cv.imread("pre_processed\\Train\\reduced_to_50\\" + image_name)
        if image is None:
            print ('Error opening image!')
            return -1
        
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create()
        kp, des = sift.detectAndCompute(gray_image, None)

        sift_image = cv.drawKeypoints(gray_image, kp, image)
        cv.imwrite("sift_features\\Train\\reduced_to_50\\" + image_name + "_sift.jpg", sift_image)

        # save sift descriptors; .npy file
        # flatten
        if des is not None:
            des = des.flatten()

        np.save("sift_descriptors\\Train\\reduced_to_50\\" + image_name + "_sift.npy", des)
 
    for image_name in os.listdir("pre_processed\\Train\\reduced_to_200"):
        image = cv.imread("pre_processed\\Train\\reduced_to_200\\" + image_name)
        if image is None:
            print ('Error opening image!')
            return -1
        
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create()
        kp, des = sift.detectAndCompute(gray_image, None)

        sift_image = cv.drawKeypoints(gray_image, kp, image)
        cv.imwrite("sift_features\\Train\\reduced_to_200\\" + image_name + "_sift.jpg", sift_image)

        # save sift descriptors; .npy file
        # flatten
        if des is not None:
            des = des.flatten()

        np.save("sift_descriptors\\Train\\reduced_to_200\\" + image_name + "_sift.npy", des)

    for image_name in os.listdir("pre_processed\\Test\\reduced_to_50"):
        image = cv.imread("pre_processed\\Test\\reduced_to_50\\" + image_name)
        if image is None:
            print ('Error opening image!')
            return -1
        
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create()
        kp, des = sift.detectAndCompute(gray_image, None)

        sift_image = cv.drawKeypoints(gray_image, kp, image)
        cv.imwrite("sift_features\\Test\\reduced_to_50\\" + image_name + "_sift.jpg", sift_image)

        # save sift descriptors; .npy file
        if des is not None:
            des = des.flatten()

        np.save("sift_descriptors\\Test\\reduced_to_50\\" + image_name + "_sift.npy", des)

    for image_name in os.listdir("pre_processed\\Test\\reduced_to_200"):
        image = cv.imread("pre_processed\\Test\\reduced_to_200\\" + image_name)
        if image is None:
            print ('Error opening image!')
            return -1
        
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create()
        kp, des = sift.detectAndCompute(gray_image, None)

        sift_image = cv.drawKeypoints(gray_image, kp, image)
        cv.imwrite("sift_features\\Test\\reduced_to_200\\" + image_name + "_sift.jpg", sift_image)

        # save sift descriptors; .npy file
        if des is not None:
            des = des.flatten()

        np.save("sift_descriptors\\Test\\reduced_to_200\\" + image_name + "_sift.npy", des)

    return 0


# Extract Histogram features on ALL training images and save the data.    
def extract_histogram():

    # create histogram_features folder
    if not os.path.exists("histogram_features"):
        os.makedirs("histogram_features")

    if not os.path.exists("histogram_features\\Train\\reduced_to_50"):
        os.makedirs("histogram_features\\Train\\reduced_to_50")

    if not os.path.exists("histogram_features\\Train\\reduced_to_200"):
        os.makedirs("histogram_features\\Train\\reduced_to_200")

    if not os.path.exists("histogram_features\\Test\\reduced_to_50"):
        os.makedirs("histogram_features\\Test\\reduced_to_50")

    if not os.path.exists("histogram_features\\Test\\reduced_to_200"):
        os.makedirs("histogram_features\\Test\\reduced_to_200")

    # extract sift features on training images

    for image_name in os.listdir("pre_processed\\Train\\reduced_to_50"):
        image = cv.imread("pre_processed\\Train\\reduced_to_50\\" + image_name)
        if image is None:
            print ('Error opening image!')
            return -1
        
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # 64 bins and range [0 to 256]
        hist = cv.calcHist([gray_image], [0], None, [64], [0, 256])
        hist_norm = cv.normalize(hist, hist).flatten()

        # save histogram image
        plt.hist(gray_image.ravel(), 256, [0, 256])
        plt.savefig("histogram_features\\Train\\reduced_to_50\\" + image_name + "_histogram.jpg")
        plt.close()   

    for image_name in os.listdir("pre_processed\\Train\\reduced_to_200"):
        image = cv.imread("pre_processed\\Train\\reduced_to_200\\" + image_name)
        if image is None:
            print ('Error opening image!')
            return -1
        
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # 64 bins and range [0 to 256]
        hist = cv.calcHist([gray_image], [0], None, [64], [0, 256])
        hist_norm = cv.normalize(hist, hist).flatten()

        # save histogram image
        plt.hist(gray_image.ravel(), 256, [0, 256])
        plt.savefig("histogram_features\\Train\\reduced_to_200\\" + image_name + "_histogram.jpg")
        plt.close()

    for image_name in os.listdir("pre_processed\\Test\\reduced_to_50"):
        image = cv.imread("pre_processed\\Test\\reduced_to_50\\" + image_name)
        if image is None:
            print ('Error opening image!')
            return -1
        
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # 64 bins and range [0 to 256]
        hist = cv.calcHist([gray_image], [0], None, [64], [0, 256])
        hist_norm = cv.normalize(hist, hist).flatten()

        # save histogram image
        plt.hist(gray_image.ravel(), 256, [0, 256])
        plt.savefig("histogram_features\\Test\\reduced_to_50\\" + image_name + "_histogram.jpg")
        plt.close()

    for image_name in os.listdir("pre_processed\\Test\\reduced_to_200"):
        image = cv.imread("pre_processed\\Test\\reduced_to_200\\" + image_name)
        if image is None:
            print ('Error opening image!')
            return -1
        
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # 64 bins and range [0 to 256]
        hist = cv.calcHist([gray_image], [0], None, [64], [0, 256])
        hist_norm = cv.normalize(hist, hist).flatten()

        # save histogram image
        plt.hist(gray_image.ravel(), 256, [0, 256])
        plt.savefig("histogram_features\\Test\\reduced_to_200\\" + image_name + "_histogram.jpg")
        plt.close()
    
    return 0


# Perform the following FOUR TRAINING on the data: 

# Represent the image directly using the 50*50 (2500) pixel values and use the Nearest Neighbor classifier 
def nearest_neighbor_50x50():

    # create train and test data
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    # train data
    for image_name in os.listdir("pre_processed\\Train\\reduced_to_50"):
        image = cv.imread("pre_processed\\Train\\reduced_to_50\\" + image_name)
        if image is None:
            print ('Error opening image!')
            return -1
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        train_data.append(gray_image.flatten())

    # test data
    for image_name in os.listdir("pre_processed\\Test\\reduced_to_50"):
        image = cv.imread("pre_processed\\Test\\reduced_to_50\\" + image_name)
        if image is None:
            print ('Error opening image!')
            return -1
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        test_data.append(gray_image.flatten())
        # test_labels.append(0)

    # 0 is bedroom, 1 is Coast, and 2 is Forest
    train_labels = np.zeros(300)
    train_labels[0:100] = 0
    train_labels[100:200] = 1
    train_labels[200:300] = 2
    train_labels = np.array(train_labels).astype(np.float32)
    train_labels = train_labels.reshape(300,1)

    # convert to numpy arrays
    test_labels = np.zeros(604)
    test_labels[0:116] = 0
    test_labels[116:376] = 1
    test_labels[376:604] = 2
    train_data = np.array(train_data).astype(np.float32)
    train_labels = np.array(train_labels).astype(np.float32)
    test_data = np.array(test_data).astype(np.float32)
    test_labels = np.array(test_labels).astype(np.float32)

    train_labels = train_labels.reshape(300,1)
    test_labels = test_labels.reshape(604,1)
  
    # save data
    np.save("pre_processed\\Train\\train_data.npy", train_data)
    np.save("pre_processed\\Train\\train_labels.npy", train_labels)
    np.save("pre_processed\\Test\\test_data.npy", test_data)
    np.save("pre_processed\\Test\\test_labels.npy", test_labels)

    # load data
    train_data = np.load("pre_processed\\Train\\train_data.npy")
    train_labels = np.load("pre_processed\\Train\\train_labels.npy")
    test_data = np.load("pre_processed\\Test\\test_data.npy")
    test_labels = np.load("pre_processed\\Test\\test_labels.npy")

    model = cv.ml.KNearest_create()
    model.train(train_data, cv.ml.ROW_SAMPLE, train_labels)
    
    # Percentage of correctly classified images in the test set

    ret,result,neighbours,dist = model.findNearest(test_data,k=3)
    temp = result == test_labels
    accuracy = (np.count_nonzero(temp))*100/result.size
    print("Part 4a: Using nearest neighbor classifier with 50x50 pixel values\n")
    print(f"Correctly classified images = {accuracy:.2f}%\n")

    # Percentage of False Positives (images that are falsely classified) & False Negatives (images that are not classified)
    # False Positive: 1 - (number of correctly classified images / total number of images in the class)
    # False Negative: 1 - (number of correctly classified images / total number of images in the class)

    false_positives = 0
    false_negatives = 0

    fp_bedroom = 0
    fp_coast = 0
    fp_forest = 0
    fn_bedroom = 0
    fn_coast = 0
    fn_forest = 0

    # Checking which prints false positive on the coast and forest data 
    for i in range(116, 604):
        if result[i] == 0:
            fp_bedroom += 1
    
    for i in range(0, 116):
        if result[i] == 1:
            fp_coast += 1
    for i in range(376, 604):
        if result[i] == 1:
            fp_coast += 1

    for i in range(0, 376):
        if result[i] == 2:
            fp_forest += 1


    for i in range(0, 116):
        if not result[i] == 0:
            fn_bedroom += 1
    for i in range(116, 376):
        if not result[i] == 1:
            fn_coast += 1
    for i in range(376, 604):
        if not result[i] == 2:
            fn_forest += 1

    fp_bedroom = fp_bedroom * 100 / (604 - 116)
    fp_coast = fp_coast * 100 / (604 - 376 + 116)
    fp_forest = fp_forest * 100 / (604 - 376)
    false_positives = (fp_bedroom + fp_coast + fp_forest) / 3

    fn_bedroom = fn_bedroom * 100 / 116
    fn_coast = fn_coast * 100 / 260
    fn_forest = fn_forest * 100 / 228
    false_negatives = (fn_bedroom + fn_coast + fn_forest) / 3

    print(f"Percentage of False Positives: {false_positives:.2f}%\n")
    print(f"False Positive: \nBedroom: {fp_bedroom:.2f}%\nCoast: {fp_coast:.2f}%\nForest: {fp_forest:.2f}%\n")
    print(f"Percentage of False Negatives: {false_negatives:.2f}%\n")
    print(f"False Negative: \nBedroom: {fn_bedroom:.2f}%\nCoast: {fn_coast:.2f}%\nForest: {fn_forest:.2f}%\n")

    return 0


# Represent the image using SIFT feature data and use Nearest Neighbor classifier
def nearest_neighbor_sift():

    # create train and test data
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    # train data
    for image_name in os.listdir("sift_features\\Train\\reduced_to_50"):
        image = cv.imread("sift_features\\Train\\reduced_to_50\\" + image_name)
        if image is None:
            print ('Error opening image!')
            return -1

        # gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # test_data.append(gray_image.flatten())

        train_data.append(image.flatten())
        # train_labels.append(0)

    # test data
    for image_name in os.listdir("sift_features\\Test\\reduced_to_50"):
        image = cv.imread("sift_features\\Test\\reduced_to_50\\" + image_name)
        if image is None:
            print ('Error opening image!')
            return -1
        
        # gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # test_data.append(gray_image.flatten())

        test_data.append(image.flatten())
        # test_labels.append(0)

    # 0 is bedroom, 1 is Coast, and 2 is Forest
    train_labels = np.zeros(300)
    train_labels[0:100] = 0
    train_labels[100:200] = 1
    train_labels[200:300] = 2
    train_labels = np.array(train_labels).astype(np.float32)
    train_labels = train_labels.reshape(300,1)

    # convert to numpy arrays
    test_labels = np.zeros(604)
    test_labels[0:100] = 0
    test_labels[116:376] = 1
    test_labels[376:604] = 2
    # ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (596,) + inhomogeneous part.
    train_data = np.array(train_data).astype(np.float32)
    train_labels = np.array(train_labels).astype(np.float32)
    test_data = np.array(test_data).astype(np.float32)
    test_labels = np.array(test_labels).astype(np.float32)

    train_labels = train_labels.reshape(300,1)
    test_labels = test_labels.reshape(604,1)
  
    # save data
    np.save("pre_processed\\Train\\train_data.npy", train_data)
    np.save("pre_processed\\Train\\train_labels.npy", train_labels)
    np.save("pre_processed\\Test\\test_data.npy", test_data)
    np.save("pre_processed\\Test\\test_labels.npy", test_labels)

    # load data
    train_data = np.load("pre_processed\\Train\\train_data.npy")
    train_labels = np.load("pre_processed\\Train\\train_labels.npy")
    test_data = np.load("pre_processed\\Test\\test_data.npy")
    test_labels = np.load("pre_processed\\Test\\test_labels.npy")

    model = cv.ml.KNearest_create()
    model.train(train_data, cv.ml.ROW_SAMPLE, train_labels)
    
    # Percentage of correctly classified images in the test set

    ret,result,neighbours,dist = model.findNearest(test_data,k=3)
    temp = result == test_labels
    accuracy = (np.count_nonzero(temp))*100/result.size
    print("Part 4b: Using nearest neighbor classifier with SIFT features\n")   
    print(f"Correctly classified images = {accuracy:.2f}%\n")

    # Percentage of False Positives (images that are falsely classified) & False Negatives (images that are not classified)
    # False Positive: 1 - (number of correctly classified images / total number of images in the class)
    # False Negative: 1 - (number of correctly classified images / total number of images in the class)

    false_positives = 0
    false_negatives = 0

    fp_bedroom = 0
    fp_coast = 0
    fp_forest = 0
    fn_bedroom = 0
    fn_coast = 0
    fn_forest = 0

    # Checking which prints false positive on the coast and forest data 
    for i in range(116, 604):
        if result[i] == 0:
            fp_bedroom += 1
    
    for i in range(0, 116):
        if result[i] == 1:
            fp_coast += 1
    for i in range(376, 604):
        if result[i] == 1:
            fp_coast += 1

    for i in range(0, 376):
        if result[i] == 2:
            fp_forest += 1


    for i in range(0, 116):
        if not result[i] == 0:
            fn_bedroom += 1
    for i in range(116, 376):
        if not result[i] == 1:
            fn_coast += 1
    for i in range(376, 604):
        if not result[i] == 2:
            fn_forest += 1

    fp_bedroom = fp_bedroom * 100 / (604 - 116)
    fp_coast = fp_coast * 100 / (604 - 376 + 116)
    fp_forest = fp_forest * 100 / (604 - 376)
    false_positives = (fp_bedroom + fp_coast + fp_forest) / 3

    fn_bedroom = fn_bedroom * 100 / 116
    fn_coast = fn_coast * 100 / 260
    fn_forest = fn_forest * 100 / 228
    false_negatives = (fn_bedroom + fn_coast + fn_forest) / 3

    print(f"Percentage of False Positives: {false_positives:.2f}%\n")
    print(f"False Positive: \nBedroom: {fp_bedroom:.2f}%\nCoast: {fp_coast:.2f}%\nForest: {fp_forest:.2f}%\n")
    print(f"Percentage of False Negatives: {false_negatives:.2f}%\n")
    print(f"False Negative: \nBedroom: {fn_bedroom:.2f}%\nCoast: {fn_coast:.2f}%\nForest: {fn_forest:.2f}%\n")

    return 0


def nearest_neighbor_histogram():

    # create train and test data
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    # train data
    for image_name in os.listdir("histogram_features\\Train\\reduced_to_50"):
        image = cv.imread("histogram_features\\Train\\reduced_to_50\\" + image_name)
        if image is None:
            print ('Error opening image!')
            return -1

        # gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # test_data.append(gray_image.flatten())

        train_data.append(image.flatten())
        # train_labels.append(0)

    # test data
    for image_name in os.listdir("histogram_features\\Test\\reduced_to_50"):
        image = cv.imread("histogram_features\\Test\\reduced_to_50\\" + image_name)
        if image is None:
            print ('Error opening image!')
            return -1
        
        # gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # test_data.append(gray_image.flatten())

        test_data.append(image.flatten())
        # test_labels.append(0)

    # 0 is bedroom, 1 is Coast, and 2 is Forest
    train_labels = np.zeros(300)
    train_labels[0:100] = 0
    train_labels[100:200] = 1
    train_labels[200:300] = 2
    train_labels = np.array(train_labels).astype(np.float32)
    train_labels = train_labels.reshape(300,1)

    # convert to numpy arrays
    test_labels = np.zeros(604)
    test_labels[0:116] = 0
    test_labels[116:376] = 1
    test_labels[376:604] = 2
    train_data = np.array(train_data).astype(np.float32)
    train_labels = np.array(train_labels).astype(np.float32)
    test_data = np.array(test_data).astype(np.float32)
    test_labels = np.array(test_labels).astype(np.float32)

    train_labels = train_labels.reshape(300,1)
    test_labels = test_labels.reshape(604,1)
  
    # save data
    np.save("pre_processed\\Train\\train_data.npy", train_data)
    np.save("pre_processed\\Train\\train_labels.npy", train_labels)
    np.save("pre_processed\\Test\\test_data.npy", test_data)
    np.save("pre_processed\\Test\\test_labels.npy", test_labels)

    # load data
    train_data = np.load("pre_processed\\Train\\train_data.npy")
    train_labels = np.load("pre_processed\\Train\\train_labels.npy")
    test_data = np.load("pre_processed\\Test\\test_data.npy")
    test_labels = np.load("pre_processed\\Test\\test_labels.npy")

    model = cv.ml.KNearest_create()
    model.train(train_data, cv.ml.ROW_SAMPLE, train_labels)
    
    # Percentage of correctly classified images in the test set

    ret,result,neighbours,dist = model.findNearest(test_data,k=3)
    temp = result == test_labels
    accuracy = (np.count_nonzero(temp))*100/result.size
    print("Part 4c: Using nearest neighbor classifier with Histogram features\n")
    print(f"Correctly classified images = {accuracy:.2f}%\n")

    false_positives = 0
    false_negatives = 0

    fp_bedroom = 0
    fp_coast = 0
    fp_forest = 0
    fn_bedroom = 0
    fn_coast = 0
    fn_forest = 0

    # Checking which prints false positive on the coast and forest data 
    for i in range(116, 604):
        if result[i] == 0:
            fp_bedroom += 1
    
    for i in range(0, 116):
        if result[i] == 1:
            fp_coast += 1
    for i in range(376, 604):
        if result[i] == 1:
            fp_coast += 1

    for i in range(0, 376):
        if result[i] == 2:
            fp_forest += 1


    for i in range(0, 116):
        if not result[i] == 0:
            fn_bedroom += 1
    for i in range(116, 376):
        if not result[i] == 1:
            fn_coast += 1
    for i in range(376, 604):
        if not result[i] == 2:
            fn_forest += 1

    fp_bedroom = fp_bedroom * 100 / (604 - 116)
    fp_coast = fp_coast * 100 / (604 - 376 + 116)
    fp_forest = fp_forest * 100 / (604 - 376)
    false_positives = (fp_bedroom + fp_coast + fp_forest) / 3

    fn_bedroom = fn_bedroom * 100 / 116
    fn_coast = fn_coast * 100 / 260
    fn_forest = fn_forest * 100 / 228
    false_negatives = (fn_bedroom + fn_coast + fn_forest) / 3

    print(f"Percentage of False Positives: {false_positives:.2f}%\n")
    print(f"False Positive: \nBedroom: {fp_bedroom:.2f}%\nCoast: {fp_coast:.2f}%\nForest: {fp_forest:.2f}%\n")
    print(f"Percentage of False Negatives: {false_negatives:.2f}%\n")
    print(f"False Negative: \nBedroom: {fn_bedroom:.2f}%\nCoast: {fn_coast:.2f}%\nForest: {fn_forest:.2f}%\n")


    return 0




# Represent the image using SIFT feature data and use linear SVM classifier
def linear_svm_sift():

    # create train and test data
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    # train data
    for filename in os.listdir("pre_processed\\Train\\reduced_to_50"):
        if filename.endswith('.npy'):
            filepath = os.path.join("pre_processed\\Train\\reduced_to_50", filename)
            array = np.load(filepath)
            train_data.append(array)

    # test data
    for filename in os.listdir("pre_processed\\Test\\reduced_to_50"):
        if filename.endswith('.npy'):
            filepath = os.path.join("pre_processed\\Test\\reduced_to_50", filename)
            array = np.load(filepath)
            test_data.append(array)

    # 0 is bedroom, 1 is Coast, and 2 is Forest
    train_labels = np.zeros(300)
    train_labels[0:100] = 0
    train_labels[100:200] = 1
    train_labels[200:300] = 2
    train_labels = np.array(train_labels).astype(np.float32)
    train_labels = train_labels.reshape(300,1)
    
    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.train(np.float32(train_data[:300]), cv.ml.ROW_SAMPLE, np.array(train_labels[:300], dtype=np.int32))
    ret, res = svm.predict(np.float32(train_data[300:]))

    # convert to numpy arrays
    test_labels = np.zeros(604)
    test_labels[0:116] = 0
    test_labels[116:376] = 1
    test_labels[376:604] = 2
    train_data = np.array(train_data).astype(np.float32)
    train_labels = np.array(train_labels).astype(np.float32)
    test_data = np.array(test_data).astype(np.float32)
    test_labels = np.array(test_labels).astype(np.float32)

    train_labels = train_labels.reshape(300,1)
    test_labels = test_labels.reshape(604,1)   


    ''' 
  
    # save data
    np.save("pre_processed\\Train\\train_data.npy", train_data)
    np.save("pre_processed\\Train\\train_labels.npy", train_labels)
    np.save("pre_processed\\Test\\test_data.npy", test_data)
    np.save("pre_processed\\Test\\test_labels.npy", test_labels)

    # load data
    train_data = np.load("pre_processed\\Train\\train_data.npy")
    train_labels = np.load("pre_processed\\Train\\train_labels.npy")
    test_data = np.load("pre_processed\\Test\\test_data.npy")
    test_labels = np.load("pre_processed\\Test\\test_labels.npy")
    ''' 
        # SVM
    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    # error: (-5:Bad argument) in the case of classification problem the responses must be categorical; 
    # either specify varType when creating TrainData, or pass integer responses in function 'cv::ml::SVMImpl::train'
    svm.train(train_data, cv.ml.ROW_SAMPLE, train_labels)

    # Percentage of correctly classified images in the test set
    ret,result = svm.predict(test_data)
    temp = result == test_labels
    accuracy = (np.count_nonzero(temp))*100/result.size
    # print("Part 4d: Using linear SVM classifier with SIFT features\n")
    # print(f"Correctly classified images = {accuracy:.2f}%\n")
    return 0


# driver
def main(argv):
    # folder path
    path = str(pathlib.Path(__file__).parent.resolve())
    # print(path)
    suffix1 = "\\ProjData\\Train\\bedroom"
    folder1 = path + suffix1
    suffix2 = "\\ProjData\\Test\\bedroom"
    folder2 = path + suffix2
    
    suffix3 = "\\ProjData\\Train\\coast"
    folder3 = path + suffix3
    suffix4 = "\\ProjData\\Test\\coast"
    folder4 = path + suffix4

    suffix5 = "\\ProjData\\Train\\forest"
    folder5 = path + suffix5
    suffix6 = "\\ProjData\\Test\\forest"
    folder6 = path + suffix6

    # make folder for pre-processed images      
    if not os.path.exists(path + "\\pre_processed\\Train\\reduced_to_200"):
        os.makedirs(path + "\\pre_processed\\Train\\reduced_to_200")

    if not os.path.exists(path + "\\pre_processed\\Test\\reduced_to_200"):
        os.makedirs(path + "\\pre_processed\\Test\\reduced_to_200")
    
    if not os.path.exists(path + "\\pre_processed\\Train\\reduced_to_50"):
        os.makedirs(path + "\\pre_processed\\Train\\reduced_to_50")

    if not os.path.exists(path + "\\pre_processed\\Test\\reduced_to_50"):
        os.makedirs(path + "\\pre_processed\\Test\\reduced_to_50")
    
    

    # pre-processing
    pre_process(folder1)
    pre_process(folder2)
    pre_process(folder3)
    pre_process(folder4)
    pre_process(folder5)
    pre_process(folder6)
    

# This will duplicate the files
    suffix7 = "\\pre_processed\\Train"
    train_dir = path + suffix7
    suffix8 = "\\pre_processed\\Test"
    test_dir = path + suffix8




    reduced_to_50_dir = os.path.join(train_dir, 'reduced_to_50')
    reduced_to_200_dir = os.path.join(train_dir, 'reduced_to_200')

    # Loop through all the files in the train directory
    for root, dirs, files in os.walk(train_dir):
         for file in files:
             # Check if the filename contains '50x50'
             if '50x50' in file:
                 # Move the file to the reduced_to_50 directory
                 src = os.path.join(root, file)
                 dst = os.path.join(reduced_to_50_dir, file)
                 shutil.move(src, dst)
             # Check if the filename contains '200x200'
             elif '200x200' in file:
                 # Move the file to the reduced_to_200 directory
                 src = os.path.join(root, file)
                 dst = os.path.join(reduced_to_200_dir, file)
                 shutil.move(src, dst)





    reduced_to_50_dir = os.path.join(test_dir, 'reduced_to_50')
    reduced_to_200_dir = os.path.join(test_dir, 'reduced_to_200')

     # Loop through all the files in the Test directory
    for root, dirs, files in os.walk(test_dir):
        for file in files:
             # Check if the filename contains '50x50'
            if '50x50' in file:
                 # Move the file to the reduced_to_50 directory
                 src = os.path.join(root, file)
                 dst = os.path.join(reduced_to_50_dir, file)
                 shutil.move(src, dst)
             # Check if the filename contains '200x200'
            elif '200x200' in file:
                 # Move the file to the reduced_to_200 directory
                 src = os.path.join(root, file)
                 dst = os.path.join(reduced_to_200_dir, file)
                 shutil.move(src, dst)


    # Project Requirements

    # extract SIFT features
    extract_sift()

    # extract histogram features
    extract_histogram()

    # Represent the image directly using the 50*50 (2500) pixel values and use the Nearest Neighbor classifier
    nearest_neighbor_50x50()

    # Represent the image directly using sift features and use the Nearest Neighbor classifier
    nearest_neighbor_sift()

    # Represent the image using histogram features and use the Nearest Neighbor classifier
    nearest_neighbor_histogram()

    # Represent the image using SIFT features and use the linear SVM classifier
    # linear_svm_sift()


    print("is done")
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])