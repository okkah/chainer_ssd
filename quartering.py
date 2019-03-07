import cv2
import os

images = os.listdir("../data/train_raw") #ディレクトリのパス

for i in images:
    filename = "../data/train_raw/" + i
    img = cv2.imread(filename)
    height, width, channels = img.shape
    print("Load {}".format(filename))
        
    clp = img[0:height//2, 0:width//2]     
    cv2.imwrite("../data/train/tl_" + i, clp)   

    clp = img[0:height//2, width//2:width]     
    cv2.imwrite("../data/train/tr_" + i, clp)   

    clp = img[height//2:height, 0:width//2]     
    cv2.imwrite("../data/train/ul_" + i, clp)   

    clp = img[height//2:height, width//2:width]     
    cv2.imwrite("../data/train/ur_" + i, clp)  

images = os.listdir("../data/test_raw") #ディレクトリのパス

for i in images:
    filename = "../data/test_raw/" + i
    img = cv2.imread(filename)
    height, width, channels = img.shape
    print("Load {}".format(filename))
        
    clp = img[0:height//2, 0:width//2]     
    cv2.imwrite("../data/test/tl_" + i, clp)   

    clp = img[0:height//2, width//2:width]     
    cv2.imwrite("../data/test/tr_" + i, clp)   

    clp = img[height//2:height, 0:width//2]     
    cv2.imwrite("../data/test/ul_" + i, clp)   

    clp = img[height//2:height, width//2:width]     
    cv2.imwrite("../data/test/ur_" + i, clp)   
