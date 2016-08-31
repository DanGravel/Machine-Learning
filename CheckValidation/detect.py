import cv2
import numpy as np
import imutils
import neuralNet

def text_regions(image):
    img = cv2.imread(image)
    img = imutils.resize(img,width = 1000)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    joining_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,5))

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    black = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

    gradX = cv2.Sobel(black, ddepth = cv2.CV_32F, dx = 1, dy =0, ksize = -1)
    gradX= np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255*((gradX - minVal)/(maxVal - minVal))).astype("uint8")
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, kernel)

    thresh = cv2.threshold(gradX, 127 , 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, joining_kernel)

    contours, hierarchy = cv2.findContours(thresh,
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dollars = []
    for contour in contours:
        #bounding region
        [x,y,w,h] = cv2.boundingRect(contour)
        #get rid of very large and small contours
        if (w < 20 or h < 10) or (w > 55 or h > 15):
            continue
        dollars = img_gray[y-10:y+h+10, x-10:x+w + 10]

    ret, im_th = cv2.threshold(dollars, 90, 255, cv2.THRESH_BINARY_INV)
    ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    for ctr in ctrs:
        [x,y,w,h] = cv2.boundingRect(ctr)
        if w < 4 or h < 4:
            continue
        temp = dollars[y-3:y+h+3, x-3:x+w+3]
        temp = cv2.resize(temp,(28,28), interpolation = cv2.INTER_AREA)
        cv2.imwrite("number" + str(i)+ ".png",temp)
        i = i + 1





text_regions('t2.png')
