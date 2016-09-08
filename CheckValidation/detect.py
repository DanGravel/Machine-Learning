import cv2
import numpy as np
import imutils
import classifier

def text_regions(image):
    img = cv2.imread(image)
    img = imutils.resize(img,width = 1000)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    joining_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,5))

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Detects dark regions, in this case writing
    black = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

    #Finds areas within dark regions that have vertical edges
    gradX = cv2.Sobel(black, ddepth = cv2.CV_32F, dx = 1, dy =0, ksize = -1)
    gradX= np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255*((gradX - minVal)/(maxVal - minVal))).astype("uint8")
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, kernel)

    thresh = cv2.threshold(gradX, 127 , 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, joining_kernel)

    cv2.imshow("thresh", thresh)
    cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(thresh,
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dollars = []
    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)
        #get rid of very large and small contours
        if w < 60 or w > 75 or h < 13 or h > 18 :
            continue
        #gets rid of contours in the check that cant be the money region(
        #(ex theyre on the right side of check)
        if x < 500:
            continue
        dollars = img_gray[y-10:y+h+10, x-10:x+w+10]
        cv2.imshow("dollars", dollars)
        cv2.waitKey(0)

    #detecting contours in the dollars field
    ret, im_th = cv2.threshold(dollars, 90, 255, cv2.THRESH_BINARY_INV)
    ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    numbers = []
    for ctr in ctrs:
        [x,y,w,h] = cv2.boundingRect(ctr)
        print("%d %d" %(w,h))
        if w < 7 and h < 7: #ignore very small regions
            continue
        temp = dollars[y-3:y+h+3, x-2:x+w+2]
        temp = cv2.resize(temp,(28,28), interpolation = cv2.INTER_AREA)
        cv2.imshow("dollars", temp)
        cv2.waitKey(0)
        numbers.append(temp)

    return numbers
