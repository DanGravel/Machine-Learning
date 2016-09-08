import cv2
import imutils
import numpy as np

def points(pts):
    rectangle = np.zeros((4,2), dtype = "float32")

    #Bottom left point has smallest sum
    #Bottom right point has largest sum
    s = map(sum, pts)
    rectangle[0] = pts[np.argmin(s)]
    rectangle[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rectangle[1] = pts[np.argmin(diff)]
    rectangle[3] = pts[np.argmax(diff)]

    return rectangle

def corner_transformation(image, pts):
    rectangle = points(pts)
    (tl, tr, br, bl) = rectangle

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA),int(heightB))

    dst = np.array([
    [0,0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")

    transform_matrix = cv2.getPerspectiveTransform(rectangle,dst)
    warp = cv2.warpPerspective(image,transform_matrix,(maxWidth,maxHeight))

    return warp

def fix_perspective(image):
    image = cv2.imread(image)
    ratio = image.shape[0]/500.0
    original = image.copy()
    image = imutils.resize(image, height = 500)


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5),0)
    edged = cv2.Canny(gray, 75, 200)

    (cnts,_) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse =True)[:5]

    for contour in cnts:

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    warped = corner_transformation(original, screenCnt.reshape(4,2)*ratio)
    cv2.imwrite('check.JPG', warped)
