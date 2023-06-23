import cv2
import numpy as np

import utlis

###########################
path = "6.jpg"
widthImg = 600
heightImg = 800
###########################


img = cv2.imread(path)

# preprocessing
img = cv2.resize(img, (widthImg,heightImg))
imgContours = img.copy()
imgBiggestContours = img.copy()
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
imgCanny = cv2.Canny(imgBlur,10,50)


#Finding all contours
contours, hierarchy = cv2.findContours (imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours,contours,-1,(0,255,0),10)

#Find rectangles
rectCon = utlis.rectCountour(contours)
biggestContour = utlis.getCornerPoints(rectCon[0])
#print(biggestContour.shape)
gradePoints = utlis.getCornerPoints(rectCon[1])
#print(biggestContour)


if biggestContour.size !=0 and gradePoints.size !=0:
    cv2.drawContours(imgBiggestContours,biggestContour,-1,(0,255,0),20)
    cv2.drawContours(imgBiggestContours,gradePoints,-1,(255,0,0),20)

    biggestContour = utlis.reorder(biggestContour)
    gradePoints = utlis.reorder(gradePoints)

    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
    matrix = cv2.getPerspectiveTransform(pt1,pt2)
    imgWarpColored = cv2.warpPerspective(img,matrix,(widthImg,heightImg))

    ptG1 = np.float32(gradePoints)
    ptG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
    matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
    imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))
    #cv2.imshow("Grade",imgGradeDisplay)

    # Apply threshold
    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgWarpGray, 100, 255, cv2.THRESH_BINARY_INV)[1]


imgBlank = np.zeros_like(img)
imageArray = ([img,imgGray,imgBlur,imgCanny],
              [imgContours,imgBiggestContours,imgWarpColored,imgThresh])

imgStacked = utlis.stackImages(imageArray,0.40)

#cv2.imshow("Stacked Images",imgStacked)



#img = cv2.resize(img,(420,594))
#img = img[37:557,36:372]
#img =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

imgThresh = imgThresh[10:790,60:540]

cv2.imshow("Image",imgThresh);


img_columns = np.hsplit(imgThresh,3)
img_row1 = np.vsplit(img_columns[0],10)
img_row2 = np.vsplit(img_columns[1], 10)
img_row3 = np.vsplit(img_columns[2], 10)
value1 = 0
value2 = 0
value3 = 0
for i in range(0,9):
    if np.count_nonzero(img_row1[value1])>np.count_nonzero(img_row1[i+1]):
        value1 = value1
    else: value1 = i+1 ;

for i in range(0, 9):
        if np.count_nonzero(img_row2[value2]) > np.count_nonzero(img_row2[i + 1]):
            value2 = value2
        else:
            value2 = i + 1

for i in range(0,9):
    if np.count_nonzero(img_row3[value3]) > np.count_nonzero(img_row3[i+1]):
        value3 = value3
    else: value3 = i+1


print(value1,value2,value3)


#test values of all

'''for i in range(0,10):
    print(i)
    print(np.count_nonzero(img_row1[i]))
    print(np.count_nonzero(img_row2[i]))
    print(np.count_nonzero(img_row3[i]),"")'''

mark = str(value1) + str(value2) + str(value3)

cv2.putText(imgGradeDisplay,mark,(50,100),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,0),3)
cv2.imshow("Mark",imgGradeDisplay)

'''cv2.imshow("0",img_row1[0])
cv2.imshow("1",img_row1[1])
cv2.imshow("2",img_row1[2])
cv2.imshow("3",img_row1[3])
cv2.imshow("4",img_row1[4])
cv2.imshow("5",img_row1[5])
cv2.imshow("6",img_row1[6])
cv2.imshow("7",img_row1[7])
cv2.imshow("8",img_row1[8])
cv2.imshow("9",img_row1[9])'''

'''cv2.imshow("0",img_row3[0])
cv2.imshow("1",img_row3[1])
cv2.imshow("2",img_row3[2])
cv2.imshow("3",img_row3[3])
cv2.imshow("4",img_row3[4])
cv2.imshow("5",img_row3[5])
cv2.imshow("6",img_row3[6])
cv2.imshow("7",img_row3[7])
cv2.imshow("8",img_row3[8])
cv2.imshow("9",img_row3[9])'''




cv2.waitKey(0)
cv2.destroyAllWindows()




