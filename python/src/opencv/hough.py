import sys
import cv2
import numpy as np

class hough:

    def __init__(self,name):
        self.name = name
    
    
    def conversion(self):
        image = cv2.imread(self.name)
        cv2.imshow("元画像",image)
        minLineLength = 100
        maxLineGap = 10
        lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
        for x1,y1,x2,y2 in lines[0]:
            cv2.line(dst,(x1,y1),(x2,y2),(0,0,255),2)

        cv2.imshow("結果画像",dst)
    

args = sys.argv
if len(args) > 1:
    hough = hough(args[1])
else:
    hough = hough("./image/angle1.png")

hough.conversion()
