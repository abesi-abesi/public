import sys
import cv2
import numpy as np

class hough:

    def __init__(self,name):
        self.name = name
    
    
    def conversion(self):
        image = cv2.imread(self.name)
        cv2.imshow("元画像",image)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,30,40,apertureSize=3)
        cv2.imshow("エッジ画像",edges)

        #長い線の検出
        lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=200,minLineLength=200,maxLineGap=50)
        #短い線の検出
        sLines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=200,minLineLength=5,maxLineGap=5)

        dst = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(dst,(x1,y1),(x2,y2),(0,0,255),2)

        for line in sLines:
            x1,y1,x2,y2 = line[0]
            cv2.line(dst,(x1,y1),(x2,y2),(0,0,255),2)


        cv2.imshow("結果画像",dst)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

args = sys.argv
if len(args) > 1:
    hough = hough(args[1])
else:
    hough = hough("./image/angle1.png")

hough.conversion()
