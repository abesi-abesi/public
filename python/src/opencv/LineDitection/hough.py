import sys
import cv2
import numpy as np

class hough:
    
    def conversion(self,image):
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,30,40,apertureSize=3)
        cv2.imshow("エッジ画像",edges)
        #blur = cv2.medianBlur(gray,5)

        #長い線の検出
        lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=90,minLineLength=10,maxLineGap=50)
        #短い線の検出
        # sLines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=50,minLineLength=10,maxLineGap=30)

        # #円の検出
        # circles = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,200,param1=50,param2=40,minRadius=0,maxRadius=0)
        # circles = np.uint16(np.around(circles))
        
        dst = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)

        # for (x, y, r) in circles[0]:
        #     cv2.circle(dst, (x, y), r, (0, 255, 0), 2)
        #     cv2.circle(dst, (x, y), 2, (0, 0, 255), 3)
        

        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(dst,(x1,y1),(x2,y2),(0,0,255),2)

        # for line in sLines:
        #     x1,y1,x2,y2 = line[0]
        #     cv2.line(dst,(x1,y1),(x2,y2),(255,0,0),2)

        # if cv2.waitKey() == ord("s"):
        #     num = self.name.rfind("/")
        #     str = "./result/hough-" + self.name[num+1:]
        #     cv2.imwrite(str,dst)
        cv2.imshow("ライン抽出",dst)
        #cv2.imwrite("./result/resultImage.png",dst)
        return dst

# def output():
#     args = sys.argv

#     if len(args) > 1:
#         hough = hough(args[1])
#     else:
#         hough = hough("./opencv/image/angle1.png")

#     hough.conversion()
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
if __name__=="__main__":
    args = sys.argv
    hough = hough()
    if len(args) > 1:
        image = cv2.imread(args[1])
    else:
        image = cv2.imread("./image/angle1.png")

    hough.conversion(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
