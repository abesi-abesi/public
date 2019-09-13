import cv2
import numpy as np
import sys

class diteciton:

    def __init__(self,image,name):
        self.image = image #画像オブジェクト
        self.name = name #画像ファイル名

    #フィールド部分の抽出    
    def fieldDitection(self,image):
        lower = np.array([35,50,50])
        upper = np.array([80,255,255])

        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        dst = cv2.inRange(hsv,lower,upper)
        cv2.imshow("抽出フィールド",dst)

        return dst

    #抽出したフィールドのノイズ除去
    def morphology(self,image):
        dst = image
        kernel = np.ones((3,3),np.uint8)

        dst = cv2.erode(dst,kernel,iterations = 15)
        dst = cv2.dilate(dst,kernel,iterations = 120)
        dst = cv2.erode(dst,kernel,iterations = 105)

        # dst = cv2.erode(dst,kernel,iterations = 30)
        # dst = cv2.dilate(dst,kernel,iterations = 30)
        # dst = cv2.dilate(dst,kernel,iterations = 30)
        # dst = cv2.erode(dst,kernel,iterations = 30)

        dst = cv2.cvtColor(dst,cv2.COLOR_GRAY2BGR)
        cv2.imshow("ノイズ除去",dst)

        return dst


    #元画像へのマスク処理
    def masking(self,image,mask):
        gmask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        dst = cv2.bitwise_and(image,image,mask=gmask) 

        cv2.imshow("マスク結果",dst)

        return dst
    
    #エッジ抽出後にハフ変換
    def hough(self,image):
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,30,40,apertureSize=3)
        cv2.imshow("エッジ画像",edges)

        lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=90,minLineLength=10,maxLineGap=50)

        dst = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)

        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(dst,(x1,y1),(x2,y2),(0,0,255),2)

        cv2.imshow("ライン抽出",dst)

        lower = np.array([0,128,0])
        upper = np.array([30,255,255])
        hsv = cv2.cvtColor(dst,cv2.COLOR_BGR2HSV)
        result = cv2.inRange(hsv,lower,upper)

        cv2.imshow("結果画像",result)

        return result

    #メイン関数
    def main(self):
        mask = self.fieldDitection(self.image)
        mask = self.morphology(mask)
        maskedImage = self.masking(self.image,mask)
        line = self.hough(maskedImage)

        cv2.imwrite("./result/resut.jpg",line)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

if __name__=="__main__":
    args = sys.argv
    if len(args) > 1:
        name = args[1]
    else:
        name = "./image/datasets/right/result_00295.jpg"
    image = cv2.imread(name)
    dt = diteciton(image,name)
    dt.main()