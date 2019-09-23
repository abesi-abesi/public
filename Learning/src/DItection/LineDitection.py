import cv2
import numpy as np
import sys
import glob

class diteciton:

    #コンストラクタ
    def __init__(self,name):
        self.name = name

    #フィールド部分の抽出    
    def fieldDitection(self,image):
        lower = np.array([35,50,50])
        upper = np.array([80,255,255])

        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        dst = cv2.inRange(hsv,lower,upper)
        #cv2.imshow("抽出フィールド",dst)

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
        #cv2.imshow("ノイズ除去",dst)

        return dst


    #元画像へのマスク処理
    def masking(self,image,mask):
        gmask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        dst = cv2.bitwise_and(image,image,mask=gmask) 

        #cv2.imshow("マスク結果",dst)

        return dst
    
    #エッジ抽出後にハフ変換
    def hough(self,image):
        #２値化
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #エッジ抽出
        edges = cv2.Canny(gray,30,40,apertureSize=3)
        #平滑化
        #edges = cv2.GaussianBlur(edges,(1,1),10)

        #ノイズ除去
        # kernel = np.ones((3,3),np.uint8)
        # edges = cv2.dilate(edges,kernel,iterations = 3)
        #edges = cv2.erode(edges,kernel,iterations = 3)

        #cv2.imshow("エッジ画像",edges)

        lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=90,minLineLength=50,maxLineGap=30)

        dst = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)

        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(dst,(x1,y1),(x2,y2),(0,0,255),2)

        #cv2.imshow("ライン抽出",dst)

        lower = np.array([0,128,0])
        upper = np.array([30,255,255])
        hsv = cv2.cvtColor(dst,cv2.COLOR_BGR2HSV)
        result = cv2.inRange(hsv,lower,upper)

        #cv2.imshow("結果画像",result)

        return result

    #メイン関数
    def main(self):
        count = 1
        for x in glob.glob("./image/datasets/" + name + "/*.jpg"):

            image = cv2.imread(x)
            
            mask = self.fieldDitection(image)
            mask = self.morphology(mask)
            maskedImage = self.masking(image,mask)
            line = self.hough(maskedImage)

            cv2.imwrite("./result/" + name + "/line_" + str(count) + ".jpg",line)
            count += 1

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        

if __name__=="__main__":
    args = sys.argv
    if len(args) > 1:
        name = args[1]
    else:
        name = "test"

    dt = diteciton(name)
    dt.main()