import sys
import cv2
import numpy as np

class houghEx:


    def ditection(self,image):
        lower = np.array([0,128,0])
        upper = np.array([30,255,255])
        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        dst = cv2.inRange(hsv,lower,upper)

        # if cv2.waitKey() == ord("s"):
        #     num = name.rfind("/")
        #     str = "./result/color-" + name[num+1:]
        #     cv2.imwrite(str,dst)

        cv2.imshow("ラインのみ",dst)
        cv2.imwrite("./result/resultLine.png",dst)
        return dst
        
if __name__ == "__main__":
    args = sys.argv
    cd = houghEx()
    if len(args) > 1:
        image = cv2.imread(args[1])
    else:
        image = cv2.imread("./image/angle1.png")

    cd.ditection(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()