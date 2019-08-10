import sys
import cv2
import numpy as np

class colorDitection:

    def __init__(self,name):
        self.name = name

    def ditection(self):
        lower = np.array([40,50,50])
        upper = np.array([80,255,255])
        image = cv2.imread(self.name)
        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        dst = cv2.inRange(hsv,lower,upper)

        cv2.imshow("元画像",image)
        cv2.imshow("HSV画像",hsv)
        cv2.imshow("結果画像",dst)

        if cv2.waitKey() == ord("s"):
            num = self.name.rfind("/")
            str = "./result/color-" + self.name[num+1:]
            cv2.imwrite(str,dst)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

args = sys.argv
if len(args) > 1:
    cd = colorDitection(args[1])
else:
    cd = colorDitection("./image/angle1.png")

cd.ditection()