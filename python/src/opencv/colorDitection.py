import cv2
import numpy as np

class colorDitection:

    def ditection(self,name="./image/angle1.png"):
        lower = np.array([40,50,50])
        upper = np.array([80,255,255])
        image = cv2.imread(name)
        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        dst = cv2.inRange(hsv,lower,upper)

        cv2.imshow("元画像",image)
        cv2.imshow("HSV画像",hsv)
        cv2.imshow("結果画像",dst)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

cd = colorDitection()
cd.ditection()