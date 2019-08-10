import sys
import cv2
import numpy as np

class morphology:

    def __init__(self,name):
        self.name = name

    def denoise(self):
        image = cv2.imread(self.name)
        dst = image
        kernel = np.ones((5,5),np.uint8)
        cv2.imshow("元画像",image)
        
        while cv2.waitKey() != ord("q"):
            cv2.imshow("結果画像",dst)
            
            if cv2.waitKey() == ord("e"):
                dst = cv2.erode(dst,kernel,iterations = 1)

            if cv2.waitKey() == ord("d"):
                dst = cv2.dilate(dst,kernel,iterations = 1)

        cv2.destroyAllWindows()

args = sys.argv
if len(args) > 1:
    mp = morphology(args[1])
else:
    mp = morphology("./image/angle1.png")

mp.denoise()