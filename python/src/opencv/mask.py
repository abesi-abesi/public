import sys
import cv2
import numpy as np

class mask:

    def __init__(self,src,mask):
        self.src = src
        self.mask = mask
    
    def masking(self):
        image = cv2.imread(self.src,1)
        mask_image = cv2.imread(self.mask,0)

        mImage = cv2.bitwise_and(image,image,mask=mask_image) 

        cv2.imshow("元画像",image)
        cv2.imshow("マスク画像",mask_image)
        cv2.imshow("結果画像",mImage)

        if cv2.waitKey() == ord("s"):
            cv2.imwrite("./result/maskedImage.png",mImage)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

args = sys.argv
if len(args) == 2:
    mask = mask(args[1],"./result/ml-color-angle1.png")

elif len(args) == 3:
    mask = mask(args[1],args[2])

else:
    mask = mask("./image/angle1.png","./result/ml-color-angle1.png")

mask.masking()