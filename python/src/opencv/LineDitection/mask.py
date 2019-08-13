import sys
import cv2
import numpy as np

class mask:
    
    def masking(self,image,mask_image):
        mask_gImage = cv2.cvtColor(mask_image,cv2.COLOR_BGR2GRAY)
        mImage = cv2.bitwise_and(image,image,mask=mask_gImage) 

        cv2.imshow("マスク結果",mImage)

        return mImage

        # if cv2.waitKey() == ord("s"):
        #     cv2.imwrite("./result/maskedImage.png",mImage)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    args = sys.argv
    mask = mask()
    if len(args) == 2:
        image = cv2.imread(args[1],1)
        mask_image = cv2.imread("./result/ml-color-angle1.png")

    elif len(args) == 3:
        image = cv2.imread(args[1],1)
        mask_image = cv2.imread(args[2])


    else:
        image = cv2.imread("./image/angle1.png",1)
        mask_image = cv2.imread("./result/ml-color-angle1.png")

    mask.masking(image,mask_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()