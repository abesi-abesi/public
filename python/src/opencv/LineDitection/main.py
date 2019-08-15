import sys
import cv2
import numpy as np
import colorDitection,morphology,mask,hough,houghEx

class main:

    def __init__(self,image):
        self.image = image

    def main(self):
        cd = colorDitection.colorDitection()
        mp = morphology.morphology()
        mk = mask.mask()
        hg = hough.hough()
        he = houghEx.houghEx()

        cImage = cd.ditection(self.image)
        mpImage = mp.denoise(cImage)
        mkImage = mk.masking(self.image,mpImage)
        hg = hg.conversion(mkImage)
        he.ditection(hg)





if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        image = cv2.imread(args[1])
    else:
        image = cv2.imread("./image/angle1.png")
    main = main(image)
    height, width, channels = image.shape[:3]
    print("width: " + str(width))
    print("height: " + str(height))
    cv2.imshow("元画像",image)
    main.main()
    cv2.waitKey(0)
    cv2.destroyAllWindows()