import sys
import cv2
import numpy as np
import colorDitection,morphology,mask,hough

class main:

    def __init__(self,image):
        self.image = image

    def main(self):
        cd = colorDitection.colorDitection()
        mp = morphology.morphology()
        mk = mask.mask()
        hg = hough.hough()

        cImage = cd.ditection(self.image)
        mpImage = mp.denoise(cImage)
        mkImage = mk.masking(self.image,mpImage)
        hg.conversion(mkImage)





if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        image = cv2.imread(args[1])
    else:
        image = cv2.imread("./image/angle1.png")
    main = main(image)
    main.main()
    cv2.imshow("元画像",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()