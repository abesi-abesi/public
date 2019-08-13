import sys
import cv2
import numpy as np

class morphology:


    def denoise(self,image):
        dst = image
        kernel = np.ones((3,3),np.uint8)

        dst = cv2.erode(dst,kernel,iterations = 15)
        dst = cv2.dilate(dst,kernel,iterations = 120)
        dst = cv2.erode(dst,kernel,iterations = 105)
        dst = cv2.cvtColor(dst,cv2.COLOR_GRAY2BGR)
        cv2.imshow("ノイズ除去",dst)

        # if cv2.waitKey() == ord("s"):
        #     num = self.name.rfind("/")
        #     str = "./result/ml-" + self.name[num+1:]
        #     cv2.imwrite(str,dst)
        
        return dst
        # while cv2.waitKey() != ord("q"):
        #     cv2.imshow("結果画像",dst)
            
        #     if cv2.waitKey() == ord("e"):
        #         dst = cv2.erode(dst,kernel,iterations = 1)

        #     if cv2.waitKey() == ord("d"):
        #         dst = cv2.dilate(dst,kernel,iterations = 1)

        #     if cv2.waitKey() == ord("s"):
        #         num = self.name.rfind("/")
        #         str = "./result/ml-" + self.name[num+1:]
        #         cv2.imwrite(str,dst)
        
        
if __name__ == "__main__":
    args = sys.argv
    mp = morphology()
    if len(args) > 1:
        image = cv2.imread(args[1])
    else:
        image = cv2.imread("./result/color-angle1.png")
    
    mp.denoise(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()