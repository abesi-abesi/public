import sys
import cv2

class edges:

    def __init__(self,name):
        self.name = name

    def cannyEdges(self):
        image = cv2.imread(self.name)
        cv2.imshow("元画像",image)
        dst = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(dst,75,25,3)
        cv2.imshow("結果画像",edges)

        if cv2.waitKey() == ord("s"):
            num = self.name.rfind("/")
            str = "./result/edges-" + self.name[num+1:]
            print(str)
            cv2.imwrite(str,edges)

        cv2.waitKey(0)
        cv2.destroyAllWindows() 

args = sys.argv
if len(args) > 1:
    edges = edges(args[1])
else:
    edges = edges("./image/angle1.png")

edges.cannyEdges()
