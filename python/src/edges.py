import cv2

class edges:

    def cannyEdges(self,name="./image/sample.png"):
        image = cv2.imread(name)
        cv2.imshow("元画像",image)
        dst = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(dst,75,25,3)
        cv2.imshow("結果画像",edges)

        cv2.waitKey(0)
        cv2.destroyAllWindow() 

edges = edges()
edges.cannyEdges()
