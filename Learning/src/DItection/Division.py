import numpy as np
import cv2
import sys

class Division:
    def Div(self,name):
        # フレームに分けたい動画ファイルを指定
        cap = cv2.VideoCapture(name)

        i=0

        while(True):
            # フレームを取得
            ret, frame = cap.read()

            # フレームを保存（'名前'＋番号＋'拡張子'）
            cv2.imwrite('./image/original2/frame_' + str("{0:05d}".format(i)) +'.jpg',frame)

            i+=1
            print(i)

        #　動画の終わりで終了
            if not ret:
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__=="__main__":
    args = sys.argv
    if len(args) > 1:
        path = args[1]
    else:
        path = "/home/yamakawa/Develop/Learning/image/movie/data2.mp4"

    division = Division()
    division.Div(path)
