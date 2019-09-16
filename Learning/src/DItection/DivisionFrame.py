import numpy as np
import cv2

# フレームに分けたい動画ファイルを指定
cap = cv2.VideoCapture("/Users/yuki_yamakawa/Desktop/public/Learning/movie/GOALS_HIGHLIGHTS1.mp4")

i=0

while(True):
    # フレームを取得
    ret, frame = cap.read()

    # フレームを保存（'名前'＋番号＋'拡張子'）
    cv2.imwrite('frame'+str(i)+'.jpg',frame)

    i+=1
    print(i)

#　動画の終わりで終了
    if not ret:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()