import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10 
good_match_rate = 0.15 #得られた特徴点のうち使用する点の割合

img1 = cv2.imread("./result/result1.png",0) # 一枚目
img2 = cv2.imread("./result/result3.png",0) # 二枚目

# Initiate ORB detector
orb = cv2.ORB_create()

# キーポイント検出,ORB記述
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFmatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
# Match descriptors.
matches = bf.match(des1,des2)


matches = sorted(matches, key = lambda x:x.distance)
good = matches[:int(len(matches) * good_match_rate)]

#MIN_MATCH_COUNT以上なら出力,それ以下ならelseへ
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
#RANSAC
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print ("Not enough matches are found - %d/%d") % (len(good),MIN_MATCH_COUNT)
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

#出力
plt.imshow(img3, 'gray'),plt.show()