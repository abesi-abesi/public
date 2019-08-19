import cv2
import os
import matplotlib.pyplot as plt

def main():
	TARGET_FILE = "result1.png"
	IMG_DIR = "./orbResult/"

	bf = cv2.BFMatcher(cv2.NORM_HAMMING)
	# 特徴点算出のアルゴリズムを決定(コメントアウトで調整して切り替え)
	detector = cv2.ORB_create()
	#detector = cv2.AKAZE_create()
	(target_kp, target_des) = calc_kp_and_des(IMG_DIR + TARGET_FILE, detector)

	print('TARGET_FILE: %s' % (TARGET_FILE))

	files = os.listdir(IMG_DIR)
	for file in files:
	    if file == '.DS_Store' or file == TARGET_FILE:
	        continue

	    comparing_img_path = IMG_DIR + file
	    try:
	        (comparing_kp, comparing_des) = calc_kp_and_des(comparing_img_path, detector)
			#画像同士をマッチング
	        matches = bf.match(target_des, comparing_des)
	        dist = [m.distance for m in matches]
			#類似度を計算する
	        ret = sum(dist) / len(dist)
	    except cv2.error:
	        ret = 100000

	    print(file, ret)

def calc_kp_and_des(img_path, detector):
	"""
		特徴点と識別子を計算する
		:param str img_path: イメージのディレクトリパス
		:param detector: 算出の際のアルゴリズム
		:return: keypoints
		:return: descriptor
	"""
	IMG_SIZE = (200, 200)
	img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	img = cv2.resize(img, IMG_SIZE)
	return detector.detectAndCompute(img, None)


# def show_imgs_match():
# 	img1 = cv2.imread('./result/resultLine.png')
# 	img2 = cv2.imread('./result/resultLine1.png')
# 	gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# 	gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# 	akaze = cv2.ORB_create()
# 	kp1, des1 = akaze.detectAndCompute(gray1, None)
# 	kp2, des2 = akaze.detectAndCompute(gray2, None)

# 	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# 	matches = bf.match(des1, des2)
# 	matches = sorted(matches, key = lambda x:x.distance)
# 	img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
# 	plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
# 	plt.show()


if __name__ == '__main__':
    main()
    # show_imgs_match()