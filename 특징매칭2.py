# Keypoints drawing
# Feature Matching 이전의 사전 공부 2단계
# 검출한 keypoints를 이미지 상에 그려주기

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/read_color.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 이전 공부했던 순서대로 진행

# 1. 특징 검출 객체 생성
orb = cv2.ORB_create()

# 2. keypoints, descriptors 검출
keypoints, descriptors = orb.detectAndCompute(img, None)

# 3. cv2.drawKeypoints(image, keypoints, outImage, color=None, flags=None) -> outImage
# image = 입력 영상
# keypoints = 검출된 특징점 정보, cv2.KeyPoint 객체의 리스트
# outImage = 출력 영상
# color = 특징점 표현 색상, default=(-1,-1,-1,-1)일 경우 임의의 색상으로 표현
# flags = 특징점 표현 방법 :
# cv2.DRAW_MATCHES_FLAGS_DEFAULT : 좌표 중심에 동그라미만 그림
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS : 특징점의 크기와 방향을 반영한 원

des1 = cv2.drawKeypoints(img, keypoints, None,
                         flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
des2 = cv2.drawKeypoints(img, keypoints, None,
                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

merged = np.hstack((des1, des2))

plt.figure(figsize=(14, 8))
plt.imshow(merged)
plt.show()
