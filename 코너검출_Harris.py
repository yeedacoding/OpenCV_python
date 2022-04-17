# Corner Detection
# 코너 = 두 에지의 교차점, 에지가 급격히 변해 밝기가 바뀌는 곳
# 이미지 위에 작은 window(mask)를 써서 이 window가 이리저리 움직였을 때, 그 window 안의 intensity 값들의 차이가 큰 점을 corner로 인지
# intensity 값 차이 : window내 각 픽셀의 값 차이를 더한 값
# u,v 만큼 shift된 window 내 픽셀의 intensity와 shift가 되지 않은 window 내의 픽셀의 intensity 값의 차이를 제곱하여 모두 합한다
# 이 값이 어느 쪽으로든 변화가 작으면 flat, 한 쪽으로만 변화가 있으면 edge, 어느 쪽으로든 변화가 크면 corner로 검출

# flat region (corner나 edge가 없는 지역): no change in all directions(어느 쪽으로든 window가 움직이나 intensity 값의 차이가 없다)
# edge : no change along the edge direction(edge가 있는 방향으로 움직였을 때 intensity값 차이가 없으나 다른 방향으로 움직였을 때 값 차이가 있음)
# corner : significant change in all directions(어느 쪽으로 움직이든 intensity 값 차이가 크다)

# 1. 해리스 코너 검출(Harris Corner Detection)
# 소벨 미분으로 경곗값을 검출하면서 경곗값의 경사도 변화량을 측정하여 변화량이 수직, 수평, 대각선 등 전 방향으로 크게 변화하는 부분을 corner로 판단

# dst = cv2.cornerHarris(src, blockSize, ksize, k, dst, borderType)
# src: 입력 이미지, 그레이 스케일, float32 타입이어야 함
# blockSize: 코너 검출을 위해 고려할 이웃 픽셀 범위
# ksize: 소벨 미분 필터 크기
# k(optional): 코너 검출 상수 (보통 0.04~0.06로 정함)
# dst(optional): 코너 검출 결과 (src와 같은 size의 1 채널 배열, *변화량의 값, 지역 최대 값이 코너점을 의미*)
# borderType(optional): 외곽 영역 보정 형식


import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/house.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.cornerHarris()의 src 이미지는 grayscale에 float32타입이어야 함
gray = np.float32(gray_img)

# 해리스 코너 함수 적용
dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)

# 최적화(Thresholding)
# dst(코너검출 결과=변화량의 값)의 최댓값의 0.01보다 큰 부분은 corner로서 검출하도록 하기
corner = np.where(dst > 0.01 * dst.max())
print(corner)   # 두 배열이 반환

# corner로 검출된 값들을 x,y좌표로서 재배열
corner = np.stack((corner[1], corner[0]), axis=1)
print(corner)

# corner의 x,y좌표에 circle 그려주기
for x, y in corner:
    cv2.circle(img, (x, y), 10, (255, 0, 0), 1)

plt.imshow(img)
plt.show()
