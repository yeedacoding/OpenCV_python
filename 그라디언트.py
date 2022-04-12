# Gradient
# edge 검출을 위함(객체 검출 등)
# 이미지 상에서 gradient는 이미지의 내 객체의 edge 및 edge의 방향을 찾는 용도
# 이미지 상에서 (x,y)좌표에서의 벡터값(크기와 방향, 즉 밝기와 밝기가 변화하는 방향)을 구해 해당 픽셀이 edge로부터 얼마나 가까운지, 변화하는 방향은 어디인지 구할 수 있음
# 방법 : 픽셀값이 급격하게 변화하는 지점을 찾기
# 이미지의 픽셀값들을 1차원 그래프로 나타내보면 급격하게 픽셀값이 변화하는 부분이 있는데, 이 부분을 미분하면 주변보다 미분값이 크게 나타남
# 연속되는 픽셀 값을 미분하여 미분값이 큰 부분을 edge로서 간주
# 픽셀이라는 것은 연속 공간에 있지 않으므로 미분 근사값을 구해야하는데 간단하게 x 또는 y 방향으로 서로 붙어있는 픽셀값을 뺴면 됨
# 연산 방법 : x방향 혹은 y방향의 붙어있는 픽셀값들을 빼도록 연산할 수 있는 컨볼루션 커널을 만들어 적용하기

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/brick.jpg', 0)

#########################################################################################################
# 커널을 직접 만들어서 적용해보기
# 컨볼루션 연산의 원리를 적용하는 것이기 때문에 filter2D를 사용해보기
# gx_kernel = x방향 그라디언트 / gy_kernel = y방향 그라디언트
# -1, 1인 이유 =붙어있는 두 픽셀에서 전 픽셀값과 후 픽셀값을 빼서 미분 근사값을 구하기 때문
gx_kernel = np.array([[-1, 1]])
gy_kernel = np.array([[-1], [1]])

gx_edge = cv2.filter2D(img, -1, gx_kernel)
gy_edge = cv2.filter2D(img, -1, gy_kernel)

merged = np.hstack((img, gx_edge, gy_edge))
plt.figure(figsize=(18, 10))
plt.title('original, gx, gy gradient')
plt.imshow(merged, cmap='gray')
plt.show()

#########################################################################################################
# 1. 소벨 필터
# 인접한 픽셀들의 차이로 기울기(Gradient)의 크기를 구함
# 이때 인접한 픽셀들의 기울기를 계산하기 위해 컨벌루션 연산을 수행(1차 미분)

# 3x3 커널 2개를 사용
# 하나는 수평변화(x방향으로의 변화), 다른 하나는 수직변화(y방향으로의 변화)값을 계산

# cv2.Sobel(src, ddepth, dx, dy, dst, ksize, scale, delta, borderType)
# src: 입력 영상
# ddepth: 출력 영상의 dtype (-1: 입력 영상과 동일)
# dx, dy: 미분 차수 (0, 1, 2 중 선택, 둘 다 0일 수는 없음)
# ksize: 커널의 크기 (1, 3, 5, 7 중 선택)
# scale: 미분에 사용할 계수
# delta: 연산 결과에 가산할 값

# x방향, y방향 각각 sobel 필터를 적용하여 보기
sobelx = cv2.Sobel(img, -1, 1, 0, ksize=3)
sobely = cv2.Sobel(img, -1, 0, 1, ksize=3)
# sobelx -> x방향 미분필터를 적용했기 때문에 x방향(좌,우) 픽셀의 변화를 측정하여 세로선 경게검출에 용이
# sobelx -> y방향 미분필터를 적용했기 때문에 y방향(상,하) 픽셀의 변화를 측정하여 가로선 경게검출에 용이

merged = np.hstack((img, sobelx, sobely))
plt.figure(figsize=(18, 10))
plt.title('original, soble x-gradient, soble y-gradient')
plt.imshow(merged, cmap='gray')
plt.show()

# sobelx, sobely 이미지 합쳐서 보기(blending)
blend = cv2.addWeighted(src1=sobelx, alpha=0.5, src2=sobely, beta=0.5, gamma=5)
plt.figure(figsize=(12, 8))
plt.title('sobel-x + soble-y')
plt.imshow(blend, cmap='gray')
plt.show()

# 합친 이미지 임계처리 해보기(thresholding)
ret, thresh = cv2.threshold(blend, 127, 255, cv2.THRESH_BINARY)
plt.figure(figsize=(12, 8))
plt.title('threshold sobel image')
plt.imshow(thresh, cmap='gray')
plt.show()

# morphology 적용해보기
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
plt.figure(figsize=(12, 8))
plt.title('closing')
plt.imshow(morph, cmap='gray')
plt.show()

#########################################################################################################
# 2. 라플라시안 필터
# 2차 미분을 적용한 필터
# 소벨은 1차 미분으로, 에지 유형에서 기울기에 따라 에지가 결정
# 라플라시안 필터는 2차 미분, 즉 기울기의 기울기 값으로, 조금 더 정확한 에지를 검출하기 위해 사용

# x,y 방향에 대한 연산을 둘 다 사용하기 때문에 커널을 두 개 만들 필요없음
laplacian = cv2.Laplacian(img, -1, ksize=3)

plt.figure(figsize=(12, 8))
plt.title('laplacian filter')
plt.imshow(laplacian, cmap='gray')
plt.show()
