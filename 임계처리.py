# Thresholding
# 셔츠의 색깔은 필요없고, 윤곽 정보만 필요할 때 사용
# 흑백 이진화를 통해 이미지의 특정 부분을 나눔
# 유색 이미지 -> grayscale로 변경 -> 이진화 적용(binary) -> 흑백값만 존재하는 이진 이미지로 바꾸기

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('images/read_color.jpg', 0)

plt.imshow(img, cmap='gray')
plt.show()

# cv2.threshold(src, thresh, maxval)
# src = 임계 처리할 이미지
# thresh = 지정할 임계값
# maxval = thresh 값보다 낮은 값은 0, 높은 값은 maxval로 전환
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# cv2.THRESH_BINARY = 이진화
# cv2.THRESH_BINARY_INV = 이진화된 흑백 부분 반전

# ret = 실제 임계값
print(ret)

# thresh1 = 이진화(binary)된 이미지
print(thresh1)
plt.imshow(thresh1, cmap='gray')
plt.show()

#############################################################
# cv2.threshold() -> 물체가 주위와 구분이 잘 되는 경우 사용 가능, 경계가 애매하면 사용하기 어려움

# 적당한 임계값 찾는 법
# 1. thresh parameter 값을 일일이 조정해보기
# 2. 적응형 이진화 사용

# 적응형 이진화 = 임계값을 자동으로 픽셀과 그 주변 회색 픽셀에 기반해 조정
# 이미지를 여러 영역으로 나눈 뒤, 그 주변 픽셀 값만 활용하여 임계값을 구하는 방법

# cv2.adaptiveThreshold(img, value, method, type_flag, block_size, C)
# img: 입력영상
# value: 임계값을 만족하는 픽셀에 적용할 값
# method: 임계값 결정 방법
# type_flag: 스레시홀딩 적용 방법 (cv2.threshod()와 동일)
# block_size: 영역으로 나눌 이웃의 크기(n x n), 홀수
# C: 계산된 임계값 결과에서 가감할 상수(음수 가능)

# method 종류
# cv2.ADAPTIVE_THRESH_MEAN_C: 이웃 픽셀의 평균으로 결정
# cv2.ADAPTIVE_THRESH_GAUSSIAN_C: 가우시안 분포에 따른 가중치의 합으로 결정


th2 = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)

plt.imshow(th2, cmap='gray')
plt.show()

th3 = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 8)

plt.imshow(th3, cmap='gray')
plt.show()
