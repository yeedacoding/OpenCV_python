# Blurring, Smoothing
# https://setosa.io/ev/image-kernels/
# 목적 : 이미지를 흐리게 blur처리 함으로서 노이지 및 손상을 줄이기 위함
# 컨볼루션 계산 적용
# 1) 이미지의 특정 부분에 커널을 적용하여 컨볼루션 계산을 거쳐 필터링
# 2) 커널 사이즈에 맞게 이미지의 특정 픽셀의 주변 값의 평균값이 해당 픽셀에 대체 적용됨
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/brick.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.pyrUp(img,
                borderType=cv2.BORDER_DEFAULT)

# 감마 교정
# 이미지의 밝기를 조절
# 1보다 작게 하면 밝기 증가

gamma = 0.4
result = np.uint8(255 * np.power(img / 255, gamma))

# plt.imshow(result)
# plt.show()


font = cv2.FONT_HERSHEY_COMPLEX
org = cv2.putText(img, text='brick', org=(180, 520),
                  fontFace=font, fontScale=12, color=(0, 255, 0), thickness=4)

#########################################################################################################
# 직접 커널 만들기
kernel = np.ones(shape=(5, 5), dtype=np.float32)/25
# 5x5 크기의 커널
# 커널의 사이즈(shape)가 넓어질 수록 평균값을 내는 데 있어 더 많은 영역의 픽셀을 활용하므로 더욱 blur처리된 이미지가 됨
# convolution 계산을 할 때 커널을 이미지의 특정 픽셀에 적용 후 다 더한 값을 커널의 총 합으로 나누기 때문에 마지막에 커널의 총합인 25를 나눠준다
print(kernel)

# 이미지에 커널 적용시키기
# 1. 이미지에 2d filter 적용
# 간단하게 커널 안의 픽셀들의 값을 평균을 내는 것, 이 평균값으로 현재 픽셀의 값을 대체
# cv2.filter2D(src, ddepth, kernel, dst=None, anchor=None, delta=None, borderType=None) -> dst
# src : 입력 이미지
# ddepth : 출력 영상 데이터 타입. (e.g) cv2.CV_8U, cv2.CV_32F, cv2.CV_64F, -1을 지정하면 src와 같은 타입의 dst 영상을 생성
# kernel: 필터 마스크 행렬. 실수형.
# anchor: 고정점 위치. (-1, -1)이면 커널 중앙을 고정점으로 사용
# delta: 추가적으로 더할 값
# borderType: 가장자리 픽셀 확장 방식
# dst: 출력 영상
dst = cv2.filter2D(img, -1, kernel)

merged = np.hstack((org, dst))
plt.figure(figsize=(16, 9))
plt.title('cv2.filter2D')
plt.imshow(merged)
plt.show()

#########################################################################################################
# 2. cv2.blur(src, kernel size) 사용하여 자동으로 커널 만들어 적용하기
dst = cv2.blur(org, (8, 8))
# (8,8)크기의 커널을 생성해 자동으로 org 이미지에 적용시켜줌

merged = np.hstack((org, dst))
plt.figure(figsize=(16, 9))
plt.title('cv2.blur')
plt.imshow(merged)
plt.show()

#########################################################################################################
# 3. cv2.GaussianBlur(src, ksize, sigmaX)
# kernel 적용한 이미지 상의 특정 픽셀에 더 많은 가중치가 적용되고 멀리 있는 픽셀은 작은 가중치가 적용됨
# ksize -> 양수의 '홀수'로 지정

gaussian = cv2.GaussianBlur(org, (5, 5), 10)

merged = np.hstack((org, gaussian))
plt.figure(figsize=(16, 9))
plt.title('Gaussian Blur')
plt.imshow(merged)
plt.show()

#########################################################################################################
# 4. cv2.medianBlur(src, ksize)
# 위의 방식들처럼 커널  안의 값들을 지정하지 않고 커널 사이즈만 지정
# 커널에 적용된 이미지 상의 픽셀 주변의 픽셀 값들을 정렬(sort)하여 그 중 중간값으로 해당 픽셀을 대체
# salt-and-pepper noise(소금 후추 노이즈 -> 이미지 상의 희고 검은 불규칙한 노이즈) 제거에 효과적

median = cv2.medianBlur(org, 5)

merged = np.hstack((org, median))
plt.figure(figsize=(16, 9))
plt.title('Median Blur')
plt.imshow(merged)
plt.show()

#########################################################################################################
# 5. cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)
# 지금까지의 Blur처리는 경계선까지 Blur처리가 되어, 경계선이 흐려짐
# Bilateral Filtering(양방향 필터)은 경계선을 유지하면서 Gaussian Blur처리를 해주는 방법
# 이미지의 윤곽은 샤프하게 살려두고 노이즈를 제거하는 데에는 아주 효과적
# Gaussian 필터를 적용하고, 또 하나의 Gaussian 필터를 주변 pixel까지 고려하여 적용하는 방식
# src : 8-bit, 1 or 3 Channel image
# d : filtering시 고려할 주변 pixel 지름
# sigmaColor : Color를 고려할 공간. 숫자가 크면 멀리 있는 색도 고려함
# sigmaSpace : 숫자가 크면 멀리 있는 pixel도 고려함

# noisy한 이미지 가져오기
noisy = cv2.imread('images/noisy.png')
noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)

# medianblur와 bilateralFilter 비교
median = cv2.medianBlur(noisy, 5)
bilateral = cv2.bilateralFilter(noisy, 9, 75, 75)

merged = np.hstack((noisy, median, bilateral))
plt.figure(figsize=(16, 9))
plt.title('original, median blur, bilateral filter')
plt.imshow(merged)
plt.show()
