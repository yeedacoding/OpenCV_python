# Morphology
# 노이즈 제거, 구멍 채우기, 끊어진 선 이어 붙이기, 붙어있는 선 분리하기 등의 목적으로 사용되는 형태학적 연산
# 흰색, 검은색의 binary 이미지나 grayscale에 적용
# 문자가 그려진 이미지 처리에 효과적
# 사용하는 방법 : erosion, dilation, opening, closing

import cv2
import matplotlib.pyplot as plt
import numpy as np


# 문자가 그려진 이미지 만들기
blank_img = np.zeros((600, 600))
font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.putText(blank_img, text='ABCDE', org=(50, 300),
                  fontFace=font, fontScale=5, color=(255, 255, 255), thickness=25)

plt.figure(figsize=(12, 10))
plt.imshow(img, cmap='gray')
plt.show()

#########################################################################################################
# * structuring element *
# 0,1로 구성된 kernel
# 1이 채워진 모양에 따리 사각형, 타원형, 십자형 등이 있음

# 1. 직접 만들기
kernel = np.ones((5, 5), dtype=np.uint8)
print(kernel)

# 2. cv2.getStructuringElement(shape, ksize[, anchor])
# shape – Element의 모양
# MORPH_RECT : 사각형 모양
# MORPH_ELLIPSE : 타원형 모양
# MORPH_CROSS : 십자 모양
# ksize : structuring element 사이즈
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
print(kernel1)


#########################################################################################################
# 1. erosion(침식)
# 이미지 내 물체의 형태를 깎아내리는 연산
# structuring element를 이미지에 적용 -> structuring element 모양과 한 부분이라도 겹쳐지지 않으면 해당 픽셀은 0으로 대체
# 물체 주변을 깎음, 작은 물체는 아예 없애 노이즈 제거 효과, 겹쳐있는 것을 떨어뜨리는 효과

# cv2.erode(src, kernel, anchor, iterations, borderType, borderValue)
# src: 입력 영상, 바이너리
# kernel: 구조화 요소 커널
# anchor(optional): cv2.getStructuringElement()와 동일, structuring element의 중심점, default = (-1,-1)
# iterations(optional): 침식 연산 적용 반복 횟수

erosion = cv2.erode(img, kernel, iterations=1)
# iteration(반복)이 커질 수록 더 많이 깎아냄

#########################################################################################################
# 2. dilation(팽창)
# 침식과 반대로 물체의 주변을 확장시키는 연산
# structuring element와 겹치는 부분이 하나라도 있으면 kernel 모양만큼 확장(OR연산)
# 경계 부드러워짐, 중간중간 작은 구멍 메워지는 효과

dilation = cv2.dilate(img, kernel, iterations=1)

plt.figure(figsize=(12, 10))
plt.title('org, erosion, dilation')
merged = np.hstack((img, erosion, dilation))
plt.imshow(merged, cmap='gray')
plt.show()


#########################################################################################################
# *noise 만들기*

# white_noise = 원본 이미지의 문자부분과 배경부분 모든 영역에 noise 제공
white_noise = np.random.randint(low=0, high=2, size=(600, 600))
# 원본 이미지(0 or 255로 이루어짐)와 scale을 맞추기 위해 noise * 255
white_noise = white_noise * 255
# noise와 원본 이미지 합치기
noise_img = white_noise + img

# black_noise = 원본 이미지의 문자부분에만 노이즈 제공
black_noise = np.random.randint(low=0, high=2, size=(600, 600))
# 원본 이미지는 0과 255로만 이루어져있음
# 문자부분만 255 이므로 문자 부분에서 랜덤하게 255를 뺴면 0(검은색)이 되므로 noise * -255
black_noise = black_noise * -255
black_noise_img = black_noise + img
# 원본이미지에서 원래 검정색인 배경의 값이 -255가 되므로 black_noise_img에서 -255인 값은 다시 0으로 만들어 배경색과 맞춰주기
black_noise_img[black_noise_img == -255] = 0


#########################################################################################################
# opening, closing
# erosion, dilation을 조합하여 연산
# erosion, dilation만 하면 이미지 상 원래 물체 모양이 더 커지거나 작아지는데 이를 방지하면서 노이즈 제거 목적


# 1. Opening
# erosion 적용 후 dilation 적용
# 작은 object, 돌기 제거, 이어져있는 것처럼 보이는 물체 분리해주는 효과
opening = cv2.morphologyEx(noise_img, cv2.MORPH_OPEN, kernel)


#########################################################################################################
# 2. Closing
# dilation 후 erosion 적용
# 전체적인 윤곽 파악에 적합, 구멍 메우기, 끊어져보이는 것 연결
closing = cv2.morphologyEx(black_noise_img, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(12, 10))
plt.title('org, opening, closing')
merged = np.hstack((img, opening, closing))
plt.imshow(merged, cmap='gray')
plt.show()

#########################################################################################################
# 3. Gradient
# dilation을 적용한 이미지에서 erosion을 적용한 이미지를 빼면 경계 픽셀만 얻게 됨
# gradient = dilation - erotion

gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

plt.figure(figsize=(12, 10))
plt.title('org, gradient')
merged = np.hstack((img, gradient))
plt.imshow(merged, cmap='gray')
plt.show()
