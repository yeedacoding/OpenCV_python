# 이미지에 mask 적용하여 ROI 부분에 대한 히스토그램 값만 추출하기

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/colorchannel.jpg')
show_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 10))
plt.imshow(show_img)
plt.show()

# roi 지정을 위해 mask 만들기
mask = np.zeros(img.shape[:2], np.uint8)


# 특정 지점 지정하여 흰색으로 바꾸기
mask[180:200, 260:400] = 255
plt.figure(figsize=(12, 10))
plt.imshow(mask, cmap='gray')
plt.show()

# bit 연산으로 원본 이미지와 마스크 이미지 합치기
masked_img = cv2.bitwise_and(img, img, mask=mask)
# matplotlib으로 보여주기 위해 또 다른 변수 지정 (opencv와 matplotlib의 컬러채널 순서가 다르기 때문)
# masked_img -> 히스토그램 계산을 위한 이미지 / show_masked -> matplotlib으로 보여주기 위한 이미지
show_masked = cv2.bitwise_or(show_img, show_img, mask=mask)

plt.figure(figsize=(12, 10))
plt.imshow(show_masked, cmap='gray')
plt.show()

# RESULT = red 색상이 거의 없는 masked img 생성

# 마스크 된 이미지의 red 색상 히스토그램 값 확인하기
hist_mask_val = cv2.calcHist([img], [2], mask, [256], [0, 256])
hist_val = cv2.calcHist([img], [2], None, [256], [0, 256])

plt.title('masked img red histogram')
plt.plot(hist_mask_val)
plt.show()
plt.title('original img red histogram')
plt.plot(hist_val)
plt.show()
