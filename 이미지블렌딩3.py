# 마스크
# 한 이미지의 일부분만 다른 이미지에 합성하기

import cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread('images/read_color.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('images/nocopy.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

img1 = cv2.resize(img1, (0, 0), img1, 2, 2)
img2 = cv2.resize(img2, (300, 300))


# roi 설정
# img1에서 roi설정 후 img2에 합성해보기
# img1 = (854, 1280, 3)
print(img1.shape)
print(img2.shape)

# roi는 img1(원본) 상에서 img2(logo)만큼의 size로 지정
x_offset = 1280 - 300
y_offset = 854 - 300

rows, cols, channels = img2.shape

roi = img1[y_offset:854, x_offset:1280]
# plt.imshow(roi)
# plt.show()


# img2에서 빨간 부분 마스킹

# logo img를 binary 이미지로 바꾸기
img2gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

ret, mask = cv2.threshold(img2gray, 60, 255, cv2.THRESH_BINARY)

plt.imshow(mask, cmap='gray')
plt.show()

# 색상 반전
mask_inv = cv2.bitwise_not(mask)

plt.title('mask inverted')
plt.imshow(mask_inv, cmap='gray')
plt.show()

# roi 이미지에서 mask_inv(logo부분이 검은색 = 0)만큼 0값을 부여 = roi 이미지에서 logo 모양을 뚫고 그 안에 다시 logo를 채움
bk = cv2.bitwise_and(roi, roi, mask=mask_inv)

plt.title('bk')
plt.imshow(bk)
plt.show()

# img2(원본 logo)에 mask(logo를 제외한 부분 = 0)을 적용해 기존 logo(빨강색)을 제외한 부분에 0값을 부여하여 logo만 살리기
fg = cv2.bitwise_and(img2, img2, mask=mask)

plt.title('fg')
plt.imshow(fg)
plt.show()

# bk = roi 이미지에서 logo 모양만 0 값, fg = 기존 logo 이미지에서 logo 모양만 원본(빨강)으로 남겨둠
# bk와 fg를 합치면 roi에 뚫려있던 logo모양이 채워짐
final_roi = cv2.add(bk, fg)

plt.title('final roi')
plt.imshow(final_roi)
plt.show()

large_img = img1
small_img = final_roi

large_img[y_offset:y_offset+small_img.shape[0],
          x_offset:x_offset+small_img.shape[1]] = small_img

plt.imshow(large_img)
plt.show()
