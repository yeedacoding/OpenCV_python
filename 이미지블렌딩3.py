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

x_offset = 1280 - 300
y_offset = 854 - 300

print(img2.shape)

rows, cols, channels = img2.shape

roi = img1[y_offset:854, x_offset:1280]
# plt.imshow(roi)
# plt.show()

# img2에서 빨간 부분 마스킹
img2gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

plt.imshow(img2gray, cmap='gray')
plt.show()

mask_inv = cv2.bitwise_not(img2gray)

plt.imshow(mask_inv, cmap='gray')
plt.show()
print(mask_inv.shape)
white_background = np.full(img2.shape, 255, dtype=np.uint8)

bk = cv2.bitwise_or(white_background, white_background, mask=mask_inv)
print(bk.shape)

# plt.imshow(bk)
# plt.show()

fg = cv2.bitwise_or(img2, img2, mask=mask_inv)
plt.imshow(fg)
plt.show()

final_roi = cv2.bitwise_or(roi, fg)
# plt.imshow(final_roi)
# plt.show()

large_img = img1
small_img = final_roi

large_img[y_offset:y_offset+small_img.shape[0],
          x_offset:x_offset+small_img.shape[1]] = small_img

plt.imshow(large_img)
plt.show()
