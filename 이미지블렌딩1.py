# 두 개의 이미지 합치기(blend)
# new_pixel = (alpha * pixel1) + (beta * pixel2) + gamma

import cv2
import matplotlib.pyplot as plt

# 1. 같은 크기의 이미지 blending
img1 = cv2.imread('images/read_color.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('images/nocopy.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# plt.imshow(img1)
# plt.show()
# plt.imshow(img2)
# plt.show()

# 크기 맞추기
print(img1.shape)
print(img2.shape)

img1 = cv2.resize(img1, (500, 600))
img2 = cv2.resize(img2, (500, 600))

plt.imshow(img1)
plt.show()
plt.imshow(img2)
plt.show()


# cv2.addWeighted(src1, alpha, src2, beta, gamma)
blend_img = cv2.addWeighted(img1, 0.5, img2, 0.5, gamma=0)
plt.imshow(blend_img)
plt.show()

# alpha, beta값 조정해보기
# 1) src1 이미지 더 진하게 = alpha값 크게
blend_img2 = cv2.addWeighted(img1, 0.8, img2, 0.2, gamma=0)
plt.imshow(blend_img2)
plt.show()

# 2) src2 이미지 더 진하게 = beta값 크게
blend_img3 = cv2.addWeighted(img1, 0.2, img2, 0.8, gamma=0)
plt.imshow(blend_img3)
plt.show()
