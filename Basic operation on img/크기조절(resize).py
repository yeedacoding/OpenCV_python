import cv2
import matplotlib.pyplot as plt
img = cv2.imread('images/read_color.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

new_img = cv2.resize(img, (1000, 400))
print(img.shape)
print(new_img.shape)
# cv2.resize(src, dstSize, fx, fy, interpolation)
# src = 입력 이미지, dstSize = 절대크기, fx, fy = 상대크기, interpolation = 보간법
# dstSize -> 튜플형식으로 전달, (너비, 높이)

# 수치 척도로 보기 위해 matplotlib 사용
plt.imshow(img)
plt.show()
plt.imshow(new_img)
plt.show()


# 비율로 조절하기
w_ratio = 0.8  # width * 0.8
h_ratio = 0.2  # height * 0.2

ratio_img = cv2.resize(img, (0, 0), img, fx=w_ratio, fy=h_ratio)
# fx, fy 생략하고 (~, w_ratio, h_ratio) 가능
plt.imshow(ratio_img)
plt.show()
