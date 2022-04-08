# bitwise 연산으로 마스킹 적용해보기

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('images/read_color.jpg')
print(img.shape)

# 검은색 배경의 흰색 원 만들기
background = np.zeros_like(img)
print(background.shape)
center = (img.shape[1]//2, img.shape[0]//2)
white = (255, 255, 255)
circle = cv2.circle(background, center, 100, white, -1)

# circle 이미지에서 흰색 부분이 1, 배경은 0
# img 이미지는 모두 1
# 따라서 img와 circle을 bitwise_and 연산 해주면 둘 다 1인 부분, 즉 원 크기만큼 마스킹 됨
masked_img = cv2.bitwise_and(img, circle)

cv2.imshow('mask', masked_img)
cv2.waitKey()
cv2.destroyAllWindows()
