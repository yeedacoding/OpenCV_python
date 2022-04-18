# contour
# 이번에는 color 이미지로 contour detection 해보기

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('images/objects.jpg')
img2 = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# threshold 이용하여 이미지 binary화
# BINARY_INV를 통해 배경은 검은색, 전경은 흰색으로 thresholding
ret, thres = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

cv2.imshow('original', thres)
cv2.waitKey(0)

# 다른 findContours 파라미터 사용해보기
# 1. 외곽contour에 대해 모든 좌표 반환
contour1, hierarchy1 = cv2.findContours(
    thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cv2.drawContours(img, contour1, -1, (0, 255, 0), 4)

cv2.imshow('CHAIN_APPROX_NONE : ', img)
cv2.waitKey(0)

# 2. 외곽contour의 꼭지점 좌표만 반환
contour2, hierarchy2 = cv2.findContours(
    thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i in contour2:
    for j in i:
        cv2.circle(img2, tuple(j[0]), 5, (0, 0, 255), -1)

# CHAIN_APPROX_SIMPLE ->  contours line을 그릴 수 있는 point 만 저장. (ex; 사각형이면 4개 point)
# 결과에서 볼 수 있듯이 사각형, 마름모 모양의 경우 각 꼭짓점에 원이 그려졌지만
# 원 같이 곡선의 경우에는 곡선을 그리기 위해 필요한 모든 지점들에 원이 그려졌다
cv2.imshow('CHAIN_APPROX_SIMPLE : ', img2)
cv2.waitKey(0)

cv2.destroyAllWindows()
