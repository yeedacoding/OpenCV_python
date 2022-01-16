import numpy as np
import cv2

img = cv2.imread('images/read_color.jpg')

# 행렬의 좌표값으로 해당 이미지의 픽셀 값에 엑세스
px = img[100, 100]
print(px)
# 반환 값은 BGR

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
px = gray_img[100, 100]
print(px)
# gray img일 경우에는 해당 픽셀의 강도(intensity) 반환

# blue 픽셀만 반환하기
# 컬러 이미지의 픽셀값은 b,g,r 순서이기 때문에 각 픽셀값의 첫번째 인덱스가 blue
# img[100,100,0] = img[100][100][0]
blue = img[100, 100, 0]
print(blue)

# 이미지 속성
# 컬러 이미지의 경우 행, 열, 채널 수
print(img.shape)
# gray 이미지의 경우 행, 열만 출력 됨
print(gray_img.shape)

# 총 픽셀 수
print(img.size)
# 이미지의 데이터 타입
print(img.dtype)
