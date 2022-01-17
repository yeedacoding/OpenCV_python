# 회전 변환 행렬(Rotation matrix)를 통해 회전
# 회전 변환 행렬은 임의의 점을 기준으로 이미지를 회전시킴

import cv2

img = cv2.imread('images/read_color.jpg')

height, width, channel = img.shape

# 원본 이미지의 height, width를 이용하여 회전 중심점 설정
# 2x3 회전 행렬 생성 함수(cv2.getRotationMatrix2D)로 회전 변환 행렬을 계산
matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)
# cv2.getRotationMatrix2D(center, angle, scale) -> 매핑 변환 행렬 생성 및 반환
# center = 중심점(튜플)
# angle = 중심점 기준으로 회전할 각도
# scale = 이미지의 확대 및 축소 비율

# 아핀 변환 함수(cv2.warpAffine)로 회전 변환 계산
rotate_img = cv2.warpAffine(img, matrix, (width, height))
# cv2.warkAffine(src, M, dsize)
# src = 원본이미지
# M = 아핀 맵 행렬 -> 회전 행렬 생성 함수에서 반환된 매핑 변환 행렬
# dsize = 출력 이미지 크기 (튜플)


cv2.imshow('img', img)
cv2.imshow('rotate', rotate_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
