# 이미지 피라미드(image pyramid)를 활용해 이미지의 크기를 샘플링
# 가우시안 피라미드(Gaussian Pyramid)와 라플라시안 피라미드(Laplacian Pyramid)를 활용

import cv2

img = cv2.imread('images/read_color.jpg')
height, width, channel = img.shape

up_img = cv2.pyrUp(img,
                   borderType=cv2.BORDER_DEFAULT)
# 이미지 확대 함수 (cv2.pyrUp)로 이미지를 2배 확대
# cv2.pyrUp(src, dstSize, borderType)
# dstSize = 출력 이미지 크기, 세밀한 크기 조정을 필요로 할 때 사용
# borderType = 테두리 외삽법, 이미지를 확대, 축소할 경우, 이미지 영역 밖의 픽셀은 추정해 값을 할당해야함
# 테두리 외삽법은 이미지 밖의 픽셀을 외삽하는 데 사용되는 테두리 모드로, 외삽 방식을 설정

down_img = cv2.pyrDown(img)
# 이미지 축소 함수 (cv2.pyrDown)로 이미지를 2배 축소

cv2.imshow('img', img)
cv2.imshow('up_img', up_img)
cv2.imshow('down_img', down_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
