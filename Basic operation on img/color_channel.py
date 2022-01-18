import numpy as np
import cv2

img = cv2.imread('images/colorchannel.jpg')
print(img.shape)
# (427, 640, 3) -> 3개의 채널(b,g,r)로 구성된 427x640 픽셀 이미지

# color channel로 이미지 분리
# b,g,r 순서이기 때문에 [:,:,0] -> 전체 행렬 중 첫번째 인덱스 즉 blue
b = img[:, :, 0]
# green
g = img[:, :, 1]
# red
r = img[:, :, 2]

while True:
    cv2.imshow('img', img)
    cv2.imshow('blue', b)
    cv2.imshow('green', g)
    cv2.imshow('red', r)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    break

# 분리한 이미지가 gray scale로 보이는 이유
print(b.shape)
# 원본 이미지는 3채널을 가졌지만 분리된 이미지는 채널이 1개만 존재하기 때문

# 그레이스케일이기 때문에 분리된 이미지는 0~255의 값을 가짐
# 각 채널에서 각 b,g,r에 해당되는 색을 가진 부분이 밝게 보인다
# 해당 채널에서 높은 값(255에 가까울수록 밝아짐)을 갖기 때문

#######################################################################
# 채널을 분리하는 다른 방법 -> split
# img_b, img_g, img_r = cv2.split(img)
# cv2.imshow('img_b', img_b)
# cv2.imshow('img_g', img_g)
# cv2.imshow('img_R', img_r)
#######################################################################

# 그레이스케일말고 색깔로 표현 -> merge
zeros = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
img_b = cv2.merge([b, zeros, zeros])
img_g = cv2.merge([zeros, g, zeros])
img_r = cv2.merge([zeros, zeros, r])


# 마찬가지로 각 색깔 채널에 해당되는 색을 가진 부분이 밝게 보임
while True:
    cv2.imshow('img', img)
    cv2.imshow('img_b', img_b)
    cv2.imshow('img_g', img_g)
    cv2.imshow('img_r', img_r)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    break
