import cv2

img = cv2.imread('images/read_color.jpg')
xy_flip_img = cv2.flip(img, -1)
x_flip_img = cv2.flip(img, 0)
y_flip_img = cv2.flip(img, 1)
# cv2.flip(src, flipCode)
# src = 원본 이미지, flipCode = 대칭 축
# flipCode < 0 : xy축대칭(상하좌우)
# flipCode = 0 : x축 대칭(상하)
# flipCode > 0 : y축 대칭(좌우)


cv2.imshow('img', img)
cv2.imshow('xy_flip', xy_flip_img)
cv2.imshow('x_flip', x_flip_img)
cv2.imshow('y_flip', y_flip_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
