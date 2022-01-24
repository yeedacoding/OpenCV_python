# addWeighted()메서드는 두 이미지의 크기가 같을 때만 가능

import cv2
import matplotlib.pyplot as plt

# 2. 큰 이미지 위에 작은 이미지를 "overlay" (no blending)
# numpy 재배정 = 큰 이미지의 특정 배열을 작은 이미지 배열로 교체한다 = blending(합치기)가 아닌 교체의 느낌
img1 = cv2.imread('images/read_color.jpg')
large_img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('images/nocopy.png')
small_img = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

print(large_img.shape)
print(img2.shape)
# img2의 크기를 작게 만들어보자
small_img = cv2.resize(small_img, (300, 300))

# 오버레이되는 이미지가 시작되는 점을 변수로서 지정
x_offset = 0
y_offset = 0
# (x,y) = (0,0) -> large_img 상에서 (0,0), 즉 좌측 상단에서 small_img의 오버레이가 시작된다

# 오버레이가 끝나는 지점
x_end = x_offset + small_img.shape[1]  # offset에 작은 이미지의 x 길이를 더한 값
y_end = y_offset + small_img.shape[0]

# large_img의 y_offset~y_end(0, 300)와 x_offset~x_end(0,300)은 small_img로 overlay
# 특정 부분만 numpy 재배열 해주면 된다
large_img[y_offset:y_end, x_offset:x_end] = small_img

plt.imshow(large_img)
plt.show()
