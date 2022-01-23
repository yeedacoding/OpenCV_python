# RGB -> RED, GREEN, BLUE가 결합한 색
# HSL -> 색조, 채도, 밝기로 색을 표헌하는 방법
# HSV -> 색조, 채도, 명도로 색을 표현하는 방법
# 색조 : 0에서 360 사이의 색상환 각도입니다. 0은 빨간색, 120은 녹색, 240은 파란색
# 채도 : 백분율 값이고 0 %는 회색 음영을 의미하고 100 %는 풀 컬러 (색의 진함과 연함)
# 밝기 : 백분율이며 0 %는 검은 색, 50 %는 밝거나 어둡지 않으며 100 %는 흰색 (어둡고 밝은 정도)
# 명도 : 색상의 밝기 또는 강도를 0에서 100%까지 나타냄, 0은 완전히 검은 색이고 100은 가장 밝으며 가장 많은 색상을 나타냄

import cv2
import matplotlib.pyplot as plt

# img 불러와 matplotlib으로 띄우기
img = cv2.imread('images/read_color.jpg')
plt.imshow(img)
plt.show()

# cv2로 불러온 이미지 bgr -> rgb로 수정
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

# HSV로 전환
img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
plt.imshow(img)
plt.show()

# HLS로 전환
img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
plt.imshow(img)
plt.show()
