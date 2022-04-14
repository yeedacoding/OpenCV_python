# Histogram
# 수 분포표를 그래프로 나타낸 것,  무엇이 몇 개 있는지 개수를 세어 놓은 것을 그래프로 나타낸 것
# 이미지 히스토그램의 경우 해당 픽셀값을 가지는 픽셀의 개수가 몇 개인지를 표현한 그래프
# 가로축 : 이미지 픽셀값 / 세로축 : 해당 픽셀값을 가지는 픽셀의 수
# 대비, 밝기, 색감 분포 등 이미지의 특성 분석에 용이
# 가로축은 보통 0-~255의 범위지만 0~10, 11~20, 21~30 같이 특정 간격의 범위에 있는 픽셀 수를 보고자 할 때는 BIN을 조정
# ex) BIN=16 -> 0~255 범위를 16개로 구분지음

import cv2
import numpy as np
import matplotlib.pyplot as plt

land_img = cv2.imread('images/landscape.jpg')   # ORIGINAL BGR
show_land = cv2.cvtColor(land_img, cv2.COLOR_BGR2RGB)
gray_land = cv2.imread('images/landscape.jpg', 0)
# CONVERTED TO RGB TO SHOW

strawberry = cv2.imread('images/strawberry.jpg')
show_straw = cv2.cvtColor(strawberry, cv2.COLOR_BGR2RGB)

############################################################################################################################
# cv2.calHist(img, channel, mask, histSize, ranges)
# img: 이미지 영상, [img]처럼 리스트로 감싸서 전달
# channel: 분석 처리할 채널, 리스트로 감싸서 전달, grayscale 이미지 : [0] / color 이미지 B, G, R 순서에 맞게 각각 [0], [1], [2] 전달
# mask: 마스크에 지정한 픽셀만 히스토그램 계산, None이면 전체 영역
# histSize: 계급(Bin)의 개수, 채널 개수에 맞게 리스트로 표현
# ranges: 각 픽셀이 가질 수 있는 값의 범위, RGB인 경우 보통 [0, 256] (0부터 255까지 양의 정수 값을 가지므로)


############################################################################################################################
# 0. grayscale 이미지의 히스토그램 값 확인하기
hist_val = cv2.calcHist([gray_land], channels=[0],
                        mask=None, histSize=[256], ranges=[0, 256])

plt.title('grayscale image histogram')
plt.plot(hist_val)
plt.show()

# RESULT = grayscale 이미지의 경우 0에서 255로 픽셀값이 증가할수록 0에 가까운, 즉 밝기가 증가하게 됨
# 해당 이미지는 40~50값을 가지는 픽셀의 개수가 가장 많은 것을 알 수 있음


############################################################################################################################
# 1. blue가 많은 이미지의 blue color의 히스토그램 값 확인하기
hist_val = cv2.calcHist([land_img], channels=[0],
                        mask=None, histSize=[256], ranges=[0, 256])


plt.title('much blue image')
plt.plot(hist_val)
plt.show()

# RESULT = blue 색상의 픽셀값이 많으므로 255값을 갖는 픽셀의 개수가 많은 그래프 = (b,g,r)에서 b의 값이 255에 가까운 픽셀의 수가 많다는 뜻


############################################################################################################################
# 2. blue가 거의 없는 이미지의 blue color의 히스토그램 값 확인하기
hist_val = cv2.calcHist([strawberry], channels=[0],
                        mask=None, histSize=[256], ranges=[0, 256])

plt.title('little blue')
plt.plot(hist_val)
plt.show()

# RESULT = blue 색상의 픽셀값이 거의 없으므로 0값을 갖는 픽셀의 개수가 많은 그래프 = (b,g,r)에서 b의 값이 0에 가까운 픽셀의 수가 많다는 뜻


############################################################################################################################
# 3. b,g,r의 히스토그램 값을 한 번에 표현한 히스토그램 그래프 표현하기(b,g,r 색상에 맞게 그래프 색상 지정)
color = ('b', 'g', 'r')

# 각 컬러를 cv2.calcHist() 메서드를 거쳐 plot 하기
for i, col in enumerate(color):
    histr = cv2.calcHist([land_img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.title('HISTOGRAM FOR LANDSCAPE IMAGE')
plt.show()
