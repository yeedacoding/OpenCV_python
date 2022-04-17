# Canny Edge
# 4단계의 알고리즘을 거쳐 경계를 검출
# 1. 노이즈 제거 (가우시안 블러링)
# 2. edge 그라디언트 크기, 방향 계산(sobel kernel convolution)
# 3. 비최대치 억제(Non-Maximum Suppression) : 그라디언트 방향에서 검출된 edge 중 가장 큰 값만 선택하고 edege가 될 수 없는 픽셀들 제거
# 4. 이력 스레스홀딩 : 두 개의 경계 값(max, min)을 지정해서 경계 영역에 있는 픽셀들 중 큰 경계값(max) 밖의 픽셀과 연결성 없는 픽셀 제거
# max보다 높은 부분 : strong edge (edge로 판정)
# min보다 낮은 부분 : not edge
# max ~ min 사이 : weak edge -> 주변에 strong edge로 판정된 edge가 있을 경우에만 edge로 판정(strong edge와 연결되어 있는 경우만)

# edges = cv2.Canny(img, threshold1, threshold2, edges, apertureSize, L2gardient)
# img: 입력 영상
# threshold1, threshold2: 이력 스레시홀딩에 사용할 Min, Max 값
# apertureSize: 소벨 마스크에 사용할 커널 크기(default = 3)
# L2gradient: 그레디언트 강도를 구할 방식 (True: 제곱 합의 루트 False: 절댓값의 합)
# edges: 엣지 결과 값을 갖는 2차원 배열

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('images/brick.jpg')

edge = cv2.Canny(img, threshold1=120, threshold2=200)
plt.imshow(edge)
plt.show()
# 이미지에 노이즈가 많아 많은 부분들이 edge로 검출되었음

##########################################################################################################
# threshold1, threshold2 초기 값을 정하는 괜찮은 방정식(정답은 아님)
# 1. 이미지의 중앙 픽셀 값 확인
med = np.median(img)
print(med)
# 2. 방정식 적용 -> lower : 최저값과 중간값의 백분율 / upper : 최고값과 중간값의 백분율
# 0 또는 중간값의 70%(중간값의 30% 이하) 중 큰 값을 threshold1으로 지정
lower = int(max(0, 0.7*med))
# 255 또는 중간값의 130%(중간값의 30# 이상) 중 작은 값을 threshold2로 지정
upper = int(min(255, 1.3*med))

print('lower : ', lower)
print('upper : ', upper)

edge = cv2.Canny(img, threshold1=lower, threshold2=upper+120)

plt.title('formula applied')
plt.imshow(edge)
plt.show()
# 아직도 노이즈 때문에 많은 부분들이 edge로 인식됨

##########################################################################################################
# 이 경우 수동적으로 블러링을 하여 노이즈를 제거 후 canny를 적용시킬 수 있음
blur = cv2.blur(img, (5, 5))

edge = cv2.Canny(blur, lower, upper)

plt.title('manually blur')
plt.imshow(edge)
plt.show()
# 확실히 사전에 이미지를 블러링하여 노이즈를 제거하니 edge 검출이 더욱 깔끔해졌다

##########################################################################################################
# 트랙바로 th1, th2 값을 조정해가며 비교해보기


def nothing():
    pass
# 콜백함수를 위한 빈 함수 정의


cv2.namedWindow("Canny Edge")
cv2.createTrackbar('lower threshold', 'Canny Edge', 0, 1000, nothing)
cv2.createTrackbar('higher threshold', 'Canny Edge', 0, 1000, nothing)

cv2.setTrackbarPos('lower threshold', 'Canny Edge', lower)
cv2.setTrackbarPos('higher threshold', 'Canny Edge', upper)


while True:
    low = cv2.getTrackbarPos('lower threshold', 'Canny Edge')
    high = cv2.getTrackbarPos('higher threshold', 'Canny Edge')

    canny_edge = cv2.Canny(img, low, high)
    cv2.imshow("Canny Edge", canny_edge)

    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()
