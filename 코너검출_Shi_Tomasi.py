# Corner Detection

# 2. Shi & Tomasi Detection(good features to track : GFTT)
# 해리스 코너 검출을 조금 더 개선한 알고리즘

# corners = cv2.goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance, corners, mask, blockSize, useHarrisDetector, k)
# img: 입력 이미지
# maxCorners: 얻고 싶은 코너의 개수, 강도가 강한 것 순으로 해당 개수만큼 검출됨
# qualityLevel: 코너로 판단할 스레시홀드 값(일반적으로 0.01~0.10사이 값 사용)
# 해리스 코너 검출 당시 최적화(threshoding)했던 부분을 파라미터로서 수행 가능
# ex) 가장 변화량이 큰 코너 intensity가 1000이고, 설정한 qualityLevel이 0.01이라면, 10이하의 코너 intensity를 갖는 코너들은 검출하지 않음

# minDistance: 검출된 코너 간 최소 거리(설정된 최소 거리 이상의 값만 검출)
# mask(optional): 검출에 제외할 마스크
# blockSize(optional)=3: 코너 주변 영역의 크기
# useHarrisDetector(optional)=False: 코너 검출 방법 선택 (True: 해리스 코너 검출 방법, False: 시-토마시 코너 검출 방법)
# k(optional): 해리스 코너 검출 방법에 사용할 k 계수
# corners: 코너 검출 좌표 결과, N x 1 x 2 크기의 배열, 실수 값이므로 정수로 변형 필요

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('images/house.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 시-토마시 코너 검출 적용
corner = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
# corner는 실수형 좌표
print(corner)

# 코너에 circle그리기 위해 실수좌표를 정수 좌표로 변환
# 몰랐는데 도형을 그릴 때 x,y좌표가 실수값이면 오류가 남
corner = np.int32(corner)
print(corner)

for i in corner:
    x, y = i[0]
    cv2.circle(img, (x, y), 10, (255, 0, 0), 3)

plt.imshow(img)
plt.show()
