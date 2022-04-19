# SIFT, SURF, ORB
# Feature Matching 이전의 사전 공부 3단계
# feature descriptor를 구해주는 알고리즘

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/read_color.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

##########################################################################################################
# 1. SIFT(Scale-Invariant Feature Transform)
# 기존의 해리스 코너 검출 알고리즘은 크기 변화에 문제를 가지고 있음
# SIFT는 이미지 피라미드를 이용해서 크기 변화에 따른 특징점 검출 문제를 해결한 알고리즘

# 특징점 검출 객체 생성
# detector = cv2.xfeatures2d.SIFT_create(nfeatures, nOctabeLayers, contrastThreshold, edgeThreshold, sigma)
# nfeatures = 검출 최대 특징 수
# nOctaveLayers = 이미지 피라미드에 사용할 계층 수
# contrastThreshold = 필터링할 빈약한 특징 문턱 값
# edgeThreshold = 필터링할 엣지 문턱 값
# sigma = 이미지 피라미드 O 계층에서 사용할 가우시안 필터의 시그마 값

sift = cv2.xfeatures2d.SIFT_create()

# keypoint 검출 + descriptor 계산
kp1, des1 = sift.detectAndCompute(img, None)

# keypoint 그리기
draw1 = cv2.drawKeypoints(
    img, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.title('SIFT')
plt.imshow(draw1)
plt.show()

##########################################################################################################
# SURF(Speeded Up Robust Features)
# SIFT는 이미지 피라미드를 통해 이미지 크기 변화에 따른 특징 검출 문제를 해결하기 때문에 속도가 느림
# SURF는 이미지 피라미드 대신 필터의 크기를 변화시키는 방식을 쓰기 때문에 SIFT알고리즘의 속도 문제를 해결

# detect = cv2.xfeatures2d.SURF_create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright)
# hessianThreshold = 특징 추출 경계 값(default = 100)
# nOctaves = 이미지 피라미드 계층 수 (default = 3)
# nOctaveLayers = 디스크립터 생성 플래그 (default = False), True -> 128개, False -> 64개
# upright = 방향 계산 플래그 (default = False), True-> 방향 무시, False -> 방향 적용

# method
# surf = cv2.xfeatures2d.SURF_create()

# kp2, des2 = surf.detectAndCompute(img, None)

# draw2 = cv2.drawKeypoints(
#     img, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# plt.title('SURF')
# plt.imshow(draw2)
# plt.show()

# Error

print(cv2.__version__)  # 4.5.3

# SIFT를 실행시켰을 때 WARN 메세지가 출력되고 SURF를 실행시켰을 때는 아예 오류가 나면서 실행되지 않음
# 찾아보니 OpenCV 버전 3.4.2.16부터 지원하지 않는다고 함

##########################################################################################################
# 3. ORB (Oriented and Rotated BRIEF)
# 디스크립터 검출기 중 BRIEF라는 것이 있는데 Keypoint 검출은 지원하지 않는 디스크립터 검출기임
# 이 BRIEF에 방향과 회전을 고려하도록 개선한 알고리즘이 ORB

# detector = cv2.ORB_create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold)
# nfeatures(optional) = 검출할 최대 특징 수 (default=500)
# scaleFactor(optional) = 이미지 피라미드 비율 (default=1.2)
# nlevels(optional): =이미지 피라미드 계층 수 (default=8)
# edgeThreshold(optional) = 검색에서 제외할 테두리 크기, patchSize와 맞출 것 (default=31)
# firstLevel(optional) = 최초 이미지 피라미드 계층 단계 (default=0)
# WTA_K(optional) = 임의 좌표 생성 수 (default=2)
# scoreType(optional) = 특징점 검출에 사용할 방식 (cv2.ORB_HARRIS_SCORE: 해리스 코너 검출(default), cv2.ORB_FAST_SCORE: FAST 코너 검출)
# patchSize(optional) = 디스크립터의 패치 크기 (default=31)
# fastThreshold(optional) = FAST에 사용할 임계 값 (default=20)

orb = cv2.ORB_create()

kp3, des3 = orb.detectAndCompute(img, None)

draw3 = cv2.drawKeypoints(
    img, kp3, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.title('ORB')
plt.imshow(draw3)
plt.show()
