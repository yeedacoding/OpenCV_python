# Feature Matching(특징 매칭) -> BFMatcher, 특징 매칭기가 가지는 함수 알아보기, 매칭 결과 그리기
# Feature Matching이란 서로 다른 두 이미지에서 keypoint와 feature descriptor들을 비교해서 비슷한 객체끼리 짝짓는 것을 의미
# opencv에서 제공하는 특징 매칭기 : BFMatcher, FLannBasedMatcher

import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('images/book1.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread('images/book2.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

##########################################################################################################

# BFMatcher
# Brute-Force Matcher : queryDescriptors와 trainDescriptors를 하나하나 확인해(전수조사) 매칭되는지 판단하는 알고리즘
# 전수조사하기 때문에 이미지 사이즈가 클 경우 속도가 느리다는 단점
# matcher = cv2.BFMatcher_create(normType, crossCheck)
# normType : 거리측정 알고리즘 (cv2.NORM_L1, cv2.NORM_L2(default), cv2.NORM_L2SQR, cv2.NORM_HAMMING, cv2.NORM_HAMMING2)
# SIFT, SURF 디스크립터 검출기의 경우 NORM_L1, NORM_L2가 적합
# ORB 디스크립터 검출기의 경우 NORM_HAMMING이 적합
# crosscheck : 상호 매칭이 되는 것만 반영(default = False)
# crosscheck = True면 두 이미지의 양쪽 디스크립터 모두에게서 매칭이 완성된 것만 반영하여 불필요한 매칭을 줄일 수 있지만 그만큼 속도가 느려짐

###########################################################################################################################################
# 특징 매칭기는 두 개의 디스크립터를 서로 비교하여 매칭해주는 함수를 가짐
# match(), knnMatch(), radiusMatch()
# 세 함수 모두 첫 번째 파라미터인 queryDescriptors를 기준으로 두 번째 파라미터인 trainDescriptors에 맞는 매칭을 찾음

# 1) match()
# queryDescriptors 한 개당 최적의 매칭을 이루는 trainDescriptors를 찾아 결과로 반환
# 최적 매칭을 찾지 못하는 경우도 있기 때문에 반환되는 매칭 결과 개수가 queryDescriptors의 개수보다 적을 수도 있음
# matches = matcher.match(quertDescriptors, trainDescriptors, mask)
# queryDescriptors : 특징 디스크립터 배열(특징 디스크립터 검출기(SIFT, SURF, ORB를 통해 얻은 descriptors)), 매칭의 기준이 될 디스크립터
# trainDescriptors : 특징 디스크립터 배열, 매칭의 대상이 될 디스크립터
# 큰 이미지 내에 있는 사물 중 찾고자 하는 비슷한 이미지가 있다면 찾고자 하는 비슷한 이미지가 queryDescriptors, 비슷한 이미지를 포함하고 있는 큰 이미지가 trainDescriptors
# mask(optional) : 매칭 진행 여부 마스크
# matches :  매칭 결과, DMatch 객체의 리스트

# 2) knnMatch() -> k개의 가장 인접한 매칭
# queryDescriptors 한 개당 k개의 최근접 이웃 개수만큼 trainDescriptors에서 찾아 반환
# matches = matcher.knnMatch(queryDescriptors, trainDescriptors, k, mask, compactResult)
# k : 매칭할 근접 이웃 개수
# compactResult(optional) : True - 매칭이 없는 경우 매칭 결과에 불포함(default = False)
# compactResult=False면 매칭 결과를 찾지 못해도 결과에 queryDescriptors의 ID를 보관하는 행 추가 (True면 아무것도 추가 x)

# 3) radiusMatch() -> maxDistance 거리 이내의 매칭을 찾아줌
# matches = matcher.radiusMatch(queryDescriptors, trainDescriptors, maxDistance, mask, compactResult)
# maxDistance : 매칭 대상 거리

# 그렇다면 반환된 결과를 담은 객체 matches에는 어떤 정보가?
# DMatch : 매칭 결과를 표현하는 객체
# queryIdx : queryDescriptors의 인덱스
# trainIdx : trainDescriptors의 인덱스
# imgIdx : trainDesciptor의 이미지 인덱스
# distance(유사도) : 유사도 거리
# queryIdx, trainIdx를 통해 두 이미지의 어느 지점이 서로 매칭되었는지 알 수 있음
# distance로 얼마나 가까운 거리인지 알 수 있음(낮을 수록 더 좋은 매칭)

###########################################################################################################################################
# 매칭 결과를 시각적으로 표현하기 위한 메서드
# 두 이미지를 하나로 합쳐 매칭점끼리 선으로 연결해줌

# cv2.drawMatches(img1, kp1, img2, kp2, matches, outImg, flags)
# img1, kp1 : queryDescriptors의 이미지와 keypoint
# img2, kp2 : trainDescriptors의 이미지와 keypoint
# matches : 매칭기 결과 객체 전달
# outImg : 출력영상(None)
# flags : 매칭점 그리기 옵션
# cv2.DRAW_MATCHES_FLAGS_DEFAULT : 결과 이미지 새로 생성(default)
# cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG : 결과 이미지 새로 생성 안 함
# cv2.DRAW_MATCHES_FLAGS_RICH_KEYPOINTS : keypoint 크기와 방향 그리기
# cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS : 한쪽만 있는 매칭 결과 그리기 제외

###########################################################################################################################################
# 1. SIFT 활용한 BFMatcher로 매칭

# 1) 특징 검출 객체 생성
sift = cv2.xfeatures2d.SIFT_create()

# 2) keypoints, descriptors 검출
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 3) BFMatcher 객체 생성
matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

# 4) 매칭 계산
matches = matcher.match(des1, des2)

# 5) matches가 가지는 값 확인해보기
print(len(matches))
print(matches[:20])
print(matches[0].queryIdx)
print(matches[0].trainIdx)
# matches[0].trainIdx = matches[0]의 des와 매칭된 image2 des의 index에 해당
print(matches[0].distance)

# 6) 매칭 결과 그리기
res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                      flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

plt.title('BFMatcher with SIFT')
plt.imshow(res)
plt.show()

###########################################################################################################################################
# 2. ORB 활용한 BFMatcher로 매칭

# 1) 특징 검출 객체 생성
orb = cv2.ORB_create()

# 2) keypoint, descriptor 검출
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 3) BFMatcher 객체 생성
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 4) 매칭 계산
matches = matcher.match(des1, des2)

# 5) 매칭 결과 그리기
res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                      flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

plt.title('BFMatcher with ORB')
plt.imshow(res)
plt.show()