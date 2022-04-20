# Feature Matching -> FLANN Matcher
# FLANN(Fast Library for Approximate Nearest Neighbors Matching)
# BFMatcher는 모든 디스크립터를 전수 조사하므로 이미지 사이즈가 클 경우 속도가 느려짐
# 이를 해결하기 위해 FLANN을 사용
# FLANN은 모든 디스크립터를 전수조사하기 보다는 이웃하는 디스크립터끼리 비교
##########################################################################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('images/book1.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread('images/book2.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

##########################################################################################################
# matches = cv2.FlannBasedMatcher(indexParams, searchParams)

# indexParams : 인덱스 파라미터 (딕셔너리)
# algorithm: 알고리즘 선택 키, 선택할 알고리즘에 따라 종속 키를 결정하면 됨
# FLANN_INDEX_LINEAR=0: 선형 인덱싱, BFMatcher와 동일
# FLANN_INDEX_KDTREE=1: KD-트리 인덱싱 (trees=4: 트리 개수(16을 권장))
# FLANN_INDEX_KMEANS=2: K-평균 트리 인덱싱 (branching=32: 트리 분기 개수, iterations=11: 반복 횟수, centers_init=0: 초기 중심점 방식)
# FLANN_INDEX_COMPOSITE=3: KD-트리, K-평균 혼합 인덱싱 (trees=4: 트리 개수, branching=32: 트리 분기 새수, iterations=11: 반복 횟수, centers_init=0: 초기 중심점 방식)
# FLANN_INDEX_LSH=6: LSH 인덱싱 (table_number: 해시 테이블 수, key_size: 키 비트 크기, multi_probe_level: 인접 버킷 검색)
# FLANN_INDEX_AUTOTUNED=255: 자동 인덱스 (target_precision=0.9: 검색 백분율, build_weight=0.01: 속도 우선순위, memory_weight=0.0: 메모리 우선순위, sample_fraction=0.1: 샘플 비율)
# searchParams: 검색 파라미터 (딕셔너리)

# searchParams: 검색 파라미터 (딕셔너리)
# checks=32: 검색할 후보 수
# eps=0.0: 사용 안 함
# sorted=True: 정렬해서 반환

# 전달할 파라미터 값이 너무 많기 때문에 아래와 같은 방법으로 설정

# SIFT나 SURF를 사용하는 경우
FLANN_INDEX_KDTREE = 1
index_params_sift = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

# ORB를 사용하는 경우
FLANN_INDEX_LSH = 5
index_params_orb = dict(algorithm=FLANN_INDEX_LSH, table_number=6,
                        key_size=12, multi_probe_level=1)

##########################################################################################################
# 1. SIFT 활용한 FLANN Matcher매칭

# 1) 특징 검출 객체 생성
sift = cv2.xfeatures2d.SIFT_create()

# 2) keypoint, descriptor 검출
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 3) 인덱스 파라미터와 검색 파라미터 설정
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# 4) FLANN Matcher 객체 생성
matcher = cv2.FlannBasedMatcher(index_params, search_params)

# 5) 매칭 계산
matches = matcher.match(des1, des2)

# 6) 매칭 결과 그리기
res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                      flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

plt.title('FLANN Matcher with SIFT')
plt.imshow(res)
plt.show()

##########################################################################################################
# 2. ORB 활용한 FLANN Matcher 매칭

# 1) 특징 검출 개체 생성
orb = cv2.ORB_create()

# 2) keypoint, descriptor 검출
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 3) 인덱스 파라미터와 검색 파라미터 설정
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6,
                    key_size=12, multi_probe_level=1)
search_params = dict(checks=32)

# 4) FLANN Matcher 객체 생성
matcher = cv2.FlannBasedMatcher(index_params, search_params)

# 5) 매칭 계산
matches = matcher.match(des1, des2)

# 6) 매칭 결과 그리기
res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                      flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

plt.title('FLANN Matcher with ORB')
plt.imshow(res)
plt.show()
