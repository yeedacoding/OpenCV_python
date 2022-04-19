# Keypoints Detection
# Feature Matching 이전의 사전 공부 1단계

# 이전 corner 검출에서 사용되었던 Harris, GFTT 코너 검출 방법은 이미지의 크기 변환에 취약함
# 이미지가 확대 후에도 같은 크기의 window를 사용한다면, 확대 전 이미지에서 corner라고 인식되었던 부분이 확대 후에는 edge로서 인식될 수 있음
# 이를 해결하기 위해 확대된 이미지를 줄이거나 window크기를 키우거나 해야함
# 즉, 코너를 찾을 때 오리지널 사이즈만 보는 것이 아니라 다양한 크기도 고려를 해서 keypoint(feature point)를 찾는 것이 핵심
# feature point, keypoint : 이미지를 분석함에 있어 이미지 상 고유한 특징을 나타내는 점(좌표)들의 집합
# descriptor(기술자) : 특징점 주변의 부분 영상을 잘라서 특징점에 대한 특징을 기술하는 방법 -> 두 개의 영상이 같은지 판별하여 두 이미지를 매칭할때 사용됨
# descriptor는 keypoint의 주변의 밝기나 색상 등의 정보를 표현한 것으로, keypoint의 주변 픽셀을 일정한 크기로 나눈 후 각 영역에 대한 pixel의 그래디언트를 계산한 것

# OpenCV에서 제공하는 특징점 검출 클래스, cv2.Feature2D 클래스에서는 feature를 검출하는 여러 다른 알고리즘을 사용하는 자식 클래스들이 존재하고
# detect(), compute(), detectAndCompute() 와 같은 특징점 검출 함수를 통해 특징점(keypoint)를 검출할 수 있음


import cv2
import numpy as np

image = cv2.imread('images/read_color.jpg')

# Keypoint 검출 방법

##########################################################################################################
# 1. 특징점 검출 객체 생성 : cv2.방법_create()
cv2.KAZE_create()               # -> retval
cv2.ORB_create()                # -> retval
cv2.xfeatures2d.SIFT_create()   # -> retval

# retval : 각 특징점 검출 알고리즘 객체
# 각 알고리즘은 고유한 파라미터를 인자로 받을 수 있으며, 대부분의 인자는 default값을 가지고 있어 파라미터 값으로서 전달하지 않아도 호출 가능

##########################################################################################################
# 이렇게 생성된 객체에 특징점 검출 함수를 사용할 수 있음
# ex )
orb = cv2.ORB_create()  # 특징점 검출 객체 생성

# 특징점 검출 함수 사용해보기
# 1) cv2.Feature2D(객체).detect(image, mask=None) -> keypoints
keypoints = orb.detect(image, mask=None)

# image = keypoint를 검출할 입력 영상
# mask = mask 영상
# keypoints = 검출된 특징점 정보(리스트), cv2.KeyPoint 객체의 리스트
# 리턴값(keypoints)는 keypoint와 keypoint 객체의 리스트로 반환

# keypoint 값 확인해보기
print(keypoints)
print(len(keypoints))
# keypoint 클래스에 포함된 정보 중 주로 확인해야 할 것 : pt, size, angle
# pt : x,y좌표(float type)
# size : 특징점 검출할 때 어느 정도의 주변 크기를 가지고 특징점을 검출했는지에 대한 정보
# angle : 부분 영상의 주된 방향 변수

# 첫번째 keypoint의 정보 확인해보기
print(keypoints[0].pt)       # (233.0, 309.0) -> 검출된 Keypoint의 좌표
print(keypoints[0].size)     # 31.0 -> 첫번째 keypoint를 검출할 때 사용된 주변 픽셀까지의 크기
print(keypoints[0].angle)    # keypoint의 주된 방향 값

##########################################################################################################
# .detect() 함수로 얻은 keypoints 객체를 통해 descriptor 계산하기

# 2) cv2.Feature2D.compute(image, keypoints, descriptors=None) -> keypoints, descriptors
# image = keypoint를 검출할 입력 영상
# keypoints = .detect() 함수를 통해 얻은 keypoints 객체를 사용
# return 값
# keypoints = 검출된 특징점 정보, cv2.KeyPoint 객체의 리스트
# descriptors = 특징점 기술자 행렬

keypoints1, descriptors1 = orb.compute(image, keypoints)

# descriptor 정보 확인해보기
print('descriptors1 : ', descriptors1)
print('length of keypoints : ', len(keypoints1),
      'length of descriptors : ', len(descriptors1))
# 위의 출력 결과에서 알 수 있듯이 descriptor는 기본적으로 keypoint에 해당하는 정보이니 keypoint와 같은 개수로 생성되어짐

##########################################################################################################
# 이전에는 .detect()함수로 keypoints를 검출하고 이를 .compute() 함수의 파라미터로 전달하여 descriptor를 계산했었음
# 이번에는 keypoints 검출과 descriptor 계산을 한번에 해보자

# 3) cv2.Feature2D.detectAndCompute(image, mask=None, descriptors=None) -> keypoints, descriptors
keypoints2, descriptors2 = orb.detectAndCompute(image, None)

# descriptor 정보 확인해보기
print('descriptors2 : ', descriptors2)
print(np.array_equal(descriptors1, descriptors2))
# 위의 출력 결과에서 알 수 있듯이 descriptors1과 descriptors2는 같은 값을 가진다
