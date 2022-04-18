# contour
# 윤곽선 : 같은 색과 강도로 객체의 경계선을 따라 계속되는 점을 이은 선
# 윤곽선은 외형을 파악하고 물체를 감지하고 인식하는 데 유용

import cv2
import matplotlib.pyplot as plt
import numpy as np

# binary된 grayscale 이미지
img = cv2.imread('images/contour.png', 0)

print(img.shape)

plt.title('original')
plt.imshow(img, 'gray')
plt.show()

#######################################################################################################
# dst, contours, hierarchy = cv2.findContours(src, mode, method, contours, hierarchy, offset)
# src: 입력 영상, 검정과 흰색으로 구성된 바이너리 이미지(배경은 검은색, 물체는 흰색)
# mode: 컨투어 제공 방식
# cv2.RETR_EXTERNAL: 가장 바깥쪽 라인만 생성
# cv2.RETR_LIST: 모든 라인을 계층 없이 생성
# cv2.RET_CCOMP: 모든 라인을 2 계층으로 생성(External contour & Internal contour)
# cv2.RETR_TREE: 모든 라인의 모든 계층 정보를 트리 구조로 생성

# method: 근사 값 방식
# cv2.CHAIN_APPROX_NONE: 근사 없이 윤곽점들의 모든 좌표 제공
# cv2.CHAIN_APPROX_SIMPLE:  contours line을 그릴 수 있는 point 만 저장(ex :사각형이면 4개 point)
# cv2.CHAIN_APPROX_TC89_L1: Teh-Chin 알고리즘으로 좌표 개수 축소
# cv2.CHAIN_APPROX_TC89_KC0S: Teh-Chin 알고리즘으로 좌표 개수 축소

# contours(optional): 검출한 컨투어 좌표 (list type)
# hierarchy(optional): 컨투어 계층 정보 (Next, Prev, FirstChild, Parent, -1 [해당 없음])
# offset(optional): ROI 등으로 인해 이동한 컨투어 좌표의 오프셋


#######################################################################################################
# cv2.drawContours(img, contours, contourIdx, color, thickness)
# img: 입력 영상
# contours: 그림 그릴 컨투어 배열 (cv2.findContours() 함수의 반환 결과를 전달해주면 됨)
# contourIdx: 그림 그릴 컨투어 인덱스, -1: 모든 컨투어 표시(특정 컨투어만 그릴 경우 -1이 아니라 그릴 contour의 인덱스 값을 전달해야함)
# color: 색상 값
# thickness: 선 두께, 0: 채우기

# contours 배열에 있는 contour 중 contourIdx에 해당하는 contour를 전달
# 지정한 색깔과 두께로 선을 그려줌

#######################################################################################################
# 얼굴 안의 눈, 입이나 육각형 내의 작은 점들처럼 내부 윤곽뿐만 아니라 삼각형, 얼굴형, 육각형의 외부 윤곽도 감지할 수 있음

# 1. 외부 contour, 내부 contour모두 표시
contours, hierarchy = cv2.findContours(
    img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

# contours = 검출한 컨투어 좌표
print(type(contours))
print(len(contours))
# hierarchy = 컨투어 계층 정보
# 다음 윤곽선, 이전 윤곽선, 내곽 윤곽선, 외곽윤곽선
# ex)
# [2 -1 1 -1] (index = 0)
# 인덱스 0의 hierarchy 값이 위와 같다면,
# 2 : 인덱스 0 도형의 다음 도형 윤곽선 정보는 인덱스 2에 존재
# -1 : 이전 윤곽선은 없다
# 1 : 내곽(자식)윤곽선 정보는 인덱스 1에 있다
# -1 : 외곽(부모) 윤곽선은 없다 = 4번째 value값이 -1인 것들만 추출하면 외곽 윤곽선만 뽑아낼 수 있음
print(type(hierarchy))
print(hierarchy)


external_contours = np.zeros(img.shape)

print(external_contours.shape)

for i in range(len(contours)):
    # DRAW EXTERNAL CONTOUR
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(external_contours, contours, i, 255, -1)

plt.title('external contours')
plt.imshow(external_contours, 'gray')
plt.show()

internal_contours = np.zeros(img.shape)

for i in range(len(contours)):
    # DRAW INTERNAL CONTOUR
    # -1이면 외곽 윤곽선이기 때문에 그 반대를 구하면 내곽 윤곽선
    if hierarchy[0][i][3] != -1:
        cv2.drawContours(internal_contours, contours, i, 255, -1)

plt.title('internal contours')
plt.imshow(internal_contours, 'gray')
plt.show()


# hirearchy를 분석하면 특정 물체 안에 포함된 내곽 윤곽선들만 추출할 수 있음
some_contours = np.zeros(img.shape)

for i in range(len(contours)):
    # DRAW INTERNAL CONTOUR
    if hierarchy[0][i][3] == 0:
        cv2.drawContours(some_contours, contours, i, 255, -1)

# 결과를 보아하니 얼굴 도형의 눈과 입(내곽 윤곽선) 윤곽선 이었다
plt.title('eyes and mouth contours')
plt.imshow(some_contours, 'gray')
plt.show()
