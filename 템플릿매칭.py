# Template Matching
# 원본 이미지에서 템플릿 이미지와 일치하는 영역을 찾는 방법
# 원본 이미지 위에 템플릿 이미지를 놓고 좌측 상단부터 조금씩 이동해가며 이미지 끝에 도달할 때까지 비교
# 이 과정에서 템플릿 이미지와 동일하거나 가장 유사한 영역을 원본 이미지에서 검출
# 템플릿 이미지는 원본 이미지보다 크기가 항상 작아야 함


import cv2
import matplotlib.pyplot as plt

super_mario = cv2.imread('images/super_mario.jpg')
super_mario = cv2.cvtColor(super_mario, cv2.COLOR_BGR2RGB)

mario = cv2.imread('images/mario.jpg')
mario = cv2.cvtColor(mario, cv2.COLOR_BGR2RGB)

mario_dst = cv2.imread('images/super_mario.jpg')
mario_dst = cv2.cvtColor(mario_dst, cv2.COLOR_BGR2RGB)
#########################################################################################################
# result = cv2.matchTemplate(img, templ, method, result, mask)

# img: 입력 이미지
# templ: 템플릿 이미지
# method: 매칭 메서드 (cv2.TM_SQDIFF: 제곱 차이 매칭, 완벽 매칭:0, 나쁜 매칭: 큰 값 / cv2.TM_SQDIFF_NORMED: 제곱 차이 매칭의 정규화 / cv2.TM_CCORR: 상관관계 매칭, 완벽 매칭: 큰 값, 나쁜 매칭: 0 / cv2.TM_CCORR_NORMED: 상관관계 매칭의 정규화 / cv2.TM_CCOEFF: 상관계수 매칭, 완벽 매칭:1, 나쁜 매칭: -1 / cv2.TM_CCOEFF_NORMED: 상관계수 매칭의 정규화)
# result(optional): 매칭 결과, (W - w + 1) x (H - h + 1) 크기의 2차원 배열 [여기서 W, H는 입력 이미지의 너비와 높이, w, h는 템플릿 이미지의 너비와 높이]
# mask(optional): TM_SQDIFF, TM_CCORR_NORMED인 경우 사용할 마스크

#########################################################################################################
# Template Matching에 사용되는 6가지 방법
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

# cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED -> 최소값이 나타난 곳(min_loc : 가장 어두운 곳)에서 작동
# 나머지는 최대값(max_loc :가장 밝은 곳)이 매칭 지점이 됨

#########################################################################################################

for m in methods:
    method = eval(m)

    # Template Matching
    result = cv2.matchTemplate(mario_dst, mario, method)

    # 히트맵(result)에서 최대값, 최소값, 최대값의 위치, 최소값의 위치 언패킹
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # 이미지에 그릴 rectangle의 top left(좌측 상단)의 위치 지정
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc  # (x, y)
    else:
        top_left = max_loc

    # rectangle의 bottom right 위치 지정
    h, w, channels = mario.shape
    # rectangle의 좌측 상단에서 width와 height을 더해 우측 하단(직사각형의 끝)지점을 지정
    bottom_right = (top_left[0]+w, top_left[1]+h)

    mario_dst = cv2.rectangle(
        mario_dst, top_left, bottom_right, (255, 0, 0), 5)

    # method의 히트맵과 rectangle이 그려진 result 이미지 plot하기
    plt.subplot(121)
    plt.imshow(result)
    plt.title('HEATMAP OF TEMPLATE MATCHING')
    plt.subplot(122)
    plt.imshow(mario_dst)
    plt.title('DETECTION OF TEMPLATE')

    plt.suptitle(m)

    plt.show()

    print('\n')
