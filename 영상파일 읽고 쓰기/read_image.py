# 영상 처리의 기본적인 작업 -> 영상파일을 처리하기 위해 데이터로 읽어 들이기

import cv2
import matplotlib.pyplot as plt
# 행렬의 정보를 알기 위해 정보를 출력해주는 함수


def print_matInfo(name, image):
    if image.dtype == 'uint8':
        mat_type = 'CV_8U'
    elif image.dtype == 'int8':
        mat_type = 'CV_8S'
    elif image.dtype == 'uint16':
        mat_type = 'CV_16U'
    elif image.dtype == 'int8':
        mat_type = 'CV_16S'
    elif image.dtype == 'float32':
        mat_type = 'CV_32F'
    elif image.dtype == 'float64':
        mat_type = 'CV_64F'

    nchannel = 3 if image.ndim == 3 else 1

    # depth, channel 출력
    print("%12s : depth(%s), chaanel(%s) -> mat_type(%sC%d)" %
          (name, image.dtype, nchannel, mat_type, nchannel))


title1, title2 = 'gray2gray', 'gray2color'

# cv2의 imread로 읽어온 이미지의 type은 numpy 배열
gray2gray = cv2.imread("images/read_gray.jpg")
gray2color = cv2.imread("images/read_color.jpg")

# cv로 불러온 임지 matplotlib으로 띄우기
plt.imshow(gray2color)
plt.show()
# 원본 이미지와 다름
# openCV랑 matplotlib에서 r,g,b 색을 불러오는 순서가 다르기 때문
# openCV -> BGR
# matplotlib -> RGB

# BGR로 순서 바꿔서 matplotlib으로 다시 띄워보기
mat_img = cv2.cvtColor(gray2color, cv2.COLOR_BGR2RGB)
plt.imshow(mat_img)
plt.show()


# 예외처리 -> 영상파일 읽기 여부 조사
if gray2gray is None or gray2color is None:
    raise Exception("영상파일 읽기 에러")

print("행렬 좌표 (100, 100) 화소값")
print("%s %s" % (title1, gray2gray[100, 100]))
print("%s %s\n" % (title2, gray2color[100, 100]))

print_matInfo(title1, gray2gray)
print_matInfo(title2, gray2color)

cv2.imshow(title1, gray2gray)
cv2.imshow(title2, gray2color)
cv2.waitKey(0)


# matplotlib으로 이미지 띄우기
