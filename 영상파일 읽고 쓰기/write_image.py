import cv2

image = cv2.imread("images/read_color.jpg", cv2.IMREAD_COLOR)
if image is None:
    raise Exception("영상파일 읽기 에러")


# 영상파일 저장을 위한 옵션을 지정
# 튜플 혹은 리스트로 지정
# jpeg 화질 설정 (0~100, 높은 값일수록 화질 좋음)
params_jpg1 = (cv2.IMWRITE_JPEG_QUALITY, 10)
params_jpg2 = (cv2.IMWRITE_JPEG_QUALITY, 80)
# png 압축레벨 설정(0~9, 높은 값일수록 용량 적고 압축 시간 길어짐)
params_png = [cv2.IMWRITE_PNG_COMPRESSION, 9]

# 행렬을 영상파일로 저장

# 그대로 저장(기본값은 95)
cv2.imwrite("images/write_test1.jpg", image)
# 지정한 화질로 저장
cv2.imwrite("images/write_test2.jpg", image, params_jpg1)
cv2.imwrite("images/write_test3.jpg", image, params_jpg2)
cv2.imwrite("images/write_test4.png", image, params_png)
cv2.imwrite("images/write_test5.bmp", image)
print("저장완료")
