# 카메라 속성 (zoom, focus) 조절

import cv2
from Common.utils import put_string


def zoom_bar(value):
    global capture
    capture.set(cv2.CAP_PROP_ZOOM, value)


def focus_bar(value):
    global capture
    capture.set(cv2.CAP_PROP_FOCUS, value)


capture = cv2.VideoCapture(0)
if capture.isOpened() == False:
    raise Exception("카메라 연결 안됨")

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 400)  # 카메라 프레임 너비
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)  # 카메라 프레임 높이
capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)      # 자동초점 중지
capture.set(cv2.CAP_PROP_BRIGHTNESS, 100)   # 프레임 밝기 초기화

title = "Change Camera Properties"
cv2.namedWindow(title)
cv2.createTrackbar('zoom', title, 0, 10, zoom_bar)
cv2.createTrackbar('focus', title, 0, 40, focus_bar)

while True:
    ret, frame = capture.read()
    if not ret:
        break
    if cv2.waitKey(30) >= 0:
        break

    zoom = int(capture.get(cv2.CAP_PROP_ZOOM))
    focus = int(capture.get(cv2.CAP_PROP_FOCUS))
    put_string(frame, 'zoom : ', (10, 240), zoom)
    put_string(frame, 'focus : ', (10, 270), focus)
    cv2.imshow(title, frame)

capture.release()

# 줌 기능이 안되는 캠에서는 안되는 듯 하다...
