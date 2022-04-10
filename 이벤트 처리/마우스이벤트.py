import numpy as np
import cv2


def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 마우스 왼쪽 클릭 시 원 그리기
        cv2.circle(image, (x, y), 100, (0, 255, 0), -1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        print("마우스 오른쪽 버튼 누르기")
    elif event == cv2.EVENT_RBUTTONUP:
        print("마우스 오른쪽 버튼 떼기")
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        print("마우스 왼쪽 버튼 더블클릭")


cv2.namedWindow(winname='Mouse Event')
cv2.setMouseCallback('Mouse Event', onMouse)


image = np.zeros((512, 512, 3))

while True:
    cv2.imshow('Mouse Event', image)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()
