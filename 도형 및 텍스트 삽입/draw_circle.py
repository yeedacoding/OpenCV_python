import numpy as np
import cv2

orange, blue, cyan = (0, 165, 255), (255, 0, 0), (255, 255, 0)
white, black = (255, 255, 255), (0, 0, 0)
# 이미지 행렬 생성한 다음에 fill로 색깔 주는 방법 말고 full로 한번에 줄 수 있었군..
image = np.full((300, 500, 3), white, np.uint8)

center = (image.shape[1]//2, image.shape[0]//2)     # 이미지의 중심
pt1, pt2 = (300, 50), (100, 220)
shade = (pt2[0] + 2, pt2[1] + 2)

# draw circle
cv2.circle(image, center, 100, blue)        # (image, 원 중심 좌표, 반지름, 색상, 두께)
cv2.circle(image, pt1, 50, orange, 2)
cv2.circle(image, pt2, 70, cyan, -1)        # 두께 -1 = 내부를 지정된 색상으로 채움

# 좌표확인을 위해 텍스트 넣기
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(image, 'center_blue', center, font, 1.0, blue)
cv2.putText(image, 'pt1_orange', pt1, font, 0.8, orange)
cv2.putText(image, 'pt2_cyan', shade, font, 1.2, black, 2)  # 그림자효과
cv2.putText(image, 'pt2_cyan', pt2, font, 1.2, cyan, 1)

cv2.imshow("Draw circles", image)
cv2.waitKey(0)
