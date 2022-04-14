# Histogram Equalization
# 이미지의 contrast, 즉 명암 대비 개선을 위함
# 이미지의 히스토그램이 특정 영역에 너무 집중되어 있으면 contrast가 낮아 좋은 이미지가 아님
# 특정 영역에 집중되어 있는 분포를 골고루 분포하는 작업 = Histogram Equalization

import cv2
import numpy as np
import matplotlib.pyplot as plt

######################################################################################
# 1. 회색이미지
img = cv2.imread('images/landscape.jpg', 0)
eq_gray = cv2.equalizeHist(img)

merged = np.hstack((img, eq_gray))
plt.figure(figsize=(10, 8))
plt.title('original image, equalized image')
plt.imshow(merged, 'gray')
plt.show()

# 원본 grayscale 이미지의 히스토그램
hist_val = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.title('original grayscale image histogram')
plt.plot(hist_val)
plt.show()

# equalize한 이미지의 히스토그램
eq_hist_val = cv2.calcHist([eq_gray], [0], None, [256], [0, 256])
plt.title('equalized image histogram')
plt.plot(eq_hist_val)
plt.show()
# 원본 grayscale 이미지 히스토그램보다 완만한 변화를 보인다

######################################################################################
# 2. color 이미지
img = cv2.imread('images/landscape.jpg')
show_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# color 이미지의 contrast를 높이는 방법
# 1) color img -> HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 2) HSV(색조, 채도, 명도) 중 명도값을 equalize
hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

# 3) 명도가 equlize 되면 다시 HSV -> RGB (matplotlib으로 보여주기 위해 rgb)
eq_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

merged = np.hstack((show_img, eq_img))
plt.figure(figsize=(10, 8))
plt.title('original image, equalized image')
plt.imshow(merged)
plt.show()

######################################################################################
# CLAHE (Contrast Limited Adaptive Histogram Equalization)
# equalize를 하면 이미지의 밝은 부분이 너무 밝아져 날아가는 현상이 발생
# 이런 현상을 막기 위해 이미지를 일정한 영역으로 나누어 평탄화를 적용
# 이때 일정 영역 내에서 극단적으로 어둡거나 밝은 부분이 있으면 다른 영역에서의 평탄화 수준과 맞지 않아 노이즈 발생할 수 있음
# 이를 방지하기 위해 어떤 영역이든 지정된 제한 값을 넘으면 그 픽섹은 다른 영역에 균일하게 배분하여 적용
# 이러한 equalize 방식을 'CLAHE'

# clahe = cv2.createCLAHE(clipLimit, tileGridSize)
# clipLimit: 대비(Contrast) 제한 경계 값, default=40.0 (제한값을 넘으면 다른 영역에 균일하게 배분)
# tileGridSize: 영역 크기, default=8 x 8 (일정하게 나눌 영역)
# clahe: 생성된 CLAHE 객체

# clahe.apply(src): CLAHE 적용
# src: 입력 이미지

img = cv2.imread('images/clahe.jpg', 0)
show_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# equalizeHist 메서드로 equalize
eq_img = cv2.equalizeHist(img)
show_eq_img = cv2.cvtColor(eq_img, cv2.COLOR_BGR2RGB)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# clahe 변수에 지정한 값들을 이미지에 적용
clahe_img = clahe.apply(img)
show_clahe_img = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB)

merged = np.hstack((show_img, show_eq_img, show_clahe_img))
plt.figure(figsize=(12, 8))
plt.title('original image, equalizeHist image, CLAHE image')
plt.imshow(merged, 'gray')
plt.show()
# equalizeHist메서드가 적용된 eq_img에서 조각상의 얼굴 경계가 너무 밝아져 사라졌음
# 반면 clahe를 적용한 이미지에서는 경계도 유지되면서 전체적으로 contrast가 높아짐
