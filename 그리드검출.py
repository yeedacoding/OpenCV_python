# Grid Detection
# camera calibration을 배우기 전의 간단한 사전 공부
# 카메라로 촬영을 하면 원래 직선인 물체가 왜곡되어 보이는 경우가 있는데 이를 보정하기 위한 작업이 필요함
# 왜곡된 grid나 점의 위치를 찾기 위한 opencv 메서드 두 가지


# 1. retval(found), corners = cv2.findChessboardCorners(image, patternSize[, corners[, flags]])
# image = source image
# patternSize = 체스보드의 (row, column)
# retval = 체스보드의 grid의 corner를 잘 검출했으면 True, 실패하면 False
# corners = 코너의 좌표값이 array로 return
# 만약 ret값이 False이면 체스 보드 이미지의 patterSize가 체스보드 이미지 상 (row, column) 값과 개수가 동일한지, (row,column)이 깔끔하게 나눠진 체스보드 이미지인지를 확인해야함

# findCirclesGrid(image, patternSize[, centers[, flags[, blobDetector]]]) -> retval, centers
# 원 그리드에서 원의 중심을 찾음
# patternSize = 원의 (row, column) 수
# flags methods
# CALIB_CB_SYMMETRIC_GRID - 대칭적인 원 패턴을 사용
# CALIB_CB_ASYMMETRIC_GRID - 비대칭적인 원 패턴을 사용
# CALIB_CB_CLUSTERING - uses a special algorithm for grid detection, (원근법 왜곡에는 더 강하지만 배경 잡음에는 훨씬 더 민감)

# 3. cv2.drawChessboardCorners(image, patternSize, corners, patternWasFound)
# findChessboardCorner나 findCirclesGrid에서 return받은 array를 corners에 넣고 ret(True)를 patternWasFound에 전달
# 체스보드 grid를 찾은 경우 선으로 연결된 컬러 모서리로 감지 된 개별 체스보드 모서리를 그려줌
# 원을 찾은 경우 선으로 연결된 컬러로 감지된 개별 원을 이어서 그려줌

import cv2
import matplotlib.pyplot as plt
import numpy as np

chess = cv2.imread('images/chess.jpg')

found, corners = cv2.findChessboardCorners(chess, (7, 7))

print(found)        # True
print(corners)      # 코너가 검출된 x,y 좌표값(array)

# 원본 chess 이미지에서 검출된 체스보드 grid의 corner값을 전달하여 grid모서리 그려주기
cv2.drawChessboardCorners(chess, (7, 7), corners, found)
plt.imshow(chess)
plt.show()


dots = cv2.imread('images/dots.jpg')

# 이미지 상 원의 row, column 수는 12, 8이기 때문에 (12,8) 전달
found, corners = cv2.findCirclesGrid(
    dots, (12, 8), cv2.CALIB_CB_SYMMETRIC_GRID)

cv2.drawChessboardCorners(dots, (12, 8), corners, found)

plt.imshow(dots)
plt.show()
