# bitwise 연산
# 두 이미지를 합성할 때 특정 영역만 선택하거나 특정 영역만 제외하는 등의 선별적인 연산에 도움
# 연산을 할 때 두 이미지는 동일한 shape을 가져야함
# mask는 0이 아닌 픽셀만 연산하게 됨

# cv2.bitwise_and(img1, img2, mask=None): 각 픽셀에 대해 AND 연산
# img1, img2 모두 흰색(1)인 부분만 흰색으로 나타남

# cv2.bitwise_or(img1, img2, mask=None): 각 픽셀에 대해 OR 연산
# img1, img2 모두 검은색(0)인 부분만 검은색으로 나타남

# cv2.bitwise_not(img1, img2, mask=None): 각 픽셀에 대해 NOT 연산
# img2 이미지에서 색 반전

# cv2.bitwise_xor(img1, img2, mask=None): 각 픽셀에 대해 XOR 연산
# img1, img2 값이 서로 같으면 검은색, 같지 않으면 반전

import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = np.zeros((200, 400), np.uint8)
img2 = np.zeros((200, 400), np.uint8)

img1[:, :200] = 255         # 왼쪽은 흰색, 오른쪽은 검정색
img2[100:200, :] = 255      # 위쪽은 검정색, 아래쪽은 흰색

plt.title('img1')
plt.imshow(img1, 'gray')
plt.show()
plt.title('img2')
plt.imshow(img2, 'gray')
plt.show()

bitAnd = cv2.bitwise_and(img1, img2)
bitOr = cv2.bitwise_or(img1, img2)
bitNot = cv2.bitwise_not(img1)
bitXor = cv2.bitwise_xor(img1, img2)

# 0 = 검은색, 255 = 흰색
# boolean 차원에서 봤을 때 0 = False, 0이 아닌 값 = True
# 따라서 검은색은 False, 흰색은 True
# 기본 and, or 연산 생각하면 쉬움

imgs = {'bitwise_and': bitAnd, 'bitwise_or': bitOr,
        'bitwise_not': bitNot, 'bitwise_xor': bitXor}

for i, (title, img) in enumerate(imgs.items()):
    plt.subplot(3, 2, i+1)
    plt.title(title)
    plt.imshow(img, 'gray')
    plt.xticks([])
    plt.yticks([])

plt.show()
