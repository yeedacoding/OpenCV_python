# Feature Matching - 더 좋은 매칭점 찾기 ratio test 적용
# 쓸데없는 매칭점은 버리고 올바른 매칭점만 골라내기

import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('images/book1.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread('images/book2.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = matcher.match(des1, des2)

# 매칭 결과를 distance 기준으로 오름차순 정렬
matches = sorted(matches, key=lambda x: x.distance)

# 최적의 매칭점 25개만 그리기
res = cv2.drawMatches(img1, kp1, img2, kp2,
                      matches[:25], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

plt.imshow(res)
plt.show()

# 아직도 결과가 썩 좋지 않다...

##########################################################################################################
# ratio test, knnMatch 적용

# knnMatch() : 디스크립터 당 k개의 최근접 이웃 매칭점을 가까운 순서대로 반환
# k개의 최근접 이웃 중 거리가 가까운 것은 좋은 매칭점, 먼 것은 좋지 않은 매칭점
# 최근접 이웃 중 거리가 가까운 것 위주로 골라내기

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

matcher = cv2.BFMatcher(cv2.NORM_L1)

# knnMatch() 사용하여 매칭
matches = matcher.knnMatch(des1, des2, k=2)

print(matches[:25])
# 매칭값의 결과(DMatch)값을 확인해보면 각 리스트 안에 DMatch 객체가 두 개씩 쌍을 이뤄 결과를 반환한다
# 첫 번째 매칭 결과(matches[0])은 첫번째 queryDescriptor와 가장 잘 매칭이 되는 매칭 결과 2가지가 반환되는 것이고
# 두 번째, 세 번재 ... 마찬가지
# k=3 라고 지정하면 한 리스트 안에 매칭이 가장 잘 된 매칭점 3가지가 좋은 순으로 반환된다
# 따라서 matches[0]의 결과 리스트 안의 두 매칭은 가장 좋은 매칭점과 그 다음 좋은 매칭점인 것이다
# 그렇기에 한 리스트 안에 첫 번째 매칭 결과와 두 번째 매칭 결과의 distance값이 차이가 많이 난다면 별로 좋지 않은 매칭점이 되므로
# 첫 번쨰, 두 번째 매칭 결과의 distance값을 활용하여 ratio test를 적용해 더욱 좋은 매칭 결과만을 얻을 수 있음

# ratio test 적용
good = []

for match1, match2 in matches:
    # 첫 번째 매칭 결과의 distance는 당연히 두 번째 매칭 결과의 distance보다 값이 작을 것이기 때문에
    # 만약 첫 번째 매칭 결과의 distance 값이 두 번째 매칭 결과의 distance 값의 75%이하로 나오면
    # 좋은 매칭 결과로 판단하고 good 리스트에 append
    if match1.distance < 0.75*match2.distance:
        good.append([match1])

print(good)
print(len(good))        # 51
print(len(matches))     # 458
# knnMatch와 ratio test를 거쳐서 좋은 매칭점이라고 판단된 매칭 결과가 458->51개로 줄었음

res = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None,
                         flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

plt.title('KnnMatch + Ratio Test')
plt.imshow(res)
plt.show()

# 훨씬 좋은 결과가 나타났다
