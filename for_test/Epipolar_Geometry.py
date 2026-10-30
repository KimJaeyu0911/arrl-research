
import os


os.environ['QT_QPA_PLATFORM'] = 'offscreen' 
import cv2
import numpy as np
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # cv2와 충돌하는 Qt 백엔드를 안 씀
import matplotlib.pyplot as plt

# 에필로라 기하학을 적용할 두 이미지를 로드
img1 = cv2.imread('/home/arrl/jaeyu/dataset/20260911_125645.jpg', 0)  # 첫 번째 예시 이미지
img2 = cv2.imread('/home/arrl/jaeyu/dataset/20260911_125649.jpg', 0)  # 두 번째 예시 이미지

# 먼저 SIFT(또는 다른 Feature Detector도 가능)로 특징점 검출
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# 특징점 매칭 (KNN 매칭)
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# 좋은 매칭점 추출
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:  # 거리 기준
        good_matches.append(m)

# 매칭된 점들의 좌표 추출
points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

# 기본 행렬 계산**
F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)

# 에필로라 선 계산 및 시각화
def draw_epilines(img1, img2, lines, pts1, pts2):
    """에필로라 선 시각화"""
    r, c = img1.shape
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1_color = cv2.line(img1_color, (x0, y0), (x1, y1), color, 1)
        img1_color = cv2.circle(img1_color, tuple(map(int, pt1)), 5, color, -1)
        img2_color = cv2.circle(img2_color, tuple(map(int, pt2)), 5, color, -1)
    return img1_color, img2_color

# 첫 번째 이미지에서 에필로라 선 계산
lines1 = cv2.computeCorrespondEpilines(points2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img1_lines, img2_points = draw_epilines(img1, img2, lines1, points1, points2)

# 결과 시각화
plt.subplot(121), plt.imshow(img1_lines)
plt.subplot(122), plt.imshow(img2_points)
# plt.show()
# plt.show() 대신
plt.savefig('/home/arrl/jaeyu/dataset/epiline_result.png', dpi=150, bbox_inches='tight')