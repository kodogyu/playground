import cv2
import numpy as np

# Essential Matrix 추정을 위한 매칭점 설정
pts1 = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], dtype=np.float32)
pts2 = np.array([[2, 1], [3, 2], [4, 3], [5, 4], [6, 5]], dtype=np.float32)

# 카메라 내부 파라미터 설정 (예시값, 실제 값으로 변경해야 함)
K = np.array([[1000, 0, 320],
              [0, 1000, 240],
              [0, 0, 1]], dtype=np.float32)

# Essential Matrix 추정
E, mask = cv2.findEssentialMat(pts1, pts2, K)

# Relative pose 복구
retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

print("Rotation Matrix:")
print(R)
print("\nTranslation Vector:")
print(t)
