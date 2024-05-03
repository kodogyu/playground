import numpy as np
import cv2 as cv
import glob
from copy import deepcopy

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((10*7,3), np.float32)
objp[:,:2] = np.mgrid[0:10,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('images/checkerboard*.jpg')

for fname in images:
 img = cv.imread(fname)
 img = cv.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

 gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

 # Find the chess board corners
 ret, corners = cv.findChessboardCorners(gray, (10,7), None)

 # If found, add object points, image points (after refining them)
 if ret == True:
    objpoints.append(objp)

    corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners2)
    # print(corners2)

    # Draw and display the corners
    cv.drawChessboardCorners(img, (10,7), corners2, ret)
    cv.imshow('img', img)
    cv.waitKey(0)

    cv.destroyAllWindows()

# Calibrate
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("camera matrix:\n", mtx)
print("distortion coeff:\n", dist)


# Undistortion
img = cv.imread('images/checkerboard1.jpg')
img = cv.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
print("new camera matrix:\n", newcameramtx)

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
 
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)

# undistort
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
 
# reprojection error
mean_error = 0
for i in range(len(objpoints)):
 imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
 error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
 mean_error += error
 
print( "total error: {}".format(mean_error/len(objpoints)) )

# essential matrix, recover pose
essential_mat = cv.findEssentialMat(imgpoints[0], imgpoints[1], mtx, method=cv.RANSAC, maxIters=500, threshold=1)
ret, R, t, mask = cv.recoverPose(essential_mat, imgpoints[0], imgpoints[1], mtx )

projected_points, _ = cv.projectPoints(objpoints[0], R, t, mtx, dist)

result_img = deepcopy(dst)
for pt in projected_points:
  cv.circle(result_img, pt, 1, (0, 255, 0))

cv.imshow("result image", result_img)
cv.waitKey(0)
cv.destroyAllWindows()
