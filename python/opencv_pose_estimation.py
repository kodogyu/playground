import numpy as np
import cv2 as cv
import glob
 
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype(np.uint16))
    x = tuple(imgpts[0].ravel())
    img = cv.line(img, corner, x, (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

# # Load previously saved data
# with np.load('B.npz') as X:
#  mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

# camera_matrix: !!opencv-matrix
#    rows: 3
#    cols: 3
#    dt: d
#    data: [ 5.3591573396163199e+02, 0., 3.4228315473308373e+02, 0.,
#        5.3591573396163199e+02, 2.3557082909788173e+02, 0., 0., 1. ]
# distortion_coefficients: !!opencv-matrix
#    rows: 5
#    cols: 1
#    dt: d
#    data: [ -2.6637260909660682e-01, -3.8588898922304653e-02,
#        1.7831947042852964e-03, -2.8122100441115472e-04,
#        2.3839153080878486e-01 ]

mtx = np.asarray([[ 5.3591573396163199e+02, 0., 3.4228315473308373e+02],
                  [0., 5.3591573396163199e+02, 2.3557082909788173e+02],
                  [0., 0., 1. ]])
dist = np.asarray([-2.6637260909660682e-01, -3.8588898922304653e-02, 1.7831947042852964e-03, -2.8122100441115472e-04, 2.3839153080878486e-01])


# main
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

files = glob.glob('python/images/left*.jpg')
for fname in files:
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (7,6),None)

    if ret == True:
       corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

    # Find the rotation and translation vectors.
    ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
    print("rvecs:", rvecs, ", tvecs:", tvecs)
    # project 3D points to image plane
    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

    # draw corners to image
    for corner in corners2.reshape(-1, 2):
        cv.circle(img, corner.astype(np.uint16), 3, (0, 255, 0), 1)

    img = draw(img,corners2,imgpts.astype(np.uint16))
    cv.imshow('img',img)

    k = cv.waitKey(0) & 0xFF
    if k == ord('s'):
        cv.imwrite(fname[:6]+'.png', img)

cv.destroyAllWindows()