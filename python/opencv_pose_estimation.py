import numpy as np
import cv2 as cv
import glob
from copy import deepcopy
print(cv.__version__)

# Find feature correspondence with SIFT
def SIFT(img1, img2):
    """Featrue extraction & description"""
    sift = cv.xfeatures2d.SIFT_create()
    img1_kp, img1_des = sift.detectAndCompute(img1, None)
    img2_kp, img2_des = sift.detectAndCompute(img2, None)
    print("img1 keypoints shape:", np.shape(img1_kp))
    print("img2 keypoints shape:", np.shape(img2_kp))
    print("img1 descriptor shape:", img1_des.shape)
    print("img2 descriptor shape:", img2_des.shape)

    bf = cv.BFMatcher_create(cv.NORM_L2)
    # bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, False)
    matches = bf.knnMatch(img1_des, img2_des, k=2)
    matches_good = [m1 for m1, m2 in matches if m1.distance < 0.6*m2.distance]

    sorted_matches = sorted(matches_good, key=lambda x: x.distance)
    res = cv.drawMatches(img1, img1_kp, img2, img2_kp, sorted_matches, img2, flags=2) 

    return matches_good, img1_kp, img2_kp


# Find feature correspondence with SIFT
def ORB(img1, img2):
    """Featrue extraction & description"""
    orb = cv.ORB_create()
    img1_kp, img1_des = orb.detectAndCompute(img1, None)
    img2_kp, img2_des = orb.detectAndCompute(img2, None)
    print("img1 keypoints shape:", np.shape(img1_kp))
    print("img2 keypoints shape:", np.shape(img2_kp))
    print("img1 descriptor shape:", img1_des.shape)
    print("img2 descriptor shape:", img2_des.shape)

    bf = cv.BFMatcher_create(cv.NORM_HAMMING)
    matches = bf.knnMatch(img1_des, img2_des, k=2)
    matches_good = [m1 for m1, m2 in matches if m1.distance < 0.6*m2.distance]

    sorted_matches = sorted(matches_good, key=lambda x: x.distance)
    res = cv.drawMatches(img1, img1_kp, img2, img2_kp, sorted_matches, img2, flags=2) 

    return matches_good, img1_kp, img2_kp


# Preprocessing
def Estimation_E(matches_good, img1_kp, img2_kp, mtx):
    '''Essential Matrix Estimation'''
    query_idx = [match.queryIdx for match in matches_good]
    train_idx = [match.trainIdx for match in matches_good]
    p1 = np.float32([img1_kp[ind].pt for ind in query_idx]) # 픽셀 좌표
    p2 = np.float32([img2_kp[ind].pt for ind in train_idx])

# KITTI
# fx: 718.856
# fy: 718.856
# s: 0.0
# cx: 607.1928
# cy: 185.2157

    # Find Essential Matrix and Inliers with RANSAC, Intrinsic parameter
    # E, mask = cv2.findEssentialMat(p1, p2, method=cv2.RANSAC, focal=3092.8, pp=(2016, 1512), maxIters = 500, threshold=1) 
    E, mask = cv.findEssentialMat(p1, p2, mtx, method=cv.RANSAC, maxIters = 500, threshold=1) 
    print("mask sum:", sum(mask))
    p1 = p1[mask.ravel()==1] # left image inlier
    p2 = p2[mask.ravel()==1] # right image inlier

    return E, p1, p2

def drawReprojectionError(image1, pt_inlier1, reprojected_points1, image2, pt_inlier2, reprojected_points2):
    result_image1 = deepcopy(image1)
    result_image2 = deepcopy(image2)

    for i in range(len(reprojected_points1)):
        kp1_pt = tuple(map(int, pt_inlier1[i]))
        kp2_pt = tuple(map(int, pt_inlier2[i]))

        # original keypoint
        result_image1 = cv.circle(result_image1, kp1_pt, 3, (0, 255, 0))
        result_image2 = cv.circle(result_image2, kp2_pt, 3, (0, 255, 0))

        # reprojected point
        reproj_pt1 = reprojected_points1[i].reshape(2).astype(np.int16)
        reproj_pt2 = reprojected_points2[i].reshape(2).astype(np.int16)
        result_image1 = cv.circle(result_image1, reproj_pt1, 2, (0, 0, 255))
        result_image2 = cv.circle(result_image2, reproj_pt2, 2, (0, 0, 255))

        # reprojection error line
        result_image1 = cv.line(result_image1, kp1_pt, reproj_pt1, (0, 0, 255), 2)
        result_image2 = cv.line(result_image2, kp2_pt, reproj_pt2, (0, 0, 255), 2)

    return result_image1, result_image2




if __name__ == "__main__":
    # # 핸드폰 (image size /2)
    # mtx = np.asarray([[1.72757570e+03, 0.00000000e+00, 1.00911566e+03],
    #                     [0.00000000e+00, 1.59389769e+03, 4.67881818e+02],
    #                     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    # dist = np.asarray([4.76866262e-01, -3.34715903e+00,  1.08608341e-03,  2.34895698e-03, 7.78932109e+00])

    # KITTI
    mtx = np.asarray([[718.856, 0.00000000e+00, 607.1928],
                        [0.00000000e+00, 718.856, 185.2157],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((10*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:10,0:7].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    img_file1 = "../cpp/vo_patch/source_frames/000000.png"
    img_file2 = "../cpp/vo_patch/source_frames/000001.png"
    image_files = [img_file1, img_file2]
    # image_files = glob.glob('images/test*.jpg')

    images = []

    for fname in image_files:
        img = cv.imread(fname)
        # img = cv.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # # Undistortion
        # h, w = img.shape[:2]
        # newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        # print("new camera matrix:\n", newcameramtx)

        # # undistort
        # dst = cv.undistort(img, mtx, dist, None, newcameramtx)

        # # crop the image
        # x, y, w, h = roi
        # dst = dst[y:y+h, x:x+w]

        # images.append(dst)
        images.append(img)

    # matches_good, img1_kp, img2_kp = SIFT(images[0], images[1])
    matches_good, img1_kp, img2_kp = ORB(images[0], images[1])

    # draw Keypoints
    kp_image1 = deepcopy(images[0])
    kp_image1 = cv.drawKeypoints(kp_image1, img1_kp, None)
    kp_image2 = deepcopy(images[1])
    kp_image2 = cv.drawKeypoints(kp_image2, img2_kp, None)
    cv.imwrite("kp_image1.png", kp_image1)
    cv.imwrite("kp_image2.png", kp_image2)

    # draw matches
    matches_image12 = cv.drawMatches(images[0], img1_kp, images[1], img2_kp, matches_good, None)
    cv.imwrite("matches_image12.png", matches_image12)


    # essential matrix
    essential_mat, p1_inlier, p2_inlier = Estimation_E(matches_good, img1_kp, img2_kp, mtx)
    print("essential matrix:\n", essential_mat )
    rec_ret, R, t, mask = cv.recoverPose(essential_mat, p1_inlier, p2_inlier, mtx)
    print("recover pose R:\n", R)
    # recover_pose_rvec = cv.Rodrigues(R)
    # print("recover pose rvec:", recover_pose_rvec)
    print("recover pose tvec:", t)
    relative_pose = np.hstack((R, t))
    print("relative pose:\n", relative_pose)


    # triangulation
    proj_mat1 = mtx @ np.eye(3, 4)
    proj_mat2 = mtx @ relative_pose
    print("proj_mat1:\n", proj_mat1)
    print("proj_mat2:\n", proj_mat2)
    proj_points1 = p1_inlier.reshape(-1, 2).transpose()
    proj_points2 = p2_inlier.reshape(-1, 2).transpose()
    points4D = cv.triangulatePoints(proj_mat1, proj_mat2, proj_points1, proj_points2)

    points3D = [point[:-1]/point[3] for point in points4D.transpose()]
    points3D = np.asarray(points3D)
    projected_points1, _ = cv.projectPoints(points3D, np.eye(3, 3), np.zeros((3, 1)), mtx, None)
    projected_points2, _ = cv.projectPoints(points3D, R, t, mtx, None)

    result_img1 = deepcopy(images[0])
    result_img2 = deepcopy(images[1])
    for pt in projected_points1:
        cv.circle(result_img1, pt[0].astype(np.int16), 2, (0, 0, 255))
    for pt in projected_points2:
        cv.circle(result_img2, pt[0].astype(np.int16), 2, (0, 0, 255))
    cv.imwrite("result_image1.png", result_img1)
    cv.imwrite("result_image2.png", result_img2)

    # draw reprojection error
    reproj_err_image1, reproj_err_image2 = drawReprojectionError(images[0], p1_inlier, projected_points1,
                                                                 images[1], p2_inlier, projected_points2)
    cv.imwrite("reproj_err_image1.png", reproj_err_image1)
    cv.imwrite("reproj_err_image2.png", reproj_err_image2)

    # show images
    cv.imshow("result image1", result_img1)
    cv.imshow("result image2", result_img2)
    cv.waitKey(0)
    cv.destroyAllWindows()