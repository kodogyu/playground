// visual odometry with 2 images.

#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <opengv/sac/Ransac.hpp>
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/StereoFactor.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include <pangolin/pangolin.h>

struct Landmark {
    int id;
    std::vector<int> observations;  // keypoint index vector
    std::vector<gtsam::Point3> measurements;  // 3D coordinate at corresponding pose
};
struct Frame{
    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::Point2f> keypoints_pts;
};
struct StereoFrame {
    int id;
    Frame left_frame;
    Frame right_frame;
    std::vector<int> keypoint_matches_R_L;  // R_L[right_idx] -> left_idx
    std::vector<int> keypoint_matches_L_R;  // L_R[left_idx] -> right_idx
    std::vector<std::shared_ptr<Landmark>> landmarks;
    std::vector<std::shared_ptr<Landmark>> matching_landmarks_with_last_frame;

    gtsam::Pose3 pose;

    double fx = 620.070096090849;
    double fy = 618.2102185572654;
    double cx = 325.29844703787114;
    double cy = 258.48711395621467;
    double baseline = 0.008;  // baseline = 0.008m (=8mm)
};

void stereoFrameMatchTemplate(const cv::Mat &image_left,
                            const cv::Mat &image_right,
                            const std::vector<cv::KeyPoint> &img_left_kps,
                            std::vector<int> &img_RL_kp_map,
                            std::vector<cv::Point2f> &img_right_kps_pts);

void stereoFrameDrawMatches(const cv::Mat &image_left,
                            const cv::Mat &image_right,
                            const std::vector<int> &img_kp_map,
                            const std::vector<cv::KeyPoint> &img_left_kps,
                            const std::vector<cv::Point2f> &img_right_kps_pts,
                            cv::Mat &result_image,
                            int number_matches = -1);

void stereoFrameDrawMatches(const cv::Mat &image_left,
                            const cv::Mat &image_right,
                            const std::vector<int> &img_kp_map,
                            const std::vector<cv::KeyPoint> &img_left_kps,
                            const std::vector<cv::KeyPoint> &img_right_kps,
                            cv::Mat &result_image,
                            int number_matches = -1);

void stereoFrameTriangulate(StereoFrame &stereo_image, const std::vector<int> &landmark_mask);

void writeVectors(const cv::Mat &rvec, const cv::Mat &tvec);

void writeBestPose(const Eigen::Isometry3d &best_pose);

void convertRt2Isometry(const cv::Mat &rvec, const cv::Mat &tvec, Eigen::Isometry3d &pose);

void displayPoses(const Eigen::Isometry3d &pose, const std::vector<gtsam::Point3> &keypoints_3d);

void displayPoses(const std::vector<Eigen::Isometry3d> &poses, const std::vector<gtsam::Point3> &keypoints_3d);

void displayPosesWithKeyPoints(const std::vector<Eigen::Isometry3d> &poses, const std::vector<std::vector<gtsam::Point3>> &keypoints_3d_vec);

int main(int argc, char** argv) {
    //** Image load **//
    if (argc != 2) {
        std::cout << "Usage: visual_odometry_example config_yaml" << std::endl;
        return 1;
    }

    cv::FileStorage config_file(argv[1], cv::FileStorage::READ);

    cv::Mat image1_left, image1_right, image2_left, image2_right;
    image1_left = cv::imread(config_file["image1_left"], cv::IMREAD_GRAYSCALE);
    image1_right = cv::imread(config_file["image1_right"], cv::IMREAD_GRAYSCALE);
    image2_left = cv::imread(config_file["image2_left"], cv::IMREAD_GRAYSCALE);
    image2_right = cv::imread(config_file["image2_right"], cv::IMREAD_GRAYSCALE);

    StereoFrame stereo_image1, stereo_image2;
    stereo_image1.id = 0;
    stereo_image1.left_frame.image = image1_left;
    stereo_image1.right_frame.image = image1_right;
    stereo_image2.id = 1;
    stereo_image2.left_frame.image = image2_left;
    stereo_image2.right_frame.image = image2_right;

    std::cout << "image1_left location: " << std::string(config_file["image1_left"]) << std::endl;

    //** Feature extraction **//
    std::vector<cv::KeyPoint> img1_left_kps;
    cv::Mat img1_left_descriptors;
    std::vector<cv::KeyPoint> img2_left_kps;
    cv::Mat img2_left_descriptors;
    // create orb feature extractor
    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    orb->detectAndCompute(image1_left, cv::Mat(), img1_left_kps, img1_left_descriptors);
    orb->detectAndCompute(image2_left, cv::Mat(), img2_left_kps, img2_left_descriptors);
    stereo_image1.left_frame.keypoints = img1_left_kps;
    stereo_image2.left_frame.keypoints = img2_left_kps;
    for (const auto kp: img1_left_kps) {
        stereo_image1.left_frame.keypoints_pts.push_back(kp.pt);
    }
    for (const auto kp: img2_left_kps) {
        stereo_image2.left_frame.keypoints_pts.push_back(kp.pt);
    }

    //** Feature matching **//
    // create a matcher
    cv::Ptr<cv::DescriptorMatcher> orb_matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);

    // image1 left & right (template matching)
    std::vector<int> img1_kp_map;
    std::vector<cv::Point2f> img1_right_kps_pts;
    stereoFrameMatchTemplate(image1_left, image1_right,
                            img1_left_kps, img1_kp_map, img1_right_kps_pts);
    stereo_image1.keypoint_matches_R_L = img1_kp_map;
    std::vector<int> img1_kp_map_L_R(stereo_image1.left_frame.keypoints.size(), -1);
    for (int i = 0; i < img1_kp_map.size(); i++) {
        img1_kp_map[img1_kp_map[i]] = i;
    }
    stereo_image1.keypoint_matches_L_R = img1_kp_map_L_R;
    stereo_image1.right_frame.keypoints_pts = img1_right_kps_pts;

    // image1 left & image2 left (matcher matching)
    std::vector<std::vector<cv::DMatch>> matches_img1_2_vec;
    double between_dist_thresh = 0.65;
    orb_matcher->knnMatch(img1_left_descriptors, img2_left_descriptors, matches_img1_2_vec, 2);

    std::vector<cv::DMatch> matches_img1_2;
    for (int i = 0; i < matches_img1_2_vec.size(); i++) {
        if (matches_img1_2_vec[i][0].distance < matches_img1_2_vec[i][1].distance * between_dist_thresh) {
            matches_img1_2.push_back(matches_img1_2_vec[i][0]);
        }
    }
    std::cout << "original features for image1&2: " << matches_img1_2_vec.size() << std::endl;
    std::cout << "good features for image1&2: " << matches_img1_2.size() << std::endl;

    // image2 left & right (template matching)
    std::vector<int> img2_kp_map;
    std::vector<cv::Point2f> img2_right_kps_pts;
    stereoFrameMatchTemplate(image2_left, image2_right,
                            img2_left_kps, img2_kp_map, img2_right_kps_pts);
    stereo_image2.keypoint_matches_R_L = img2_kp_map;
    std::vector<int> img2_kp_map_L_R(stereo_image2.left_frame.keypoints.size(), -1);
    for (int i = 0; i < img2_kp_map.size(); i++) {
        img2_kp_map[img2_kp_map[i]] = i;
    }
    stereo_image2.keypoint_matches_L_R = img2_kp_map_L_R;
    stereo_image2.right_frame.keypoints_pts = img2_right_kps_pts;

    // draw matches
    cv::Mat img1_matches, img2_matches;
    // image1 left & right matches
    stereoFrameDrawMatches(image1_left, image1_right,
                        img1_kp_map,
                        img1_left_kps, img1_right_kps_pts,
                        img1_matches);
    // image2 left & right matches
    stereoFrameDrawMatches(image2_left, image2_right,
                        img2_kp_map,
                        img2_left_kps, img2_right_kps_pts,
                        img2_matches);


    //** Triangulation **//
    // camera intrinsic parameters
    double fx = 620.070096090849;
    double fy = 618.2102185572654;
    double cx = 325.29844703787114;
    double cy = 258.48711395621467;
    double baseline = 0.008;  // baseline = 0.008m (=8mm)

    // stereo image1에서 모든 keypoint의 3D 좌표 계산
    std::vector<int> img1_landmark_mask(stereo_image1.left_frame.keypoints.size(), 0);
    stereoFrameTriangulate(stereo_image1, img1_landmark_mask);
    std::cout << "landmarks in image1: " << stereo_image1.landmarks.size() << std::endl;

    // image2에서 image1과 중복되는 landmark 검출
    std::vector<int> img2_landmark_mask(stereo_image2.left_frame.keypoints.size(), 0);
    for (auto plandmark: stereo_image1.landmarks) {
        for (int j = 0; j < matches_img1_2.size(); j++) {
            // landmark match found between stereo image1 (observations[0]) and stereo image2 (observations[1])
            if (plandmark->observations[0] == matches_img1_2[j].queryIdx) {
                plandmark->observations.push_back(matches_img1_2[j].trainIdx);
                stereo_image2.landmarks.push_back(plandmark);
                img2_landmark_mask[matches_img1_2[j].trainIdx] = 1;

                stereo_image2.matching_landmarks_with_last_frame.push_back(plandmark);
                break;
            }
        }
    }
    // image2에서 image1과 중복되지 않는 landmark 계산
    stereoFrameTriangulate(stereo_image2, img2_landmark_mask);
    std::cout << "landmarks in image2: " << stereo_image2.landmarks.size() << std::endl;

    // calculate img2_1_kp_map
    std::vector<int> img2_1_kp_map;
    for (const auto plandmark: stereo_image2.matching_landmarks_with_last_frame) {
        img2_1_kp_map.push_back(plandmark->observations[0]);
    }
    std::cout << "object points: " << img2_1_kp_map.size() << std::endl;

    // calculate img2_1_kp_map0 (old version)
    std::vector<int> img2_1_kp_map0;
    for (int i = 0; i < matches_img1_2.size(); i++) {
        for (int j = 0; j < img1_kp_map.size(); j++) {
            if (matches_img1_2[i].queryIdx == img1_kp_map[j]) {
                // keypoint in camera coordinate (normalized image plane)
                cv::KeyPoint keypoint = img1_left_kps[matches_img1_2[i].queryIdx];  // image1 keypoint
                // disparity to depth
                cv::Point img1_left_kp_pt = keypoint.pt;
                cv::Point img1_right_kp_pt = img1_right_kps_pts[j];
                float disparity = img1_left_kp_pt.x - img1_right_kp_pt.x;
                if (disparity > 0) {
                    // corresponding image2 keypoint
                    img2_1_kp_map0.push_back(matches_img1_2[i].queryIdx);
                }
            }
        }
    }

    // draw matches between image1 left & image2 left
    cv::Mat img1_2_matches_temp;
    stereoFrameDrawMatches(image1_left, image2_left, img2_1_kp_map, stereo_image1.left_frame.keypoints, stereo_image2.left_frame.keypoints, img1_2_matches_temp, 10);
    cv::imshow("image 1 & 2 matches", img1_2_matches_temp);


    //** PnP algorithm **//
    cv::Mat cameraMatrix(cv::Size(3, 3), CV_8UC1);
    cv::Mat distCoeffs(cv::Size(4, 1), CV_8UC1);
    cv::Mat rvec, tvec;

    cameraMatrix.at<float>(0, 0) = fx;
    cameraMatrix.at<float>(0, 1) = 0;
    cameraMatrix.at<float>(0, 2) = cx;
    cameraMatrix.at<float>(1, 0) = 0;
    cameraMatrix.at<float>(1, 1) = fy;
    cameraMatrix.at<float>(1, 2) = cy;
    cameraMatrix.at<float>(2, 0) = 0;
    cameraMatrix.at<float>(2, 1) = 0;
    cameraMatrix.at<float>(2, 2) = 1;

    // distCoeffs.at<float>(0,0) = 0;   // rvec: [0.7220963272379515; 0.4954686910202425; -1.216350494376585] // rotation matrix: [0.2873451776483652, 0.957253433933565, -0.03314531798953041; -0.6616346380781808, 0.1733475728411332, -0.729513690539967; -0.6925838249463989, 0.2315523314771105, 0.6831626184226783]
    // distCoeffs.at<float>(1,0) = 0;   // tvec: [1.862240918769173; 0; 0.09360853660961244]
    // distCoeffs.at<float>(2,0) = 0;
    // distCoeffs.at<float>(3,0) = 0;
    distCoeffs.at<float>(0,0) = 0.14669700865145466;    // rvec: [0.8978477333458742; 0.09755148042756377; -0.6906326389118941]  // rotation matrix: [0.7818537877805597, 0.5904450913001934, -0.2001980236982824; -0.5118961116618573, 0.4246450498730385, -0.7467522698216565; -0.3559031123756337, 0.6863316805873317, 0.6342568870918992]
    distCoeffs.at<float>(1,0) = -0.2735315348568459;    // tvec: [0.5493038210961142; -0.637639893508527; 6.432556693713797]
    distCoeffs.at<float>(2,0) = 0.007300675413449662;
    distCoeffs.at<float>(3,0) = -0.003734002028256388;

    // make object point vector
    std::vector<gtsam::Point3> keypoints_3d;
    for (const auto plandmark: stereo_image2.matching_landmarks_with_last_frame) {
        // landmark coordinate in first pose
        keypoints_3d.push_back(plandmark->measurements[0]);
    }
    // make vector of OpenCV format
    std::vector<cv::Point3f> keypoints_3d_cv;
    for (const auto plandmark: stereo_image2.matching_landmarks_with_last_frame) {
        cv::Point3f point;
        // landmark coordinate in first pose
        point.x = plandmark->measurements[0].x();
        point.y = plandmark->measurements[0].y();
        point.z = plandmark->measurements[0].z();

        keypoints_3d_cv.push_back(point);
    }
    std::vector<cv::Point2f> img2_left_kp_pts = stereo_image2.left_frame.keypoints_pts;

    std::vector<cv::Point3f> object_points(keypoints_3d_cv.begin(), keypoints_3d_cv.begin() + 4);
    std::vector<cv::Point2f> image_points(img2_left_kp_pts.begin(), img2_left_kp_pts.begin() + 4);
    // start time of solvePnP()
    const std::chrono::time_point<std::chrono::steady_clock> start_pnp = std::chrono::steady_clock::now();
    cv::solvePnP(object_points,
                image_points,
                cameraMatrix, distCoeffs,
                rvec, tvec);
    // end time of solvePnP()
    const std::chrono::time_point<std::chrono::steady_clock> end_pnp = std::chrono::steady_clock::now();
    const std::chrono::duration<double> sec_pnp = end_pnp - start_pnp;
    std::cout << "solvePnP elapsed time: " << sec_pnp.count() << std::endl;  // 0.00058992

    cv::Mat rotationMatrix;
    cv::Rodrigues(rvec, rotationMatrix);
    std::cout << "rvec: " << rvec << std::endl;
    std::cout << "rotation matrix: " << rotationMatrix << std::endl;
    std::cout << "tvec: " << tvec << std::endl;

    //** OpenGV RANSAC **//
    // get bearing vectors, world points
    opengv::bearingVectors_t bearing_vectors;
    opengv::points_t W_points;

    for (int i = 0; i < stereo_image2.matching_landmarks_with_last_frame.size(); i++) {
        const auto plandmark = stereo_image2.matching_landmarks_with_last_frame[i];
        cv::Point2f keypoint = img2_left_kp_pts[plandmark->observations[1]];
        gtsam::Point3 versor;
        versor[0] = keypoint.x / fx;
        versor[1] = keypoint.y / fx;
        versor[2] = 1;
        bearing_vectors.push_back(versor.normalized());
        W_points.push_back(plandmark->measurements[0]);
    }

    // create the central adapter
    opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearing_vectors, W_points);
    // create a RANSAC object
    opengv::sac::Ransac<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem> ransac;

    // create an AbsolutePoseSacProblem
    std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
        absposeproblem_ptr(
            new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
                adapter, opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem::EPNP ) );

    // create a RANSAC threshold
    double reprojection_error = 0.5;
    const double threshold = 1.0 - std::cos(std::atan(std::sqrt(2.0) * reprojection_error / fx));

    // run ransac
    ransac.sac_model_ = absposeproblem_ptr;
    ransac.threshold_ = threshold;
    ransac.max_iterations_ = 300;
    // start time of OpenGV RANSAC
    const std::chrono::time_point<std::chrono::steady_clock> start_opengv_ransac = std::chrono::steady_clock::now();
    ransac.computeModel();
    // end time of OpenGV RANSAC
    const std::chrono::time_point<std::chrono::steady_clock> end_opengv_ransac = std::chrono::steady_clock::now();
    const std::chrono::duration<double> sec_opengv_ransac = end_opengv_ransac - start_opengv_ransac;
    std::cout << "OpenGV RANSAC elapsed time: " << sec_opengv_ransac.count() << std::endl;  // 0.0271196

    // get the best pose
    opengv::transformation_t best_pose = ransac.model_coefficients_;
    Eigen::Isometry3d best_pose_eigen(best_pose);

    //** GTSAM optimize **//
    // create a graph
    gtsam::NonlinearFactorGraph graph;
    // stereo camera calibration object
    gtsam::Cal3_S2Stereo::shared_ptr K(
        new gtsam::Cal3_S2Stereo(fx, fy, 0, cx, cy, baseline));

    // create initial values
    gtsam::Values initial_estimates;

    // add factors
    // prior factor
    const auto priorNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1);
    gtsam::Pose3 first_pose;
    // graph.push_back(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('x', 0), first_pose, priorNoise));
    // stereo factors
    const auto noiseModel = gtsam::noiseModel::Isotropic::Sigma(3, 1);
    // pose1
    for (int i = 0; i < stereo_image1.landmarks.size(); i++) {
        const auto plandmark = stereo_image1.landmarks[i];
        cv::Point left_kp_pt = stereo_image1.left_frame.keypoints_pts[plandmark->observations[0]];
        cv::Point right_kp_pt = stereo_image1.right_frame.keypoints_pts[i];
        graph.emplace_shared<gtsam::GenericStereoFactor<gtsam::Pose3, gtsam::Point3>>(
            gtsam::StereoPoint2(left_kp_pt.x, right_kp_pt.x, left_kp_pt.y),
            noiseModel,
            gtsam::Symbol('x', stereo_image1.id),
            gtsam::Symbol('l', plandmark->id),
            K);
        // initial value of the landmark
        initial_estimates.insert(gtsam::Symbol('l', plandmark->id), plandmark->measurements[0]);
    }
    // pose2
    // TODO landmark 대응관계 정리하기
    for (int j = 0; j < stereo_image2.landmarks.size(); j++) {
        const auto plandmark = stereo_image2.landmarks[j];
        cv::Point left_kp_pt = stereo_image2.left_frame.keypoints_pts[plandmark->observations[1]];
        cv::Point right_kp_pt = stereo_image2.right_frame.keypoints_pts[j];
        graph.emplace_shared<gtsam::GenericStereoFactor<gtsam::Pose3, gtsam::Point3>>(
            gtsam::StereoPoint2(left_kp_pt.x, right_kp_pt.x, left_kp_pt.y),
            noiseModel,
            gtsam::Symbol('x', stereo_image2.id),
            gtsam::Symbol('l', plandmark->id),
            K);
        // initial value of the landmark
        if (!initial_estimates.exists(gtsam::Symbol('l', plandmark->id))) {
            initial_estimates.insert(gtsam::Symbol('l', plandmark->id), plandmark->measurements[1]);
        }
    }

    // add initial pose estimates
    initial_estimates.insert(gtsam::Symbol('x', stereo_image1.id), first_pose);
    initial_estimates.insert(gtsam::Symbol('x', stereo_image2.id), gtsam::Pose3(gtsam::Rot3(best_pose_eigen.rotation()), gtsam::Point3(best_pose_eigen.translation())));
    // constrain the first pose
    graph.emplace_shared<gtsam::NonlinearEquality<gtsam::Pose3> >(gtsam::Symbol('x', stereo_image1.id), first_pose);

    // optimize
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial_estimates);
    // start time of GTSAM LM optimization
    const std::chrono::time_point<std::chrono::steady_clock> start_gtsam_opt = std::chrono::steady_clock::now();
    gtsam::Values result = optimizer.optimize();
    // end time of GTSAM LM optimization
    const std::chrono::time_point<std::chrono::steady_clock> end_gtsam_opt = std::chrono::steady_clock::now();
    const std::chrono::duration<double> sec_gtsam_opt = end_gtsam_opt - start_gtsam_opt;
    std::cout << "GTSAM Optimization elapsed time: " << sec_gtsam_opt.count() << std::endl;  // 0.251295

    stereo_image1.pose = result.at<gtsam::Pose3>(gtsam::Symbol('x', stereo_image1.id));
    stereo_image2.pose = result.at<gtsam::Pose3>(gtsam::Symbol('x', stereo_image2.id));
    result.print("optimization result:\n");

    gtsam::Pose3 gtsam_pose = result.at<gtsam::Pose3>(gtsam::Symbol('x', stereo_image2.id));
    Eigen::Isometry3d gtsam_pose_eigen;
    gtsam_pose_eigen.matrix().block<3, 3>(0, 0) = gtsam_pose.rotation().matrix();
    gtsam_pose_eigen.matrix().block<3, 1>(0, 3) = gtsam_pose.translation().matrix();

    //** Visualize **//
    // write rotation, translation vector to .csv file
    writeVectors(rvec, tvec);
    writeBestPose(best_pose_eigen);

    // visualize through pangolin
    std::vector<Eigen::Isometry3d> poses;
    Eigen::Isometry3d pnp_pose;
    std::vector<std::vector<gtsam::Point3>> keypoints_3d_vec;

    // convert rvec, tvec to Isometry matrix
    convertRt2Isometry(rvec, tvec, pnp_pose);
    poses.push_back(pnp_pose);
    poses.push_back(best_pose_eigen);
    poses.push_back(gtsam_pose_eigen);

    // make keypoints vector with image1 and image2 3D keypoints
    keypoints_3d_vec.push_back(keypoints_3d);
    std::vector<gtsam::Point3> img2_keypoints_3d;
    for (const auto plandmark: stereo_image2.landmarks) {
        img2_keypoints_3d.push_back(plandmark->measurements[1]);
    }
    keypoints_3d_vec.push_back(img2_keypoints_3d);

    // display
    // void (*tdisplayPoses)(const std::vector<Eigen::Isometry3d>&, const std::vector<gtsam::Point3>&) = &displayPoses;  // function pointer for overloaded function
    // std::thread display_thread(tdisplayPoses, poses, keypoints_3d);
    std::thread display_thread(displayPosesWithKeyPoints, poses, keypoints_3d_vec);

    cv::waitKey(0);
    cv::destroyAllWindows();
    display_thread.join();

    return 0;
}


void stereoFrameMatchTemplate(const cv::Mat &image_left,
                            const cv::Mat &image_right,
                            const std::vector<cv::KeyPoint> &img_left_kps,
                            std::vector<int> &img_RL_kp_map,
                            std::vector<cv::Point2f> &img_right_kps_pts) {
    int templ_width = 30, templ_height = 30;
    int stripe_width = 50, stripe_height = 40;
    cv::Rect roi_templ, roi_stripe;
    cv::Mat templ, stripe;

    for (int i = 0; i < img_left_kps.size(); i++) {
        cv::KeyPoint img1_left_kp = img_left_kps[i];
        if (img1_left_kp.pt.x < stripe_width/2 || img1_left_kp.pt.x + stripe_width/2 > image_left.cols ||
            img1_left_kp.pt.y < stripe_height/2 || img1_left_kp.pt.y + stripe_height/2 > image_left.rows) {
                continue;
            }
        // template
        roi_templ = cv::Rect(cv::Point(img1_left_kp.pt) - cv::Point(templ_width/2, templ_height/2),
                        cv::Point(img1_left_kp.pt) + cv::Point(templ_width/2, templ_height/2));
        templ = cv::Mat(image_left, roi_templ);
        // stripe
        roi_stripe = cv::Rect(cv::Point(img1_left_kp.pt) - cv::Point(stripe_width/2, stripe_height/2),
                        cv::Point(img1_left_kp.pt) + cv::Point(stripe_width/2, stripe_height/2));
        stripe = cv::Mat(image_right, roi_stripe);

        // match template
        cv::Mat result;
        cv::matchTemplate(stripe, templ, result, CV_TM_SQDIFF_NORMED);
        // find best matching point
        double minval, maxval;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(result, &minval, &maxval, &minLoc, &maxLoc);
        img_RL_kp_map.push_back(i);  // img_RL_kp_map size = amount of right keypoints
        img_right_kps_pts.push_back(minLoc + cv::Point(img1_left_kp.pt) - cv::Point(stripe_width/2, stripe_height/2) + cv::Point(templ_width/2, templ_height/2));
    }

}

void stereoFrameDrawMatches(const cv::Mat &image_left,
                            const cv::Mat &image_right,
                            const std::vector<int> &img_kp_map,
                            const std::vector<cv::KeyPoint> &img_left_kps,
                            const std::vector<cv::Point2f> &img_right_kps_pts,
                            cv::Mat &result_image,
                            int number_matches) {
    cv::Mat left_image_temp, right_image_temp;
    cv::cvtColor(image_left, left_image_temp, cv::COLOR_GRAY2BGR);
    cv::cvtColor(image_right, right_image_temp, cv::COLOR_GRAY2BGR);

    cv::hconcat(left_image_temp, right_image_temp, result_image);

    int visual_matches = (number_matches < 0) ? img_right_kps_pts.size() : number_matches;

    for (int i = 0; i < visual_matches; i++) {
        cv::Point left_kp_pt;
        cv::Point right_kp_pt;

        left_kp_pt = img_left_kps[img_kp_map[i]].pt;
        right_kp_pt = img_right_kps_pts[i];
        right_kp_pt.x += left_image_temp.cols;

        cv::rectangle(result_image,
            left_kp_pt - cv::Point(5, 5),
            left_kp_pt + cv::Point(5, 5),
            cv::Scalar(0, 255, 0));  // green
        cv::rectangle(result_image,
            right_kp_pt - cv::Point(5, 5),
            right_kp_pt + cv::Point(5, 5),
            cv::Scalar(0, 255, 0));  // green
        cv::line(result_image, left_kp_pt, right_kp_pt, cv::Scalar(0, 255, 0));
    }
}

void stereoFrameDrawMatches(const cv::Mat &image_left,
                            const cv::Mat &image_right,
                            const std::vector<int> &img_kp_map,
                            const std::vector<cv::KeyPoint> &img_left_kps,
                            const std::vector<cv::KeyPoint> &img_right_kps,
                            cv::Mat &result_image,
                            int number_matches) {
    std::vector<cv::Point2f> img_right_kps_pts;
    for (const auto kp: img_right_kps) {
        img_right_kps_pts.push_back(kp.pt);
    }

    stereoFrameDrawMatches(image_left,
                        image_right,
                        img_kp_map,
                        img_left_kps,
                        img_right_kps_pts,
                        result_image,
                        number_matches);
}

void stereoFrameTriangulate(StereoFrame &stereo_image, const std::vector<int> &landmark_mask) {
    // stereo image에서 모든 keypoint의 3D 좌표 계산
    static int landmark_id = 0;
    for (int i = 0; i < stereo_image.left_frame.keypoints.size(); i++) {
        // if masked, skip calculating the corresponding landmark
        if (landmark_mask[i]) {
            continue;
        }
        cv::Point2f left_kp_pt = stereo_image.left_frame.keypoints[i].pt;
        cv::Point2f right_kp_pt = stereo_image.right_frame.keypoints_pts[stereo_image.keypoint_matches_L_R[i]];  //! this could lead out of index

        gtsam::Point3 versor;
        versor[0] = left_kp_pt.x / stereo_image.fx;
        versor[1] = left_kp_pt.y / stereo_image.fx;
        versor[2] = 1;

        double disparity = left_kp_pt.x - right_kp_pt.x;
        if (disparity > 0) {
            double depth = stereo_image.fx * stereo_image.baseline / disparity;

            // 3D keypoint is a landmark
            std::shared_ptr<Landmark> plandmark = std::make_shared<Landmark>();
            plandmark->id = landmark_id;
            plandmark->observations.push_back(stereo_image.keypoint_matches_R_L[i]);
            plandmark->measurements.push_back(depth * versor);

            stereo_image.landmarks.push_back(plandmark);
            landmark_id++;
        }
    }

}

void writeVectors(const cv::Mat &rvec, const cv::Mat &tvec) {
    // open file
    std::ofstream file("files/visual_odometry_example.csv");
    // write header
    file << "rvec[0]" << ", " << "rvec[1]" << ", " << "rvec[2]" << ", "
        << "tvec[0]" << ", " << "tvec[1]" << ", " << "tvec[2]" << std::endl;
    // write vectors
    file << rvec.at<double>(0, 0) << ", " << rvec.at<double>(0, 1) << ", " << rvec.at<double>(0, 2) << ", "
        << tvec.at<double>(0, 0) << ", " << tvec.at<double>(0, 1) << ", " << tvec.at<double>(0, 2) << std::endl;
    // close file
    file.close();
}

void writeBestPose(const Eigen::Isometry3d &best_pose) {
    // open file
    std::ofstream file("files/visual_odometry_example.csv", std::ios::app);
    // write header
    file << "qx" << ", " << "qy" << ", " << "qz" << ", " << "qw" << ", "
        << "tx" << ", " << "ty" << ", " << "tz" << std::endl;
    gtsam::Rot3 rotationMatrix(best_pose.rotation());
    gtsam::Vector q = rotationMatrix.quaternion();
    gtsam::Vector t = best_pose.translation();
    // write vectors
    file << q.x() << ", " << q.y() << ", " << q.z() << ", " << q.w() << ", "
        << t.x() << ", " << t.y() << ", " << t.z() << std::endl;
    // close file
    file.close();
}

void convertRt2Isometry(const cv::Mat &rvec, const cv::Mat &tvec, Eigen::Isometry3d &pose) {
    cv::Mat rotationMatrix;
    Eigen::Matrix3d rotationMatrix_eigen;
    Eigen::Vector3d translation_eigen;
    // rotation vector to rotation matrix
    cv::Rodrigues(rvec, rotationMatrix);
    // cv types to Eigen types
    cv::cv2eigen(rotationMatrix, rotationMatrix_eigen);
    cv::cv2eigen(rvec, translation_eigen);
    // make pose
    pose.matrix().block<3, 3>(0, 0) = rotationMatrix_eigen;
    pose.matrix().block<3, 1>(0, 3) = translation_eigen;
}

void displayPoses(const Eigen::Isometry3d &pose, const std::vector<gtsam::Point3> &keypoints_3d) {
    std::vector<Eigen::Isometry3d> poses;
    poses.push_back(pose);

    displayPoses(poses, keypoints_3d);
}

void displayPoses(const std::vector<Eigen::Isometry3d> &poses, const std::vector<gtsam::Point3> &keypoints_3d) {
    pangolin::CreateWindowAndBind("Visual Odometry Example", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState vis_camera(
        pangolin::ProjectionMatrix(1024, 768, 400, 400, 512, 384, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -3, -3, 0, 0, 0, 0.0, -1.0, 0.0));

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& vis_display =
        pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(vis_camera));

    const float blue[3] = {0, 0, 1};
    const float green[3] = {0, 1, 0};

    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        vis_display.Activate(vis_camera);

        // draw the original axis
        glLineWidth(3);
        glColor3f(1.0, 0.0, 0.0);
        glBegin(GL_LINES);
        glVertex3f(0, 0, 0);
        glVertex3f(1, 0, 0);
        glColor3f(0.0, 1.0, 0.0);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 1, 0);
        glColor3f(0.0, 0.0, 1.0);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 0, 1);
        glEnd();

        // draw map points
        glPointSize(5.0f);
        glBegin(GL_POINTS);
        for (int i = 0; i < keypoints_3d.size(); i++) {
            gtsam::Point3 point = keypoints_3d[i];

            glColor3f(1.0, 0.0, 0.0);
            glVertex3d(point[0], point[1], point[2]);
        }
        glEnd();

        // draw transformed axis
        Eigen::Vector3d last_center(0.0, 0.0, 0.0);

        for (auto cam_pose : poses) {
            Eigen::Vector3d Ow = cam_pose.translation();
            Eigen::Vector3d Xw = cam_pose * (0.1 * Eigen::Vector3d(1.0, 0.0, 0.0));
            Eigen::Vector3d Yw = cam_pose * (0.1 * Eigen::Vector3d(0.0, 1.0, 0.0));
            Eigen::Vector3d Zw = cam_pose * (0.1 * Eigen::Vector3d(0.0, 0.0, 1.0));
            glBegin(GL_LINES);
            glColor3f(1.0, 0.0, 0.0);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Xw[0], Xw[1], Xw[2]);
            glColor3f(0.0, 1.0, 0.0);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Yw[0], Yw[1], Yw[2]);
            glColor3f(0.0, 0.0, 1.0);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Zw[0], Zw[1], Zw[2]);
            glEnd();
            // draw odometry line
            glBegin(GL_LINES);
            glColor3f(0.0, 0.0, 0.0);
            glVertex3d(last_center[0], last_center[1], last_center[2]);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glEnd();

            last_center = Ow;
        }

        pangolin::FinishFrame();
    }
}

void displayPosesWithKeyPoints(const std::vector<Eigen::Isometry3d> &poses, const std::vector<std::vector<gtsam::Point3>> &keypoints_3d_vec) {
    pangolin::CreateWindowAndBind("Visual Odometry Example", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState vis_camera(
        pangolin::ProjectionMatrix(1024, 768, 400, 400, 512, 384, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -3, -3, 0, 0, 0, 0.0, -1.0, 0.0));

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& vis_display =
        pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(vis_camera));

    const float blue[3] = {0, 0, 1};
    const float green[3] = {0, 1, 0};

    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        vis_display.Activate(vis_camera);

        // draw the original axis
        glLineWidth(3);
        glColor3f(1.0, 0.0, 0.0);
        glBegin(GL_LINES);
        glVertex3f(0, 0, 0);
        glVertex3f(1, 0, 0);
        glColor3f(0.0, 1.0, 0.0);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 1, 0);
        glColor3f(0.0, 0.0, 1.0);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 0, 1);
        glEnd();

        // draw poses[0] map points
        glPointSize(5.0f);
        glBegin(GL_POINTS);
        for (int i = 0; i < keypoints_3d_vec[0].size(); i++) {
            gtsam::Point3 point = keypoints_3d_vec[0][i];

            glColor3f(1.0, 0.0, 0.0);  // red
            glVertex3d(point[0], point[1], point[2]);
        }
        glEnd();

        // draw poses[1] map points
        glPointSize(5.0f);
        glBegin(GL_POINTS);
        for (int i = 0; i < keypoints_3d_vec[1].size(); i++) {
            gtsam::Point3 point = poses[1] * keypoints_3d_vec[1][i];

            glColor3f(0.0, 0.0, 1.0);  // blue
            glVertex3d(point[0], point[1], point[2]);
        }
        glEnd();

        // draw poses[2] map points
        glPointSize(5.0f);
        glBegin(GL_POINTS);
        for (int i = 0; i < keypoints_3d_vec[1].size(); i++) {
            gtsam::Point3 point = poses[2] * keypoints_3d_vec[1][i];

            glColor3f(0.0, 1.0, 0.0);  // green
            glVertex3d(point[0], point[1], point[2]);
        }
        glEnd();

        // draw transformed axis
        std::vector<std::vector<float>> colors {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
        Eigen::Vector3d last_center(0.0, 0.0, 0.0);

        int color_idx = 0;
        for (auto cam_pose : poses) {
            Eigen::Vector3d Ow = cam_pose.translation();
            Eigen::Vector3d Xw = cam_pose * (0.1 * Eigen::Vector3d(1.0, 0.0, 0.0));
            Eigen::Vector3d Yw = cam_pose * (0.1 * Eigen::Vector3d(0.0, 1.0, 0.0));
            Eigen::Vector3d Zw = cam_pose * (0.1 * Eigen::Vector3d(0.0, 0.0, 1.0));
            glBegin(GL_LINES);
            glColor3f(1.0, 0.0, 0.0);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Xw[0], Xw[1], Xw[2]);
            glColor3f(0.0, 1.0, 0.0);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Yw[0], Yw[1], Yw[2]);
            glColor3f(0.0, 0.0, 1.0);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Zw[0], Zw[1], Zw[2]);
            glEnd();
            // draw odometry line
            glBegin(GL_LINES);
            glColor3f(colors[color_idx][0], colors[color_idx][1], colors[color_idx][2]);
            // glColor3f(0.0, 0.0, 0.0);
            glVertex3d(last_center[0], last_center[1], last_center[2]);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glEnd();

            last_center = Ow;
            color_idx++;
        }

        pangolin::FinishFrame();
    }
}
