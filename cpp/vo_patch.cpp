#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <sstream>
#include <list>

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>

#include <pangolin/pangolin.h>

void logTrajectory(std::vector<Eigen::Isometry3d> poses);
void displayPoses(const std::vector<Eigen::Isometry3d> &gt_poses, const std::vector<Eigen::Isometry3d> &est_poses, const std::vector<Eigen::Isometry3d> &aligned_poses);
void displayFramesAndLandmarks(const std::vector<Eigen::Isometry3d> &est_poses, const std::vector<Eigen::Vector3d> &landmarks);


cv::Mat readImage(std::string image_file, int img_entry_idx, cv::Mat intrinsic, cv::Mat distortion) {
    cv::Mat result;

    // read the image
    cv::Mat image = cv::imread(image_file, cv::IMREAD_GRAYSCALE);

    cv::undistort(image, result, intrinsic, distortion);

    // // fisheye image processing (rectification)
    // if (pConfig_->is_fisheye_) {
    //     cv::Size new_size(640, 480);
    //     cv::Mat Knew = (cv::Mat_<double>(3, 3) << new_size.width/4, 0, new_size.width/2,
    //                                             0, new_size.height/4, new_size.height/2,
    //                                             0, 0, 1);
    //     cv::omnidir::undistortImage(image, result, pCamera_->intrinsic_, pCamera_->distortion_, pCamera_->xi_, cv::omnidir::RECTIFY_PERSPECTIVE, Knew, new_size);
    // }

    return result;
}

// response comparison, for list sorting
bool compare_response(cv::KeyPoint first, cv::KeyPoint second)
{
  if (first.response < second.response) return true;
  else return false;
}

void triangulate2(cv::Mat intrinsic, std::vector<cv::Point2f> img0_kp_pts, std::vector<cv::Point2f> img1_kp_pts, Eigen::Isometry3d &cam1_pose, const cv::Mat &mask, std::vector<Eigen::Vector3d> &landmarks) {
    Eigen::Matrix3d camera_intrinsic;
    cv::cv2eigen(intrinsic, camera_intrinsic);
    Eigen::MatrixXd prev_proj = Eigen::MatrixXd::Identity(3, 4);
    Eigen::MatrixXd curr_proj = Eigen::MatrixXd::Identity(3, 4);

    prev_proj = camera_intrinsic * prev_proj;
    curr_proj = camera_intrinsic * curr_proj * cam1_pose.inverse().matrix();

    for (int i = 0; i < img0_kp_pts.size(); i++) {
        if (mask.at<unsigned char>(i) != 1) {
            continue;
        }

        Eigen::Matrix4d A;
        A.row(0) = img0_kp_pts[i].x * prev_proj.row(2) - prev_proj.row(0);
        A.row(1) = img0_kp_pts[i].y * prev_proj.row(2) - prev_proj.row(1);
        A.row(2) = img1_kp_pts[i].x * curr_proj.row(2) - curr_proj.row(0);
        A.row(3) = img1_kp_pts[i].y * curr_proj.row(2) - curr_proj.row(1);

        Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector4d point_3d_homo = svd.matrixV().col(3);
        Eigen::Vector3d point_3d = point_3d_homo.head(3) / point_3d_homo[3];
        landmarks.push_back(point_3d);
    }
}

void triangulate2(cv::Mat intrinsic, std::vector<cv::Point2f> img0_kp_pts, std::vector<cv::Point2f> img1_kp_pts, Eigen::Isometry3d &cam1_pose, std::vector<Eigen::Vector3d> &landmarks) {
    Eigen::Matrix3d camera_intrinsic;
    cv::cv2eigen(intrinsic, camera_intrinsic);
    Eigen::MatrixXd prev_proj = Eigen::MatrixXd::Identity(3, 4);
    Eigen::MatrixXd curr_proj = Eigen::MatrixXd::Identity(3, 4);

    prev_proj = camera_intrinsic * prev_proj;
    curr_proj = camera_intrinsic * curr_proj * cam1_pose.inverse().matrix();

    for (int i = 0; i < img0_kp_pts.size(); i++) {
        Eigen::Matrix4d A;
        A.row(0) = img0_kp_pts[i].x * prev_proj.row(2) - prev_proj.row(0);
        A.row(1) = img0_kp_pts[i].y * prev_proj.row(2) - prev_proj.row(1);
        A.row(2) = img1_kp_pts[i].x * curr_proj.row(2) - curr_proj.row(0);
        A.row(3) = img1_kp_pts[i].y * curr_proj.row(2) - curr_proj.row(1);

        Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector4d point_3d_homo = svd.matrixV().col(3);
        Eigen::Vector3d point_3d = point_3d_homo.head(3) / point_3d_homo[3];
        landmarks.push_back(point_3d);
    }
}

void triangulate_cv(cv::Mat intrinsic, std::vector<cv::Point2f> img0_kp_pts, std::vector<cv::Point2f> img1_kp_pts, Eigen::Isometry3d &cam1_pose, std::vector<Eigen::Vector3d> &landmarks) {
    cv::Mat points4D;
    cv::Mat prev_proj_mat, curr_proj_mat;
    cv::Mat prev_pose, curr_pose;
    Eigen::MatrixXd pose_temp;
    pose_temp = Eigen::Isometry3d::Identity().inverse().matrix().block<3, 4>(0, 0);
    cv::eigen2cv(pose_temp, prev_pose);
    pose_temp = cam1_pose.inverse().matrix().block<3, 4>(0, 0);
    cv::eigen2cv(pose_temp, curr_pose);
    prev_proj_mat = intrinsic * prev_pose;
    curr_proj_mat = intrinsic * curr_pose;

    std::cout << "prev_proj_mat:\n" << prev_proj_mat << std::endl;
    std::cout << "curr_proj_mat:\n" << curr_proj_mat << std::endl;

    // triangulate
    cv::triangulatePoints(prev_proj_mat, curr_proj_mat, img0_kp_pts, img1_kp_pts, points4D);

    // homogeneous -> 3D
    for (int i = 0; i < points4D.cols; i++) {
        cv::Mat point_homo = points4D.col(i);
        Eigen::Vector3d point_3d(point_homo.at<float>(0) / point_homo.at<float>(3),
                                point_homo.at<float>(1) / point_homo.at<float>(3),
                                point_homo.at<float>(2) / point_homo.at<float>(3));

        landmarks.push_back(point_3d);
    }
}

int triangulate2_count(cv::Mat intrinsic, std::vector<cv::Point2f> img0_kp_pts, std::vector<cv::Point2f> img1_kp_pts, Eigen::Isometry3d &cam1_pose, const cv::Mat &mask/*, std::vector<Eigen::Vector3d> &landmarks*/) {
    Eigen::Matrix3d camera_intrinsic;
    cv::cv2eigen(intrinsic, camera_intrinsic);
    Eigen::MatrixXd prev_proj = Eigen::MatrixXd::Identity(3, 4);
    Eigen::MatrixXd curr_proj = Eigen::MatrixXd::Identity(3, 4);

    prev_proj = camera_intrinsic * prev_proj;
    curr_proj = camera_intrinsic * curr_proj * cam1_pose.inverse().matrix();

    int positive_cnt = 0;
    for (int i = 0; i < img0_kp_pts.size(); i++) {
        if (mask.at<unsigned char>(i) != 1) {
            continue;
        }

        Eigen::Matrix4d A;
        A.row(0) = img0_kp_pts[i].x * prev_proj.row(2) - prev_proj.row(0);
        A.row(1) = img0_kp_pts[i].y * prev_proj.row(2) - prev_proj.row(1);
        A.row(2) = img1_kp_pts[i].x * curr_proj.row(2) - curr_proj.row(0);
        A.row(3) = img1_kp_pts[i].y * curr_proj.row(2) - curr_proj.row(1);

        Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector4d point_3d_homo = svd.matrixV().col(3);
        Eigen::Vector3d point_3d = point_3d_homo.head(3) / point_3d_homo[3];
        // landmarks.push_back(point_3d);

        Eigen::Vector3d cam1_point_3d = cam1_pose.inverse().matrix().block<3, 4>(0, 0) * point_3d_homo;
        // std::cout << "landmark(world) z: " << point_3d.z() << ", (camera) z: " << cam1_point_3d.z() << std::endl;
        if (point_3d.z() > 0 && point_3d.z() < 70) {
            positive_cnt++;
        }
    }
    return positive_cnt;
}

double calcReprojectionError(cv::Mat &intrinsic,
                            int p_id,
                            cv::Mat image0, std::vector<cv::Point2f> img0_kp_pts,
                            int c_id,
                            cv::Mat image1, std::vector<cv::Point2f> img1_kp_pts, cv::Mat mask,
                            Eigen::Isometry3d &cam1_pose, std::vector<Eigen::Vector3d> &landmarks,
                            int num_matches) {
    double reproj_error0 = 0, reproj_error1 = 0;
    double inlier_reproj_error0 = 0, inlier_reproj_error1 = 0;
    int inlier_cnt = 0;
    cv::Mat image0_copy, image1_copy;

    if (image0.type() == CV_8UC1) {
        cv::cvtColor(image0, image0_copy, cv::COLOR_GRAY2BGR);
    }
    else {
        image0.copyTo(image0_copy);
    }
    if (image1.type() == CV_8UC1) {
        cv::cvtColor(image1, image1_copy, cv::COLOR_GRAY2BGR);
    }
    else {
        image1.copyTo(image1_copy);
    }

    Eigen::Matrix3d camera_intrinsic;
    cv::cv2eigen(intrinsic, camera_intrinsic);
    Eigen::MatrixXd prev_proj = Eigen::MatrixXd::Identity(3, 4);
    Eigen::MatrixXd curr_proj = Eigen::MatrixXd::Identity(3, 4);

    prev_proj = camera_intrinsic * prev_proj;
    curr_proj = camera_intrinsic * curr_proj * cam1_pose.matrix();

    for (int i = 0; i < img0_kp_pts.size(); i++) {
        // calculate reprojection error
        Eigen::Vector3d img0_x_tilde, img1_x_tilde;

        double error0 = 0, error1 = 0;
        Eigen::Vector3d landmark_point_3d = landmarks[i];
        Eigen::Vector4d landmark_point_3d_homo(landmark_point_3d[0],
                                                landmark_point_3d[1],
                                                landmark_point_3d[2],
                                                1);

        img0_x_tilde = prev_proj * landmark_point_3d_homo;
        img1_x_tilde = curr_proj * landmark_point_3d_homo;

        cv::Point2f projected_point0(img0_x_tilde[0] / img0_x_tilde[2],
                                    img0_x_tilde[1] / img0_x_tilde[2]);
        cv::Point2f projected_point1(img1_x_tilde[0] / img1_x_tilde[2],
                                    img1_x_tilde[1] / img1_x_tilde[2]);
        cv::Point2f measurement_point0 = img0_kp_pts[i];
        cv::Point2f measurement_point1 = img1_kp_pts[i];

        cv::Point2f error_vector0 = projected_point0 - measurement_point0;
        cv::Point2f error_vector1 = projected_point1 - measurement_point1;
        error0 = sqrt(error_vector0.x * error_vector0.x + error_vector0.y * error_vector0.y);
        error1 = sqrt(error_vector1.x * error_vector1.x + error_vector1.y * error_vector1.y);


        if (mask.at<unsigned char>(i) == 1) {
            reproj_error0 += error0;
            reproj_error1 += error1;

            inlier_reproj_error0 += error0;
            inlier_reproj_error1 += error1;
            inlier_cnt++;
        }

        // draw images
        cv::rectangle(image0_copy,
                    measurement_point0 - cv::Point2f(5, 5),
                    measurement_point0 + cv::Point2f(5, 5),
                    mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 125, 255));  // green, (orange)
        cv::circle(image0_copy,
                    projected_point0,
                    2,
                    mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0));  // red, (blue)
        cv::line(image0_copy,
                    measurement_point0,
                    projected_point0,
                    mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0));  // red, (blue)
        cv::rectangle(image1_copy,
                    measurement_point1 - cv::Point2f(5, 5),
                    measurement_point1 + cv::Point2f(5, 5),
                    mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 125, 255));  // green, (orange)
        cv::circle(image1_copy,
                    projected_point1,
                    2,
                    mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0));  // red, (blue)
        cv::line(image1_copy,
                    measurement_point1,
                    projected_point1,
                    mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0));  // red, (blue)
    }
    cv::putText(image0_copy, "frame" + std::to_string(p_id),
                                    cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(image0_copy, "#landmarks: " + std::to_string(landmarks.size()),
                                    cv::Point(0, 40), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(image0_copy, "#inliers: " + std::to_string(inlier_cnt),
                                    cv::Point(0, 60), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(image1_copy, "frame" + std::to_string(c_id),
                                    cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(image1_copy, "#landmarks: " + std::to_string(landmarks.size()),
                                    cv::Point(0, 40), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(image1_copy, "#inliers: " + std::to_string(inlier_cnt),
                                    cv::Point(0, 60), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));

    cv::imwrite("vo_patch/reprojected_landmarks/frame" + std::to_string(p_id) + "_" + std::to_string(num_matches) + "_proj.png", image0_copy);
    cv::imwrite("vo_patch/reprojected_landmarks/frame" + std::to_string(c_id) + "_" + std::to_string(num_matches) + "_proj.png", image1_copy);

    std::cout << "pose inliers: " << inlier_cnt << std::endl;
    std::cout << "inlier reprojected error: " << ((inlier_reproj_error0 + inlier_reproj_error1) / inlier_cnt) / 2 << std::endl;

    double reprojection_error = ((reproj_error0 + reproj_error1) / landmarks.size()) / 2;
    return reprojection_error;
}

void drawKeypoints(int p_id,
                    cv::Mat image0, std::vector<cv::Point2f> img0_kp_pts,
                    int c_id,
                    cv::Mat image1, std::vector<cv::Point2f> img1_kp_pts,
                    int num_features
                    ) {
    cv::Mat image0_copy, image1_copy;

    if (image0.type() == CV_8UC1) {
        cv::cvtColor(image0, image0_copy, cv::COLOR_GRAY2BGR);
    }
    else {
        image0.copyTo(image0_copy);
    }
    if (image1.type() == CV_8UC1) {
        cv::cvtColor(image1, image1_copy, cv::COLOR_GRAY2BGR);
    }
    else {
        image1.copyTo(image1_copy);
    }

    for (int i = 0; i < img0_kp_pts.size(); i++) {
        // draw images
        cv::rectangle(image0_copy,
                    img0_kp_pts[i] - cv::Point2f(5, 5),
                    img0_kp_pts[i] + cv::Point2f(5, 5),
                    cv::Scalar(0, 255, 0));  // green, (yellow)
        cv::circle(image0_copy,
                    img0_kp_pts[i],
                    1,
                    cv::Scalar(0, 0, 255));  // red, (blue)
        cv::rectangle(image1_copy,
                    img1_kp_pts[i] - cv::Point2f(5, 5),
                    img1_kp_pts[i] + cv::Point2f(5, 5),
                    cv::Scalar(0, 255, 0));  // green, (yellow)
        cv::circle(image1_copy,
                    img1_kp_pts[i],
                    1,
                    cv::Scalar(0, 0, 255));  // red, (blue)
    }
    cv::putText(image0_copy, "frame" + std::to_string(p_id),
                                    cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(image0_copy, "#features: " + std::to_string(num_features),
                                    cv::Point(0, 40), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(image1_copy, "frame" + std::to_string(c_id),
                                    cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(image1_copy, "#features: " + std::to_string(num_features),
                                    cv::Point(0, 40), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));

    cv::imwrite("vo_patch/intra_frames/frame" + std::to_string(p_id) + "_" + std::to_string(num_features) + "_kps.png", image0_copy);
    cv::imwrite("vo_patch/intra_frames/frame" + std::to_string(c_id) + "_" + std::to_string(num_features) + "_kps.png", image1_copy);
}

void drawKeypointsSingleImage(int id,
                    cv::Mat image, std::vector<cv::KeyPoint> img_kps,
                    std::string folder,
                    std::string tail
                    ) {
    cv::Mat image_copy;

    if (image.type() == CV_8UC1) {
        cv::cvtColor(image, image_copy, cv::COLOR_GRAY2BGR);
    }
    else {
        image.copyTo(image_copy);
    }

    for (int i = 0; i < img_kps.size(); i++) {
        cv::Point2f img_kp_pt = img_kps[i].pt;
        // draw images
        cv::rectangle(image_copy,
                    img_kp_pt - cv::Point2f(5, 5),
                    img_kp_pt + cv::Point2f(5, 5),
                    cv::Scalar(0, 255, 0));  // green, (yellow)
        cv::circle(image_copy,
                    img_kp_pt,
                    1,
                    cv::Scalar(0, 0, 255));  // red, (blue)
    }
    cv::putText(image_copy, "frame" + std::to_string(id),
                                    cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(image_copy, "#features: " + std::to_string(img_kps.size()),
                                    cv::Point(0, 40), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));

    cv::imwrite(folder + "/frame" + std::to_string(id) + "_kps" + tail + ".png", image_copy);
}

void loadGT(std::string gt_path, int prev_frame_id, std::vector<Eigen::Isometry3d> &gt_poses) {
    std::ifstream gt_poses_file(gt_path);
    double r11, r12, r13, r21, r22, r23, r31, r32, r33, t1, t2, t3;
    std::string line;

    for (int l = 0; l < prev_frame_id; l++) {
        std::getline(gt_poses_file, line);
    }

    for (int i = 0; i < 2; i++) {
        std::getline(gt_poses_file, line);
        std::stringstream ssline(line);

        // KITTI format
        ssline
            >> r11 >> r12 >> r13 >> t1
            >> r21 >> r22 >> r23 >> t2
            >> r31 >> r32 >> r33 >> t3;

        // std::cout << "gt_pose[" << prev_frame_id + i << "]: "
        //             << r11 << " " << r12 << " " << r13 << " " << t1
        //             << r21 << " " << r22 << " " << r23 << " " << t2
        //             << r31 << " " << r32 << " " << r33 << " " << t3 << std::endl;

        Eigen::Matrix3d rotation_mat;
        rotation_mat << r11, r12, r13,
                        r21, r22, r23,
                        r31, r32, r33;
        Eigen::Vector3d translation_mat;
        translation_mat << t1, t2, t3;

        Eigen::Isometry3d gt_pose;
        gt_pose.linear() = rotation_mat;
        gt_pose.translation() = translation_mat;

        gt_poses.push_back(gt_pose);
    }
}

Eigen::Isometry3d alignPose(const std::vector<Eigen::Isometry3d> &gt_poses, const Eigen::Isometry3d &est_rel_pose) {
    Eigen::Isometry3d gt_rel_pose = gt_poses[0].inverse() * gt_poses[1];

    Eigen::Isometry3d aligned_est_rel_pose(est_rel_pose);
    double scale = gt_rel_pose.translation().norm() / est_rel_pose.translation().norm();
    aligned_est_rel_pose.translation() = est_rel_pose.translation() * scale;
    // std::cout << "distance before align: " << est_rel_pose.translation().norm() << std::endl;
    // std::cout << "distance after align: " << aligned_est_rel_pose.translation().norm() << std::endl;
    // std::cout << "gt distance: " << gt_rel_pose.translation().norm() << std::endl;

    return aligned_est_rel_pose;
}

std::vector<Eigen::Isometry3d> calcRPE(const std::vector<Eigen::Isometry3d> &gt_poses, const Eigen::Isometry3d &est_rel_pose) {
    std::vector<Eigen::Isometry3d> rpe_vec;

    Eigen::Isometry3d gt_rel_pose = gt_poses[0].inverse() * gt_poses[1];

    // calculate the relative pose error
    Eigen::Isometry3d relative_pose_error = gt_rel_pose.inverse() * est_rel_pose;
    rpe_vec.push_back(relative_pose_error);

    return rpe_vec;
}

void calcRPE_rt(const std::vector<Eigen::Isometry3d> &gt_poses, const Eigen::Isometry3d &est_rel_pose, double &rpe_rot, double &rpe_trans) {
    std::vector<Eigen::Isometry3d> rpe_vec = calcRPE(gt_poses, est_rel_pose);

    double acc_trans_error = 0;
    double acc_theta = 0;

    for (int i = 0; i < rpe_vec.size(); i++) {
        Eigen::Isometry3d rpe = rpe_vec[i];

        // RPE rotation
        Eigen::Matrix3d rotation = rpe.rotation();
        double theta = acos((rotation.trace() - 1) / 2);
        acc_theta += theta;

        // RPE translation
        double translation_error = rpe.translation().norm() * rpe.translation().norm();
        acc_trans_error += translation_error;
    }

    // mean RPE
    rpe_rot = acc_theta / double(rpe_vec.size());
    rpe_trans = sqrt(acc_trans_error / double(rpe_vec.size()));
}

void filterKeypoints(const cv::Size image_size, const cv::Size patch_size, int kps_per_patch, std::vector<cv::KeyPoint> &img_kps, cv::Mat &img_descriptor) {

    int row_iter = (image_size.height - 1) / patch_size.height;
    int col_iter = (image_size.width - 1) / patch_size.width;

    std::vector<std::vector<int>> bins(row_iter + 1, std::vector<int>(col_iter + 1));
    std::cout << "bin size: (" << bins.size() << ", " << bins[0].size() << ")" << std::endl;

    std::vector<cv::KeyPoint> filtered_kps;
    cv::Mat filtered_descriptors;
    std::vector<cv::Mat> filtered_descriptors_vec;

    for (int i = 0; i < img_kps.size(); i++) {
        cv::Point2f kp_pt = img_kps[i].pt;

        int bin_cnt = bins[kp_pt.y / patch_size.height][kp_pt.x / patch_size.width];
        if (bin_cnt < kps_per_patch) {
            filtered_kps.push_back(img_kps[i]);
            filtered_descriptors_vec.push_back(img_descriptor.row(i));

            bins[kp_pt.y / patch_size.height][kp_pt.x / patch_size.width]++;
        }
    }

    for (auto desc : filtered_descriptors_vec) {
        filtered_descriptors.push_back(desc);
    }

    img_kps = filtered_kps;
    img_descriptor = filtered_descriptors;
}

void drawGrid(cv::Mat &image, const cv::Size patch_size, int kps_per_patch) {
    cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);

    int row_iter = (image.rows) / patch_size.height;
    int col_iter = (image.cols) / patch_size.width;

    // 가로선
    for (int i = 0; i < row_iter; i++) {
        int row = (i + 1) * patch_size.height;
        cv::line(image, cv::Point2f(0, row), cv::Point2f(image.cols, row), cv::Scalar(0, 0, 0), 2);
    }
    // 세로선
    for (int j = 0; j < col_iter; j++) {
        int col = (j + 1) * patch_size.width;
        cv::line(image, cv::Point2f(col, 0), cv::Point2f(col, image.rows), cv::Scalar(0, 0, 0), 2);
    }

    cv::putText(image, "patch width: " + std::to_string(patch_size.width),
                                    cv::Point(image.cols - 200, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(image, "patch height: " + std::to_string(patch_size.height),
                                    cv::Point(image.cols - 200, 40), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(image, "keypoints / patch: " + std::to_string(kps_per_patch),
                                    cv::Point(image.cols - 250, 60), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
}



int main(int argc, char** argv) {
    std::cout << CV_VERSION << std::endl;

    //**========== Parse config file ==========**//
    cv::FileStorage config_file(argv[1], cv::FileStorage::READ);

    //**========== Initialize variables ==========**//
    cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << config_file["fx"], config_file["s"], config_file["cx"],
                                                    0, config_file["fy"], config_file["cy"],
                                                    0, 0, 1);
    cv::Mat distortion = (cv::Mat_<double>(5, 1) << config_file["k1"], config_file["k2"], config_file["p1"], config_file["p2"], config_file["k3"]);
    cv::Ptr<cv::ORB> orb = cv::ORB::create(config_file["num_features"], 1.2, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 25);
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    int num_matches = config_file["num_matches"];

    std::vector<cv::Mat> keypoints_3d_vec;
    Eigen::Isometry3d relative_pose;
    std::vector<Eigen::Isometry3d> poses;
    std::vector<double> scales;
    std::vector<int64_t> feature_extraction_costs;
    std::vector<int64_t> feature_matching_costs;
    std::vector<int64_t> motion_estimation_costs;
    std::vector<int64_t> triangulation_costs;
    std::vector<int64_t> scaling_costs;
    std::vector<int64_t> total_time_costs;

    //**========== 0. Image load ==========**//
    // read images
    cv::Mat prev_image_dist = cv::imread(config_file["prev_frame"], cv::IMREAD_GRAYSCALE);
    cv::Mat prev_image = prev_image_dist.clone();
    std::cout << "prev_image size: " << prev_image.size << std::endl;
    cv::undistort(prev_image_dist, prev_image, intrinsic, distortion);
    std::string prev_id = std::string(config_file["prev_frame"]);
    int p_id = std::stoi(prev_id.substr(prev_id.length() - 10, 6));
    // int p_id = 13;
    poses.push_back(Eigen::Isometry3d::Identity());

    // new Frame!
    cv::Mat curr_image_dist = cv::imread(config_file["curr_frame"], cv::IMREAD_GRAYSCALE);
    cv::Mat curr_image = curr_image_dist.clone();
    std::cout << "curr_image size: " << curr_image.size << std::endl;
    cv::undistort(curr_image_dist, curr_image, intrinsic, distortion);
    std::string curr_id = std::string(config_file["curr_frame"]);
    int c_id = std::stoi(curr_id.substr(curr_id.length() - 10, 6));
    // int c_id = 14;

    //**========== 1. Feature extraction ==========**//
    cv::Mat curr_image_descriptors;
    std::vector<cv::KeyPoint> curr_image_keypoints;
    cv::Mat prev_image_descriptors;
    std::vector<cv::KeyPoint> prev_image_keypoints;
    orb->detectAndCompute(prev_image, cv::Mat(), prev_image_keypoints, prev_image_descriptors);
    orb->detectAndCompute(curr_image, cv::Mat(), curr_image_keypoints, curr_image_descriptors);
    // sift->detectAndCompute(prev_image, cv::Mat(), prev_image_keypoints, prev_image_descriptors);
    // sift->detectAndCompute(curr_image, cv::Mat(), curr_image_keypoints, curr_image_descriptors);

    //**========== 2. Feature matching ==========**//
    // create a matcher
    cv::Ptr<cv::DescriptorMatcher> orb_matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    // cv::Ptr<cv::BFMatcher> bf_matcher = cv::BFMatcher::create();

    // image0 & image1 (matcher matching)
    std::vector<std::vector<cv::DMatch>> image_matches01_vec;
    std::vector<std::vector<cv::DMatch>> image_matches10_vec;
    double des_dist_thresh = config_file["des_dist_thresh"];
    // bf_matcher->knnMatch(prev_image_descriptors, curr_image_descriptors, image_matches01_vec, 2);  // prev -> curr matches
    orb_matcher->knnMatch(prev_image_descriptors, curr_image_descriptors, image_matches01_vec, 2);  // prev -> curr matches
    orb_matcher->knnMatch(curr_image_descriptors, prev_image_descriptors, image_matches10_vec, 2);  // curr -> prev matches

    // good matches
    std::vector<cv::DMatch> good_matches;
    // for (int i = 0; i < image_matches01_vec.size(); i++) {
    //     if (image_matches01_vec[i][0].distance < image_matches01_vec[i][1].distance * 0.95) {
    //         good_matches.push_back(image_matches01_vec[i][0]);
    //     }
    // }

    for (int i = 0; i < image_matches01_vec.size(); i++) {
        if (image_matches01_vec[i][0].distance < image_matches01_vec[i][1].distance * des_dist_thresh) {  // prev -> curr match에서 좋은가?
            int image1_keypoint_idx = image_matches01_vec[i][0].trainIdx;
            if (image_matches10_vec[image1_keypoint_idx][0].distance < image_matches10_vec[image1_keypoint_idx][1].distance * des_dist_thresh) {  // curr -> prev match에서 좋은가?
                if (image_matches01_vec[i][0].queryIdx == image_matches10_vec[image1_keypoint_idx][0].trainIdx)
                    good_matches.push_back(image_matches01_vec[i][0]);
            }
        }
    }
    // if (num_matches > -1) {
    //     std::sort(good_matches.begin(), good_matches.end());
    //     good_matches.resize(num_matches);
    // }

    std::cout << "original features for image" + std::to_string(p_id) + "&" + std::to_string(c_id) + ": " << image_matches01_vec.size() << std::endl;
    std::cout << "good features for image" + std::to_string(p_id) + "&" + std::to_string(c_id) + ": " << good_matches.size() << std::endl;

    // cv::Mat image_matches;
    // cv::drawMatches(prev_image, prev_image_keypoints, curr_image, curr_image_keypoints, good_matches, image_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), cv::Mat(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    // cv::imwrite("vo_patch/inter_frames/(vo_patch)frame"
    //         + std::to_string(p_id)
    //         + "&"
    //         + std::to_string(c_id)
    //         + "_kp_matches(raw).png", image_matches);

    // RANSAC
    std::vector<cv::KeyPoint> image0_kps;
    std::vector<cv::KeyPoint> image1_kps;
    std::vector<cv::Point2f> image0_kp_pts;
    std::vector<cv::Point2f> image1_kp_pts;
    for (auto match : good_matches) {
        image0_kp_pts.push_back(prev_image_keypoints[match.queryIdx].pt);
        image1_kp_pts.push_back(curr_image_keypoints[match.trainIdx].pt);
        image0_kps.push_back(prev_image_keypoints[match.queryIdx]);
        image1_kps.push_back(curr_image_keypoints[match.trainIdx]);
    }

    // draw keypoints
    // drawKeypoints(p_id, prev_image, image0_kp_pts, c_id, curr_image, image1_kp_pts, num_matches);
    drawKeypoints(p_id, prev_image, image0_kp_pts, c_id, curr_image, image1_kp_pts, good_matches.size());

    cv::Mat mask;
    // cv::Mat essential_mat = cv::findEssentialMat(frame2_kp_pts_test, frame1_kp_pts_test, intrinsic, cv::RANSAC, 0.999, 1.0, mask);
    cv::Mat essential_mat = cv::findEssentialMat(image0_kp_pts, image1_kp_pts, intrinsic, cv::RANSAC, 0.999, 1.0, 500, mask);
    std::cout << "essential matrix: \n" << essential_mat << std::endl;

    int essential_inlier = 0;
    for (int i = 0; i < mask.rows; i++) {
        if (mask.at<unsigned char>(i) == 1) {
            essential_inlier++;
        }
    }
    std::cout << "essential inlier: " << essential_inlier << std::endl;

    // // draw matches 
    // cv::Mat ransac_matches;
    // cv::drawMatches(prev_image, prev_image_keypoints,
    //                 curr_image, curr_image_keypoints,
    //                 good_matches, ransac_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), mask, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    // cv::putText(ransac_matches, "(vo_patch)frame" + std::to_string(p_id) + " & frame" + std::to_string(c_id),
    //                             cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    // cv::putText(ransac_matches, "dist_thresh: " + std::to_string(des_dist_thresh),
    //                             cv::Point(0, 40), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    // cv::putText(ransac_matches, "num_matches: " + std::to_string(good_matches.size()),
    //                             cv::Point(0, 60), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    // cv::imwrite("vo_patch/inter_frames/(vo_patch)frame"
    //         + std::to_string(p_id)
    //         + "&"
    //         + std::to_string(c_id)
    //         + "_kp_matches(ransac).png", ransac_matches);

    //**========== 3. Motion estimation ==========**//
    cv::Mat R, t;
    cv::recoverPose(essential_mat, image0_kp_pts, image1_kp_pts, intrinsic, R, t, mask);

    Eigen::Matrix3d rotation_mat;
    Eigen::Vector3d translation_mat;

    cv::cv2eigen(R, rotation_mat);
    cv::cv2eigen(t, translation_mat);

    relative_pose.linear() = rotation_mat.transpose();
    relative_pose.translation() = - rotation_mat.transpose() * translation_mat;

    std::cout << "relative pose:\n" << relative_pose.matrix() << std::endl;

    poses.push_back(relative_pose);

    //**========== 4. Triangulation ==========**//
    std::vector<Eigen::Vector3d> landmarks;
    // triangulate2(intrinsic, frame1_kp_pts_test, frame2_kp_pts_test, relative_pose, landmarks);
    // triangulate2(intrinsic, image0_kp_pts, image1_kp_pts, relative_pose, landmarks);
    // triangulate2(intrinsic, image0_kp_pts, image1_kp_pts, relative_pose, mask, landmarks);
    triangulate_cv(intrinsic, image0_kp_pts, image1_kp_pts, relative_pose, landmarks);

    // for (int i = 0; i < landmarks.size(); i++) {
    //     std::cout << "cv::Point3d(" << landmarks[i].transpose() << ")" << std::endl;
    // }

    // calculate reprojection error & save the images
    // double reproj_error = calcReprojectionError(intrinsic, p_id, prev_image, frame1_kp_pts_test, c_id, curr_image, frame2_kp_pts_test, mask, relative_pose, landmarks, 10);
    double reproj_error = calcReprojectionError(intrinsic, p_id, prev_image, image0_kp_pts, c_id, curr_image, image1_kp_pts, mask, relative_pose, landmarks, num_matches);
    std::cout << "reprojection error: " << reproj_error << std::endl;


    //============ Evaluate ============//
    std::vector<Eigen::Isometry3d> gt_poses, aligned_poses;
    double rpe_rot, rpe_trans;
    loadGT(config_file["gt_path"], p_id, gt_poses);
    Eigen::Isometry3d aligned_est_pose = alignPose(gt_poses, relative_pose);
    aligned_poses.push_back(aligned_est_pose);
    calcRPE_rt(gt_poses, aligned_est_pose, rpe_rot, rpe_trans);

    std::cout << "RPEr: " << rpe_rot << std::endl;
    std::cout << "RPEt: " << rpe_trans << std::endl;

    //============ Log ============//
    logTrajectory(std::vector<Eigen::Isometry3d>{relative_pose});
    // displayPoses(gt_poses, poses, aligned_poses);
    displayFramesAndLandmarks(poses, landmarks);

    return 0;
}

void logTrajectory(std::vector<Eigen::Isometry3d> poses) {
    std::ofstream trajectory_file("vo_patch/trajectory.csv");
    trajectory_file << "qw,qx,qy,qz,x,y,z\n";
    for (auto pose : poses) {
        Eigen::Quaterniond quaternion(pose.rotation());
        Eigen::Vector3d position = pose.translation();
        trajectory_file << quaternion.w() << "," << quaternion.x() << "," << quaternion.y() << "," << quaternion.z() << ","
                    << position.x() << "," << position.y() << "," << position.z() << "\n";
    }
}

void drawGT(const std::vector<Eigen::Isometry3d> &_gt_poses) {
    Eigen::Isometry3d first_pose = _gt_poses[0];
    Eigen::Vector3d last_center(0.0, 0.0, 0.0);

    for(auto gt_pose : _gt_poses) {
        gt_pose = first_pose.inverse() * gt_pose;
        Eigen::Vector3d Ow = gt_pose.translation();
        Eigen::Vector3d Xw = gt_pose * (0.1 * Eigen::Vector3d(1.0, 0.0, 0.0));
        Eigen::Vector3d Yw = gt_pose * (0.1 * Eigen::Vector3d(0.0, 1.0, 0.0));
        Eigen::Vector3d Zw = gt_pose * (0.1 * Eigen::Vector3d(0.0, 0.0, 1.0));
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
        glColor3f(0.0, 0.0, 1.0); // blue
        glVertex3d(last_center[0], last_center[1], last_center[2]);
        glVertex3d(Ow[0], Ow[1], Ow[2]);
        glEnd();

        last_center = Ow;
    }
}
void drawPoses(const std::vector<Eigen::Isometry3d> &poses) {
    Eigen::Vector3d last_center(0.0, 0.0, 0.0);

    for(auto pose : poses) {
        Eigen::Vector3d Ow = pose.translation();
        Eigen::Vector3d Xw = pose * (0.1 * Eigen::Vector3d(1.0, 0.0, 0.0));
        Eigen::Vector3d Yw = pose * (0.1 * Eigen::Vector3d(0.0, 1.0, 0.0));
        Eigen::Vector3d Zw = pose * (0.1 * Eigen::Vector3d(0.0, 0.0, 1.0));
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
        glColor3f(1.0, 1.0, 0.0); // yellow
        glVertex3d(last_center[0], last_center[1], last_center[2]);
        glVertex3d(Ow[0], Ow[1], Ow[2]);
        glEnd();

        last_center = Ow;
    }
}

void displayPoses(const std::vector<Eigen::Isometry3d> &gt_poses, const std::vector<Eigen::Isometry3d> &est_poses, const std::vector<Eigen::Isometry3d> &aligned_poses) {
    pangolin::CreateWindowAndBind("Visual Odometry Patch", 1024, 768);
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


        // draw transformed axis
        Eigen::Vector3d last_center(0.0, 0.0, 0.0);

        for (auto cam_pose : est_poses) {
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

        // draw gt
        drawGT(gt_poses);
        // // draw aligned poses
        // drawPoses(aligned_poses);

        pangolin::FinishFrame();
    }
}

void displayFramesAndLandmarks(const std::vector<Eigen::Isometry3d> &est_poses, const std::vector<Eigen::Vector3d> &landmarks) {
    pangolin::CreateWindowAndBind("Visual Odometry Viewer", 1024, 768);
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

    const float red[3] = {1, 0, 0};
    const float orange[3] = {1, 0.5, 0};
    const float yellow[3] = {1, 1, 0};
    const float green[3] = {0, 1, 0};
    const float blue[3] = {0, 0, 1};
    const float navy[3] = {0, 0.02, 1};
    const float purple[3] = {0.5, 0, 1};
    std::vector<const float*> colors {red, orange, yellow, green, blue, navy, purple};


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

        int color_idx = 0;
        // draw transformed axis
        Eigen::Vector3d last_center(0.0, 0.0, 0.0);
        for (auto cam_pose : est_poses) {
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
            glVertex3d(last_center[0], last_center[1], last_center[2]);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glEnd();

            // draw map points in world coordinate
            glPointSize(5.0f);
            glBegin(GL_POINTS);
            for (auto landmark : landmarks) {
                glColor3f(colors[color_idx][0], colors[color_idx][1], colors[color_idx][2]);
                glVertex3d(landmark[0], landmark[1], landmark[2]);
            }
            glEnd();

            last_center = Ow;

            color_idx++;
            color_idx = color_idx % colors.size();
        }

        pangolin::FinishFrame();
    }
}



