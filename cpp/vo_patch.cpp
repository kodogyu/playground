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

// response comparison, for list sorting
bool compare_response(cv::KeyPoint first, cv::KeyPoint second)
{
  if (first.response < second.response) return true;
  else return false;
}

void triangulate2(cv::Mat intrinsic, std::vector<cv::Point2f> img0_kp_pts, std::vector<cv::Point2f> img1_kp_pts, Eigen::Isometry3d &cam1_pose, std::vector<Eigen::Vector3d> &landmarks) {
    Eigen::Matrix3d camera_intrinsic;
    cv::cv2eigen(intrinsic, camera_intrinsic);
    Eigen::MatrixXd prev_proj = Eigen::MatrixXd::Identity(3, 4);
    Eigen::MatrixXd curr_proj = Eigen::MatrixXd::Identity(3, 4);

    prev_proj = camera_intrinsic * prev_proj;
    curr_proj = camera_intrinsic * curr_proj * cam1_pose.matrix();

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

double calcReprojectionError(cv::Mat &intrinsic,
                            cv::Mat image0, std::vector<cv::Point2f> img0_kp_pts,
                            cv::Mat image1, std::vector<cv::Point2f> img1_kp_pts, cv::Mat mask,
                            Eigen::Isometry3d &cam1_pose, std::vector<Eigen::Vector3d> &landmarks) {
    double reproj_error0 = 0, reproj_error1 = 0;
    double inlier_reproj_error0 = 0, inlier_reproj_error1 = 0;
    int inlier_cnt = 0;
    cv::Mat image0_copy, image1_copy;
    cv::cvtColor(image0, image0_copy, cv::COLOR_GRAY2BGR);
    cv::cvtColor(image1, image1_copy, cv::COLOR_GRAY2BGR);

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

        reproj_error0 += error0;
        reproj_error1 += error1;

        if (mask.at<unsigned char>(i) == 1) {
            inlier_reproj_error0 += error0;
            inlier_reproj_error1 += error1;
            inlier_cnt++;
        }

        // draw images
        cv::rectangle(image0_copy,
                    measurement_point0 - cv::Point2f(5, 5),
                    measurement_point0 + cv::Point2f(5, 5),
                    mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255));  // green, (yellow)
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
                    mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255));  // green, (yellow)
        cv::circle(image1_copy,
                    projected_point1,
                    2,
                    mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0));  // red, (blue)
        cv::line(image1_copy,
                    measurement_point1,
                    projected_point1,
                    mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0));  // red, (blue)
    }
    cv::putText(image0_copy, "prev_frame",
                                    cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(image0_copy, "#landmarks: " + std::to_string(landmarks.size()),
                                    cv::Point(0, 40), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(image0_copy, "#inliers: " + std::to_string(inlier_cnt),
                                    cv::Point(0, 60), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(image1_copy, "curr_frame",
                                    cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(image1_copy, "#landmarks: " + std::to_string(landmarks.size()),
                                    cv::Point(0, 40), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(image1_copy, "#inliers: " + std::to_string(inlier_cnt),
                                    cv::Point(0, 60), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));

    cv::imwrite("vo_patch/reprojected_landmarks/prev_frame_proj.png", image0_copy);
    cv::imwrite("vo_patch/reprojected_landmarks/curr_frame_proj.png", image1_copy);

    std::cout << "RANSAC inliers: " << inlier_cnt << std::endl;
    std::cout << "inlier reprojected error: " << ((inlier_reproj_error0 + inlier_reproj_error1) / inlier_cnt) / 2 << std::endl;

    double reprojection_error = ((reproj_error0 + reproj_error1) / landmarks.size()) / 2;
    return reprojection_error;
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



int main(int argc, char** argv) {
    std::cout << CV_VERSION << std::endl;

    //**========== Parse config file ==========**//
    cv::FileStorage config_file(argv[1], cv::FileStorage::READ);

    //**========== Initialize variables ==========**//
    cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << config_file["fx"], config_file["s"], config_file["cx"],
                                                    0, config_file["fy"], config_file["cy"],
                                                    0, 0, 1);
    cv::Ptr<cv::ORB> orb = cv::ORB::create(config_file["num_features"], 1.2, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 25);

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
    cv::Mat prev_image = cv::imread(config_file["prev_frame"], cv::IMREAD_GRAYSCALE);
    std::string prev_id = std::string(config_file["prev_frame"]);
    int p_id = std::stoi(prev_id.substr(prev_id.length() - 10, 6));
    poses.push_back(Eigen::Isometry3d::Identity());

    // new Frame!
    cv::Mat curr_image = cv::imread(config_file["curr_frame"], cv::IMREAD_GRAYSCALE);
    std::string curr_id = std::string(config_file["curr_frame"]);
    int c_id = std::stoi(curr_id.substr(curr_id.length() - 10, 6));

    //**========== 1. Feature extraction ==========**//
    cv::Mat curr_image_descriptors;
    std::vector<cv::KeyPoint> curr_image_keypoints;
    cv::Mat prev_image_descriptors;
    std::vector<cv::KeyPoint> prev_image_keypoints;
    orb->detectAndCompute(prev_image, cv::Mat(), prev_image_keypoints, prev_image_descriptors);
    orb->detectAndCompute(curr_image, cv::Mat(), curr_image_keypoints, curr_image_descriptors);

    //**========== 2. Feature matching ==========**//
    // create a matcher
    cv::Ptr<cv::DescriptorMatcher> orb_matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);

    // image0 & image1 (matcher matching)
    std::vector<std::vector<cv::DMatch>> image_matches01_vec;
    std::vector<std::vector<cv::DMatch>> image_matches10_vec;
    double des_dist_thresh = config_file["des_dist_thresh"];
    orb_matcher->knnMatch(prev_image_descriptors, curr_image_descriptors, image_matches01_vec, 2);  // prev -> curr matches
    orb_matcher->knnMatch(curr_image_descriptors, prev_image_descriptors, image_matches10_vec, 2);  // curr -> prev matches

    // top matches
    std::vector<cv::DMatch> matches;
    orb_matcher->match(prev_image_descriptors, curr_image_descriptors, matches);
    std::sort(matches.begin(), matches.end());
    std::vector<cv::DMatch> top_matches(matches.begin(), matches.begin() + int(config_file["num_top_matches"]));

    // // good matches
    // std::vector<cv::DMatch> good_matches;
    // for (int i = 0; i < image_matches01_vec.size(); i++) {
    //     if (image_matches01_vec[i][0].distance < image_matches01_vec[i][1].distance * des_dist_thresh) {  // prev -> curr match에서 좋은가?
    //         int image1_keypoint_idx = image_matches01_vec[i][0].trainIdx;
    //         if (image_matches10_vec[image1_keypoint_idx][0].distance < image_matches10_vec[image1_keypoint_idx][1].distance * des_dist_thresh) {  // curr -> prev match에서 좋은가?
    //             if (image_matches01_vec[i][0].queryIdx == image_matches10_vec[image1_keypoint_idx][0].trainIdx)
    //                 good_matches.push_back(image_matches01_vec[i][0]);
    //         }
    //     }
    // }

    std::cout << "original features for image" + std::to_string(p_id) + "&" + std::to_string(c_id) + ": " << image_matches01_vec.size() << std::endl;
    // std::cout << "good features for image" + std::to_string(p_id) + "&" + std::to_string(c_id) + ": " << good_matches.size() << std::endl;
    std::cout << "top matches for image" + std::to_string(p_id) + "&" + std::to_string(c_id) + ": " << top_matches.size() << std::endl;

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
    std::vector<cv::Point2f> top_image0_kp_pts;
    std::vector<cv::Point2f> top_image1_kp_pts;
    // for (auto match : good_matches) {
    //     image0_kp_pts.push_back(prev_image_keypoints[match.queryIdx].pt);
    //     image1_kp_pts.push_back(curr_image_keypoints[match.trainIdx].pt);
    //     image0_kps.push_back(prev_image_keypoints[match.queryIdx]);
    //     image1_kps.push_back(curr_image_keypoints[match.trainIdx]);
    // }
    for (auto match : top_matches) {
        top_image0_kp_pts.push_back(prev_image_keypoints[match.queryIdx].pt);
        top_image1_kp_pts.push_back(curr_image_keypoints[match.trainIdx].pt);
    }

    cv::Mat mask;
    cv::Mat essential_mat = cv::findEssentialMat(top_image1_kp_pts, top_image0_kp_pts, intrinsic, cv::RANSAC, 0.999, 1.0, mask);
    // cv::Mat essential_mat = cv::findEssentialMat(image1_kp_pts, image0_kp_pts, intrinsic, cv::RANSAC, 0.999, 1.0, mask);
    cv::Mat ransac_matches;
    cv::drawMatches(prev_image, prev_image_keypoints,
                    curr_image, curr_image_keypoints,
                    top_matches, ransac_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), mask, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    // cv::drawMatches(prev_image, prev_image_keypoints,
    //                 curr_image, curr_image_keypoints,
    //                 good_matches, ransac_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), mask, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::putText(ransac_matches, "(vo_patch)frame" + std::to_string(p_id) + " & frame" + std::to_string(c_id),
                                cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(ransac_matches, "dist_thresh: " + std::to_string(des_dist_thresh),
                                cv::Point(0, 40), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(ransac_matches, "num_matches: " + std::to_string(top_matches.size()),
                                cv::Point(0, 60), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::imwrite("vo_patch/inter_frames/(vo_patch)frame"
            + std::to_string(p_id)
            + "&"
            + std::to_string(c_id)
            + "_kp_matches(ransac).png", ransac_matches);

    //**========== 3. Motion estimation ==========**//
    cv::Mat R, t;
    cv::recoverPose(essential_mat, top_image1_kp_pts, top_image0_kp_pts, intrinsic, R, t, mask);
    // cv::recoverPose(essential_mat, image1_kp_pts, image0_kp_pts, intrinsic, R, t, mask);

    Eigen::Matrix3d rotation_mat;
    Eigen::Vector3d translation_mat;
    cv::cv2eigen(R, rotation_mat);
    cv::cv2eigen(t, translation_mat);
    relative_pose.linear() = rotation_mat;
    relative_pose.translation() = translation_mat;
    poses.push_back(relative_pose);

    //**========== 4. Triangulation ==========**//
    std::vector<Eigen::Vector3d> landmarks;
    triangulate2(intrinsic, top_image0_kp_pts, top_image1_kp_pts, relative_pose, landmarks);

    // calculate reprojection error
    double reproj_error = calcReprojectionError(intrinsic, prev_image, top_image0_kp_pts, curr_image, top_image1_kp_pts, mask, relative_pose, landmarks);
    std::cout << "reprojection error: " << reproj_error << std::endl;


    //============ Evaluate ============//
    std::vector<Eigen::Isometry3d> gt_poses, aligned_poses;
    double rpe_rot, rpe_trans;
    loadGT(config_file["gt_path"], p_id, gt_poses);
    Eigen::Isometry3d aligned_est_pose = alignPose(gt_poses, relative_pose);
    aligned_poses.push_back(aligned_est_pose);
    calcRPE_rt(gt_poses, aligned_est_pose, rpe_rot, rpe_trans);

    std::cout << "PREr: " << rpe_rot << std::endl;
    std::cout << "PREt: " << rpe_trans << std::endl;

    //============ Log ============//
    logTrajectory(std::vector<Eigen::Isometry3d>{relative_pose});
    displayPoses(gt_poses, poses, aligned_poses);

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
        // draw aligned poses
        drawPoses(aligned_poses);

        pangolin::FinishFrame();
    }
}