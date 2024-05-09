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

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/StereoFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

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

void drawKeypoints(int p_id,
                    cv::Mat image0, std::vector<cv::Point2f> img0_kp_pts,
                    int c_id,
                    cv::Mat image1, std::vector<cv::Point2f> img1_kp_pts,
                    const cv::Mat mask,
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
        bool is_masked = false;
        if (mask.at<unsigned char>(i) == 1) {
            is_masked = true;
        }

        // draw images
        cv::rectangle(image0_copy,
                    img0_kp_pts[i] - cv::Point2f(5, 5),
                    img0_kp_pts[i] + cv::Point2f(5, 5),
                    is_masked ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 125, 255));  // green, (orange)
        cv::circle(image0_copy,
                    img0_kp_pts[i],
                    1,
                    is_masked ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0));  // red, (blue)
        cv::rectangle(image1_copy,
                    img1_kp_pts[i] - cv::Point2f(5, 5),
                    img1_kp_pts[i] + cv::Point2f(5, 5),
                    is_masked ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 125, 255));  // green, (orange)
        cv::circle(image1_copy,
                    img1_kp_pts[i],
                    1,
                    is_masked ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0));  // red, (blue)
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

void triangulate(cv::Mat intrinsic, std::vector<cv::Point2f> img0_kp_pts, std::vector<cv::Point2f> img1_kp_pts, Eigen::Isometry3d &cam1_pose, std::vector<Eigen::Vector3d> &landmarks, std::vector<Eigen::Vector4d> &landmarks_4D) {
    std::cout << "----- triangulate -----" << std::endl;

    Eigen::Matrix3d camera_intrinsic;
    cv::cv2eigen(intrinsic, camera_intrinsic);
    Eigen::MatrixXd prev_proj = Eigen::MatrixXd::Identity(3, 4);
    Eigen::MatrixXd curr_proj = Eigen::MatrixXd::Identity(3, 4);

    prev_proj = camera_intrinsic * prev_proj;
    curr_proj = camera_intrinsic * curr_proj * cam1_pose.inverse().matrix();
    std::cout << "prev_proj: \n" << prev_proj << std::endl;
    std::cout << "curr_proj: \n" << curr_proj << std::endl;

    for (int i = 0; i < img0_kp_pts.size(); i++) {
        Eigen::MatrixXd A(6, 4);
        A.row(0) = img0_kp_pts[i].x * prev_proj.row(2) - prev_proj.row(0);
        A.row(1) = img0_kp_pts[i].y * prev_proj.row(2) - prev_proj.row(1);
        A.row(2) = img0_kp_pts[i].x * prev_proj.row(1) - img0_kp_pts[i].y * prev_proj.row(0);
        A.row(3) = img1_kp_pts[i].x * curr_proj.row(2) - curr_proj.row(0);
        A.row(4) = img1_kp_pts[i].y * curr_proj.row(2) - curr_proj.row(1);
        A.row(5) = img1_kp_pts[i].x * curr_proj.row(1) - img1_kp_pts[i].y * curr_proj.row(0);

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector4d point_3d_homo = svd.matrixV().col(3);
        Eigen::Vector3d point_3d = point_3d_homo.head(3) / point_3d_homo[3];
        landmarks.push_back(point_3d);
        landmarks_4D.push_back(point_3d_homo);
    }
}

void triangulate2(cv::Mat intrinsic, std::vector<cv::Point2f> img0_kp_pts, std::vector<cv::Point2f> img1_kp_pts, Eigen::Isometry3d &cam1_pose, const cv::Mat &mask, std::vector<Eigen::Vector3d> &landmarks, std::vector<Eigen::Vector4d> &landmarks_4D) {
    std::cout << "----- triangulate2 -----" << std::endl;
    Eigen::Matrix3d camera_intrinsic;
    cv::cv2eigen(intrinsic, camera_intrinsic);
    Eigen::MatrixXd prev_proj = Eigen::MatrixXd::Identity(3, 4);
    Eigen::MatrixXd curr_proj = Eigen::MatrixXd::Identity(3, 4);

    prev_proj = camera_intrinsic * prev_proj;
    curr_proj = camera_intrinsic * curr_proj * cam1_pose.inverse().matrix();
    std::cout << "prev_proj: \n" << prev_proj << std::endl;
    std::cout << "curr_proj: \n" << curr_proj << std::endl;


    for (int i = 0; i < img0_kp_pts.size(); i++) {
        if (mask.at<unsigned char>(i) != 1) {
            landmarks.push_back(Eigen::Vector3d(0, 0, 0));
            continue;
        }
        // std::cout << "triangulate " << i << std::endl;

        Eigen::Matrix4d A;
        A.row(0) = img0_kp_pts[i].x * prev_proj.row(2) - prev_proj.row(0);
        A.row(1) = img0_kp_pts[i].y * prev_proj.row(2) - prev_proj.row(1);
        A.row(2) = img1_kp_pts[i].x * curr_proj.row(2) - curr_proj.row(0);
        A.row(3) = img1_kp_pts[i].y * curr_proj.row(2) - curr_proj.row(1);

        Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector4d point_3d_homo = svd.matrixV().col(3);
        Eigen::Vector3d point_3d = point_3d_homo.head(3) / point_3d_homo[3];
        landmarks.push_back(point_3d);
        landmarks_4D.push_back(point_3d_homo);
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
    curr_proj = camera_intrinsic * curr_proj * cam1_pose.inverse().matrix();

    for (int i = 0; i < img0_kp_pts.size(); i++) {
        bool is_3d_point = false;
        if (mask.at<unsigned char>(i) == 1) {
            is_3d_point = true;
        }

        cv::Point2f measurement_point0 = img0_kp_pts[i];
        cv::Point2f measurement_point1 = img1_kp_pts[i];

        if (is_3d_point) {
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

            cv::Point2f error_vector0 = projected_point0 - measurement_point0;
            cv::Point2f error_vector1 = projected_point1 - measurement_point1;
            error0 = sqrt(error_vector0.x * error_vector0.x + error_vector0.y * error_vector0.y);
            error1 = sqrt(error_vector1.x * error_vector1.x + error_vector1.y * error_vector1.y);

            reproj_error0 += error0;
            reproj_error1 += error1;

            inlier_reproj_error0 += error0;
            inlier_reproj_error1 += error1;
            inlier_cnt++;

            // mark reprojected landmarks
            cv::circle(image0_copy,
                    projected_point0,
                    2,
                    mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0));  // red, (blue)
            cv::line(image0_copy,
                    measurement_point0,
                    projected_point0,
                    mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0));  // red, (blue)
            cv::circle(image1_copy,
                    projected_point1,
                    2,
                    mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0));  // red, (blue)
            cv::line(image1_copy,
                    measurement_point1,
                    projected_point1,
                    mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0));  // red, (blue)
        }

        // draw keypoint box
        cv::rectangle(image0_copy,
                    measurement_point0 - cv::Point2f(5, 5),
                    measurement_point0 + cv::Point2f(5, 5),
                    mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 125, 255));  // green, (orange)
        cv::rectangle(image1_copy,
                    measurement_point1 - cv::Point2f(5, 5),
                    measurement_point1 + cv::Point2f(5, 5),
                    mask.at<unsigned char>(i) == 1 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 125, 255));  // green, (orange)
    }
    cv::putText(image0_copy, "frame" + std::to_string(p_id),
                                    cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(image0_copy, "#landmarks: " + std::to_string(landmarks.size()),
                                    cv::Point(0, 40), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(image0_copy, "#pose inliers: " + std::to_string(inlier_cnt),
                                    cv::Point(0, 60), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(image1_copy, "frame" + std::to_string(c_id),
                                    cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(image1_copy, "#landmarks: " + std::to_string(landmarks.size()),
                                    cv::Point(0, 40), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
    cv::putText(image1_copy, "#pose inliers: " + std::to_string(inlier_cnt),
                                    cv::Point(0, 60), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));

    cv::imwrite("vo_patch/reprojected_landmarks/frame" + std::to_string(p_id) + "_" + std::to_string(num_matches) + "_proj.png", image0_copy);
    cv::imwrite("vo_patch/reprojected_landmarks/frame" + std::to_string(c_id) + "_" + std::to_string(num_matches) + "_proj.png", image1_copy);

    cv::Mat image_concat;
    cv::vconcat(image0_copy, image1_copy, image_concat);
    cv::imwrite("vo_patch/reprojected_landmarks/frame" + std::to_string(p_id) + "&frame" + std::to_string(c_id) + "_" + std::to_string(num_matches) + "_proj.png", image_concat);

    std::cout << "pose inliers: " << inlier_cnt << std::endl;
    std::cout << "inlier reprojected error: " << ((inlier_reproj_error0 + inlier_reproj_error1) / inlier_cnt) / 2 << std::endl;

    double reprojection_error = ((reproj_error0 + reproj_error1) / landmarks.size()) / 2;
    return reprojection_error;
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
    pangolin::CreateWindowAndBind("Visual Odometry Patch2", 1024, 768);
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

void loadGT(std::string gt_path, int prev_frame_id, std::vector<Eigen::Isometry3d> &gt_poses) {
    std::ifstream gt_poses_file(gt_path);
    double time_stamp;
    double r11, r12, r13, r21, r22, r23, r31, r32, r33, t1, t2, t3;
    double qw, qx, qy, qz;

    std::string line;

    for (int l = 0; l < prev_frame_id; l++) {
    // for (int l = 0; l < 3; l++) {
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

        // // TUM format
        // ssline >> time_stamp
        //         >> t1 >> t2 >> t3
        //         >> qx >> qy >> qz >> qw;

        Eigen::Matrix3d rotation_mat;
        rotation_mat << r11, r12, r13,
                        r21, r22, r23,
                        r31, r32, r33;
        // Eigen::Quaterniond quaternion(qw, qx, qy, qz);
        // Eigen::Matrix3d rotation_mat(quaternion);

        Eigen::Vector3d translation_mat;
        translation_mat << t1, t2, t3;

        Eigen::Isometry3d gt_pose = Eigen::Isometry3d::Identity();
        gt_pose.linear() = rotation_mat;
        gt_pose.translation() = translation_mat;

        std::cout << "gt_pose: \n" << gt_pose.matrix() << std::endl;
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

void optimize(const cv::Mat &intrinsic,
                const std::vector<cv::Point2f> &prev_frame_kp_pts, Eigen::Isometry3d prev_pose,
                const std::vector<cv::Point2f> &curr_frame_kp_pts, Eigen::Isometry3d &curr_pose,
                std::vector<Eigen::Vector3d> &landmarks, const cv::Mat &mask, bool verbose) {
    // create a graph
    gtsam::NonlinearFactorGraph graph;

    // stereo camera calibration object
    gtsam::Cal3_S2::shared_ptr K(
        new gtsam::Cal3_S2(intrinsic.at<double>(0, 0),      // fx
                                intrinsic.at<double>(1, 1), // fy
                                intrinsic.at<double>(0, 1), // s
                                intrinsic.at<double>(0, 2), // cx
                                intrinsic.at<double>(1, 2)));  // cy
    std::cout << K << std::endl;

    // create initial values
    gtsam::Values initial_estimates;

    // 1. Add Values and Factors
    const auto measurement_noise = gtsam::noiseModel::Isotropic::Sigma(2, 1.0);  // std of 1px.

    // insert values and factors of the frames
    int landmark_idx = 0;
    int prev_frame_idx = 0;
    int curr_frame_idx = 1;

    // insert initial value of the frame pose
    gtsam::Pose3 prev_pose_gtsam = gtsam::Pose3(gtsam::Rot3(prev_pose.rotation()), gtsam::Point3(prev_pose.translation()));
    gtsam::Pose3 curr_pose_gtsam = gtsam::Pose3(gtsam::Rot3(curr_pose.rotation()), gtsam::Point3(curr_pose.translation()));
    initial_estimates.insert(gtsam::Symbol('x', prev_frame_idx), prev_pose_gtsam);
    initial_estimates.insert(gtsam::Symbol('x', curr_frame_idx), curr_pose_gtsam);

    for (int i = 0; i < mask.rows; i++) {
        if (mask.at<unsigned char>(i) != 1) {
            continue;
        }

        // insert initial value of the landmark
        auto landmark = landmarks[landmark_idx];
        initial_estimates.insert<gtsam::Point3>(gtsam::Symbol('l', landmark_idx), gtsam::Point3(landmark));

        // 2D measurement
        cv::Point2f prev_measurement_cv = prev_frame_kp_pts[landmark_idx];
        cv::Point2f curr_measurement_cv = curr_frame_kp_pts[landmark_idx];
        gtsam::Point2 prev_measurement(prev_measurement_cv.x, prev_measurement_cv.y);
        gtsam::Point2 curr_measurement(curr_measurement_cv.x, curr_measurement_cv.y);
        // add measurement factor
        graph.emplace_shared<gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>>(prev_measurement, measurement_noise,
                                                                                                        gtsam::Symbol('x', prev_frame_idx), gtsam::Symbol('l', landmark_idx),
                                                                                                        K);
        graph.emplace_shared<gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>>(curr_measurement, measurement_noise,
                                                                                                        gtsam::Symbol('x', curr_frame_idx), gtsam::Symbol('l', landmark_idx),
                                                                                                        K);

        landmark_idx++;
    }


    // 2. prior factors
    gtsam::Pose3 first_pose = gtsam::Pose3(gtsam::Rot3(prev_pose.rotation()), gtsam::Point3(prev_pose.translation()));
    // gtsam::Pose3 second_pose = gtsam::Pose3(gtsam::Rot3(frames[1]->pose_.rotation()), gtsam::Point3(frames[1]->pose_.translation()));
    // add a prior on the first pose
    const auto pose_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << gtsam::Vector3::Constant(0.3), gtsam::Vector3::Constant(0.1)).finished());  // std of 0.3m for x,y,z, 0.1rad for r,p,y
    graph.addPrior(gtsam::Symbol('x', 0), prev_pose_gtsam, pose_noise);
    // // add a prior on the second pose (for scale)
    // graph.addPrior(gtsam::Symbol('x', 1), curr_pose_gtsam, pose_noise);
    // add a prior on the first landmark (for scale)
    auto point_noise = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);
    graph.addPrior(gtsam::Symbol('l', 0), landmarks[0], point_noise);

    // 3. Optimize
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial_estimates);
    // start timer [optimization]
    const std::chrono::time_point<std::chrono::steady_clock> optimization_start = std::chrono::steady_clock::now();
    gtsam::Values result = optimizer.optimize();
    // end timer [optimization]
    const std::chrono::time_point<std::chrono::steady_clock> optimization_end = std::chrono::steady_clock::now();
    auto optimization_diff = optimization_end - optimization_start;
    std::cout << "GTSAM Optimization elapsed time: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(optimization_diff).count() << "[ms]" << std::endl;

    std::cout << "initial error = " << graph.error(initial_estimates) << std::endl;
    std::cout << "final error = " << graph.error(result) << std::endl;


    // 4. Recover result pose
    prev_pose = result.at<gtsam::Pose3>(gtsam::Symbol('x', prev_frame_idx)).matrix();
    curr_pose = result.at<gtsam::Pose3>(gtsam::Symbol('x', curr_frame_idx)).matrix();

    // Recover Landmark point
    landmark_idx = 0;
    for (int j = 0; j < mask.rows; j++) {
        if (mask.at<unsigned char>(j) != 1) {
            continue;
        }
        landmarks[landmark_idx] = result.at<gtsam::Point3>(gtsam::Symbol('l', landmark_idx));

        landmark_idx++;
    }
    if (verbose) {
        graph.print("graph print:\n");
        result.print("optimization result:\n");
    }
}


int countMask(const cv::Mat &mask) {
    int count = 0;
    for (int i = 0; i < mask.rows; i++) {
        if (mask.at<unsigned char>(i) == 1) {
            count++;
        }
    }
    return count;
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


void decomposeEssentialMat(const cv::Mat &essential_mat, cv::Mat &R1, cv::Mat &R2, cv::Mat &t) {
    cv::Mat U, S, VT;
    cv::SVD::compute(essential_mat, S, U, VT);

    // // OpenCV decomposeEssentialMat()
    // cv::Mat W = (cv::Mat_<double>(3, 3) << 0, 1, 0,
    //                                         -1, 0, 0,
    //                                         0, 0, 1);

    // hand function.
    cv::Mat W = (cv::Mat_<double>(3, 3) << 0, -1, 0,
                                            1, 0, 0,
                                            0, 0, 1);


    // double det_U = cv::determinant(U);
    // double det_W = cv::determinant(W);
    // double det_VT = cv::determinant(VT);
    // std::cout << "det U: " << det_U << std::endl;
    // std::cout << "det W: " << det_W << std::endl;
    // std::cout << "det VT: " << det_VT << std::endl;

    R1 = U * W * VT;
    R2 = U * W.t() * VT;
    t = U.col(2);

    double det_R1 = cv::determinant(R1);
    double det_R2 = cv::determinant(R2);
    // std::cout << "det R1: " << det_R1 << std::endl;
    // std::cout << "det R2: " << det_R2 << std::endl;

    if (det_R1 < 0) {
        R1 = -R1;
    }
    if (det_R1 < 0) {
        R2 = -R2;
    }
    // std::cout << "R1:\n" << R1 << std::endl;
    // std::cout << "R2:\n" << R2 << std::endl;
    // std::cout << "t:\n" << t << std::endl;

    // std::vector<cv::Mat> pose_candidates(4);
    // cv::hconcat(std::vector<cv::Mat>{R1, t}, pose_candidates[0]);
    // cv::hconcat(std::vector<cv::Mat>{R1, -t}, pose_candidates[1]);
    // cv::hconcat(std::vector<cv::Mat>{R2, t}, pose_candidates[2]);
    // cv::hconcat(std::vector<cv::Mat>{R2, -t}, pose_candidates[3]);

    // for (int i = 0; i < pose_candidates.size(); i++) {
    //     std::cout << "pose_candidate [" << i << "]: \n" << pose_candidates[i] << std::endl;
    // }
}

int chiralityCheck(const cv::Mat &intrinsic,
                    const std::vector<cv::Point2f> &image0_kp_pts,
                    const std::vector<cv::Point2f> &image1_kp_pts,
                    const Eigen::Isometry3d &cam1_pose,
                    cv::Mat &mask) {
    Eigen::Matrix3d intrinsic_eigen;
    cv::cv2eigen(intrinsic, intrinsic_eigen);
    Eigen::MatrixXd prev_proj = Eigen::MatrixXd::Identity(3, 4);
    Eigen::MatrixXd curr_proj = Eigen::MatrixXd::Identity(3, 4);

    prev_proj = intrinsic_eigen * prev_proj;
    curr_proj = intrinsic_eigen * curr_proj * cam1_pose.inverse().matrix();

    int positive_cnt = 0;

    for (int i = 0; i < image0_kp_pts.size(); i++) {
        if (mask.at<unsigned char>(i) != 1) {
            continue;
        }

        Eigen::Matrix4d A;
        A.row(0) = image0_kp_pts[i].x * prev_proj.row(2) - prev_proj.row(0);
        A.row(1) = image0_kp_pts[i].y * prev_proj.row(2) - prev_proj.row(1);
        A.row(2) = image1_kp_pts[i].x * curr_proj.row(2) - curr_proj.row(0);
        A.row(3) = image1_kp_pts[i].y * curr_proj.row(2) - curr_proj.row(1);

        Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector4d point_3d_homo = svd.matrixV().col(3);
        Eigen::Vector3d point_3d = point_3d_homo.head(3) / point_3d_homo[3];
        // landmarks.push_back(point_3d);

        Eigen::Vector4d cam1_point_3d_homo = cam1_pose.inverse() * point_3d_homo;
        Eigen::Vector3d cam1_point_3d = cam1_point_3d_homo.head(3) / cam1_point_3d_homo[3];
        // std::cout << "landmark(world) z: " << point_3d.z() << ", (camera) z: " << cam1_point_3d.z() << std::endl;
        if (point_3d.z() > 0 && cam1_point_3d.z() > 0 && point_3d.z() < 70) {
            mask.at<unsigned char>(i) = 1;
            positive_cnt++;
        }
        else {
            mask.at<unsigned char>(i) = 0;
        }
    }

    return positive_cnt;
}

void recoverPose(const cv::Mat &intrinsic,
                        const cv::Mat &essential_mat,
                        const std::vector<cv::Point2f> &image0_kp_pts,
                        const std::vector<cv::Point2f> &image1_kp_pts,
                        Eigen::Isometry3d &relative_pose,
                        cv::Mat &mask) {

    // Decompose essential matrix
    cv::Mat R1, R2, t;
    // cv::decomposeEssentialMat(essential_mat, R1, R2, t);
    // std::cout << "cv_R1:\n" << R1 << std::endl;
    // std::cout << "cv_R2:\n" << R2 << std::endl;
    // std::cout << "cv_t:\n" << t << std::endl;
    decomposeEssentialMat(essential_mat, R1, R2, t);

    Eigen::Matrix3d rotation_mat;
    Eigen::Vector3d translation_mat;

    std::vector<Eigen::Isometry3d> rel_pose_candidates(4, Eigen::Isometry3d::Identity());
    std::vector<cv::Mat> masks(4);

    for (int  i = 0; i < 4; i++) {
        masks[i] = mask.clone();
    }

    int valid_point_cnts[4];
    for (int k = 0; k < 4; k++) {
        if (k == 0) {
            cv::cv2eigen(R1, rotation_mat);
            cv::cv2eigen(t, translation_mat);
        }
        else if (k == 1) {
            cv::cv2eigen(R1, rotation_mat);
            cv::cv2eigen(-t, translation_mat);
        }
        else if (k == 2) {
            cv::cv2eigen(R2, rotation_mat);
            cv::cv2eigen(t, translation_mat);
        }
        else if (k == 3) {
            cv::cv2eigen(R2, rotation_mat);
            cv::cv2eigen(-t, translation_mat);
        }
        // rotation_mat << 0.999992,  0.00248744,  0.00303404,
        //                 -0.00249412,    0.999994,  0.00219979,
        //                 -0.00302855, -0.00220734,    0.999993;
        // translation_mat << 0.00325004, 0.00744242, -0.999967;

        rel_pose_candidates[k].linear() = rotation_mat;
        rel_pose_candidates[k].translation() = translation_mat;
        // rel_pose_candidates[k].linear() = rotation_mat.transpose();
        // rel_pose_candidates[k].translation() = - rotation_mat.transpose() * translation_mat;

        std::cout << "pose candidate[" << k << "]:\n" << rel_pose_candidates[k].matrix() << std::endl;

        valid_point_cnts[k] = chiralityCheck(intrinsic, image0_kp_pts, image1_kp_pts, rel_pose_candidates[k], masks[k]);
    }

    int max_cnt = 0, max_idx = 0;
    for (int k = 0; k < 4; k++) {
        std::cout << "cnt[" << k << "]: " << valid_point_cnts[k] << std::endl;
        if (valid_point_cnts[k] > max_cnt) {
            max_cnt = valid_point_cnts[k];
            max_idx = k;
        }
    }
    // std::cout << "max idx: " << max_idx << std::endl;
    relative_pose = rel_pose_candidates[max_idx];
    mask = masks[max_idx];
    std::cout << "mask count: " << countMask(mask) << std::endl;
    // std::cout << "relative pose:\n " << relative_pose.matrix() << std::endl;
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
    cv::Mat prev_image;
    cv::undistort(prev_image_dist, prev_image, intrinsic, distortion);
    std::string prev_id = std::string(config_file["prev_frame"]);
    int p_id = std::stoi(prev_id.substr(prev_id.length() - 10, 6));
    poses.push_back(Eigen::Isometry3d::Identity());

    // new Frame!
    cv::Mat curr_image_dist = cv::imread(config_file["curr_frame"], cv::IMREAD_GRAYSCALE);
    cv::Mat curr_image;
    cv::undistort(curr_image_dist, curr_image, intrinsic, distortion);
    std::string curr_id = std::string(config_file["curr_frame"]);
    int c_id = std::stoi(curr_id.substr(curr_id.length() - 10, 6));

    //**========== 2. Feature matching ==========**//





    // frame0
    // std::vector<cv::Point2f> frame0_kp_pts_test = {cv::Point2f(280, 148),cv::Point2f(344, 205), cv::Point2f(440, 203), cv::Point2f(487, 191), cv::Point2f(640, 123), cv::Point2f(808, 126), cv::Point2f(917, 111), cv::Point2f(908, 295), cv::Point2f(771, 157), cv::Point2f(870, 155)};
    // std::vector<cv::Point2f> frame0_kp_pts_test = {cv::Point2f(89, 92), cv::Point2f(280, 148), cv::Point2f(437, 204), cv::Point2f(434, 161), cv::Point2f(495, 188), cv::Point2f(660, 150), cv::Point2f(709, 136), cv::Point2f(808, 127), cv::Point2f(913, 110), cv::Point2f(899, 294)};
    // subpixel, 기본 10개
    std::vector<cv::Point2f> frame0_kp_pts_test = {cv::Point2f(89.0, 92.0), cv::Point2f(279.65207, 147.67671), cv::Point2f(440.3918, 203.43727), cv::Point2f(434.0, 161.0), cv::Point2f(486.80063, 190.52914), cv::Point2f(661.3766, 148.82133), cv::Point2f(709.0, 136.0), cv::Point2f(807.58307, 125.778046), cv::Point2f(916.523, 110.59789), cv::Point2f(907.6637, 294.75223)};
    // subpixel, 22개
    // std::vector<cv::Point2f> frame0_kp_pts_test = {cv::Point2f(279.65207, 147.67671), cv::Point2f(280.12967, 183.69873), cv::Point2f(352.09125, 148.84175), cv::Point2f(344.0637, 204.65979), cv::Point2f(315.4563, 222.74753), cv::Point2f(391.87952, 216.45517), cv::Point2f(486.80063, 190.52913), cv::Point2f(436.20886, 248.59961), cv::Point2f(499.10287, 263.28366), cv::Point2f(550.5249, 253.72993), cv::Point2f(658.0655, 149.22586), cv::Point2f(654.8235, 172.63608), cv::Point2f(708.209, 155.30844), cv::Point2f(725.3333, 168.00961), cv::Point2f(771.3757, 157.38417), cv::Point2f(813.2508, 133.72275), cv::Point2f(807.58307, 125.77807), cv::Point2f(870.40405, 154.61826), cv::Point2f(900.5791, 129.21245), cv::Point2f(916.52295, 110.59792), cv::Point2f(936.54004, 126.18597), cv::Point2f(916.33746, 243.73369)};
    // std::vector<cv::Point2f> frame0_kp_pts_test = {cv::Point2f(280, 148), cv::Point2f(280, 184), cv::Point2f(352, 149), cv::Point2f(344, 205), cv::Point2f(315, 223), cv::Point2f(392, 216), cv::Point2f(487, 191), cv::Point2f(436, 249), cv::Point2f(499, 263), cv::Point2f(551, 254), cv::Point2f(658, 149), cv::Point2f(655, 173), cv::Point2f(708, 155), cv::Point2f(725, 168), cv::Point2f(771, 157), cv::Point2f(813, 134), cv::Point2f(808, 126), cv::Point2f(870, 155), cv::Point2f(901, 129), cv::Point2f(917, 111), cv::Point2f(937, 126), cv::Point2f(916, 244)};

    // tum checkerboard, 63개 (.409720.png)
    // std::vector<cv::Point2f> frame0_kp_pts_test = {cv::Point2f(185.98361, 255.33424), cv::Point2f(199.0621, 257.06512), cv::Point2f(211.93787, 258.88757), cv::Point2f(225.30211, 260.8278), cv::Point2f(238.0471, 262.44495), cv::Point2f(251.32112, 264.19113), cv::Point2f(263.93756, 265.82367), cv::Point2f(277.21017, 267.4266), cv::Point2f(289.7167, 269.32452), cv::Point2f(183.60417, 268.46786), cv::Point2f(196.75133, 270.04303), cv::Point2f(210.08305, 271.65692), cv::Point2f(223.33095, 273.50418), cv::Point2f(236.30656, 275.05783), cv::Point2f(249.50627, 276.86057), cv::Point2f(262.3463, 278.86392), cv::Point2f(275.50507, 280.47092), cv::Point2f(288.29028, 282.06607), cv::Point2f(181.45612, 281.2323), cv::Point2f(194.626, 282.87787), cv::Point2f(207.89014, 284.67996), cv::Point2f(221.0085, 286.65448), cv::Point2f(233.89561, 288.1843), cv::Point2f(247.44759, 289.8412), cv::Point2f(260.40274, 291.54437), cv::Point2f(273.57608, 293.13394), cv::Point2f(286.5793, 294.89896), cv::Point2f(178.94438, 294.63147), cv::Point2f(192.10973, 296.19675), cv::Point2f(205.64519, 297.79102), cv::Point2f(218.89854, 299.5839), cv::Point2f(232.19884, 301.0378), cv::Point2f(245.59622, 302.8427), cv::Point2f(258.5278, 304.77774), cv::Point2f(271.6318, 306.38046), cv::Point2f(284.8339, 308.03354), cv::Point2f(176.39893, 307.6902), cv::Point2f(190.0887, 309.26962), cv::Point2f(203.64032, 311.12424), cv::Point2f(216.75804, 312.74728), cv::Point2f(229.92258, 314.5272), cv::Point2f(243.43706, 316.25223), cv::Point2f(256.5954, 317.77914), cv::Point2f(269.90576, 319.37866), cv::Point2f(283.29272, 321.0969), cv::Point2f(174.2677, 320.8919), cv::Point2f(187.97784, 322.7443), cv::Point2f(201.19373, 324.6455), cv::Point2f(214.4658, 326.1613), cv::Point2f(228.03183, 327.8519), cv::Point2f(241.62836, 329.47943), cv::Point2f(254.76276, 330.86276), cv::Point2f(267.80664, 332.7646), cv::Point2f(281.32922, 334.5889), cv::Point2f(172.07285, 334.64755), cv::Point2f(185.34409, 336.56198), cv::Point2f(198.8366, 338.05148), cv::Point2f(212.45972, 339.69293), cv::Point2f(225.9317, 341.32623), cv::Point2f(239.38484, 342.8262), cv::Point2f(252.78525, 344.6163), cv::Point2f(266.17078, 346.4669), cv::Point2f(279.53818, 347.9511)};
    // std::vector<cv::Point2f> frame0_kp_pts_test = {cv::Point2f(186, 255), cv::Point2f(199, 257), cv::Point2f(212, 259), cv::Point2f(225, 261), cv::Point2f(238, 262), cv::Point2f(251, 264), cv::Point2f(264, 266), cv::Point2f(277, 267), cv::Point2f(290, 269), cv::Point2f(184, 268), cv::Point2f(197, 270), cv::Point2f(210, 272), cv::Point2f(223, 274), cv::Point2f(236, 275), cv::Point2f(250, 277), cv::Point2f(262, 279), cv::Point2f(276, 280), cv::Point2f(288, 282), cv::Point2f(181, 281), cv::Point2f(195, 283), cv::Point2f(208, 285), cv::Point2f(221, 287), cv::Point2f(234, 288), cv::Point2f(247, 290), cv::Point2f(260, 292), cv::Point2f(274, 293), cv::Point2f(287, 295), cv::Point2f(179, 295), cv::Point2f(192, 296), cv::Point2f(206, 298), cv::Point2f(219, 300), cv::Point2f(232, 301), cv::Point2f(246, 303), cv::Point2f(259, 305), cv::Point2f(272, 306), cv::Point2f(285, 308), cv::Point2f(176, 308), cv::Point2f(190, 309), cv::Point2f(204, 311), cv::Point2f(217, 313), cv::Point2f(230, 315), cv::Point2f(243, 316), cv::Point2f(257, 318), cv::Point2f(270, 319), cv::Point2f(283, 321), cv::Point2f(174, 321), cv::Point2f(188, 323), cv::Point2f(201, 325), cv::Point2f(214, 326), cv::Point2f(228, 328), cv::Point2f(242, 329), cv::Point2f(255, 331), cv::Point2f(268, 333), cv::Point2f(281, 335), cv::Point2f(172, 335), cv::Point2f(185, 337), cv::Point2f(199, 338), cv::Point2f(212, 340), cv::Point2f(226, 341), cv::Point2f(239, 343), cv::Point2f(253, 345), cv::Point2f(266, 346), cv::Point2f(280, 348)};

    // frame1
    // std::vector<cv::Point2f> frame1_kp_pts_test = {cv::Point2f(275, 150),cv::Point2f(340, 208), cv::Point2f(439, 206), cv::Point2f(488, 193), cv::Point2f(642, 124), cv::Point2f(816, 125), cv::Point2f(929, 109), cv::Point2f(943, 308), cv::Point2f(779, 158), cv::Point2f(881, 155)};
    // std::vector<cv::Point2f> frame1_kp_pts_test = {cv::Point2f(77, 92), cv::Point2f(275, 149), cv::Point2f(435, 207), cv::Point2f(435, 163), cv::Point2f(496, 190), cv::Point2f(664, 150), cv::Point2f(714, 136), cv::Point2f(817, 126), cv::Point2f(925, 109), cv::Point2f(933, 307)};
    // subpixel, 기본 10개
    std::vector<cv::Point2f> frame1_kp_pts_test = {cv::Point2f(76.435524, 92.402176), cv::Point2f(274.63477, 149.50064), cv::Point2f(439.3174, 206.34424), cv::Point2f(435.0, 163.0), cv::Point2f(487.71252, 192.5296), cv::Point2f(664.4662, 149.73691), cv::Point2f(714.8803, 135.63571), cv::Point2f(816.14087, 125.12716), cv::Point2f(928.7381, 109.08112), cv::Point2f(942.7255, 307.60168)};
    // subpixel, 22개.
    // std::vector<cv::Point2f> frame1_kp_pts_test = {cv::Point2f(274.63458, 149.50125), cv::Point2f(275.0794, 186.3648), cv::Point2f(349.50986, 150.97981), cv::Point2f(339.62225, 207.53798), cv::Point2f(309.6779, 226.33817), cv::Point2f(387.3351, 218.88358), cv::Point2f(487.71255, 192.52972), cv::Point2f(433.01346, 253.78752), cv::Point2f(496.2911, 269.09958), cv::Point2f(550.5292, 258.7385), cv::Point2f(661.39496, 150.06165), cv::Point2f(657.99445, 173.96358), cv::Point2f(713.2638, 154.84555), cv::Point2f(730.56287, 168.4511), cv::Point2f(778.8773, 157.77487), cv::Point2f(821.83105, 133.28596), cv::Point2f(816.1409, 125.127235), cv::Point2f(881.11017, 154.5988), cv::Point2f(912.2954, 128.29897), cv::Point2f(928.7381, 109.08129), cv::Point2f(949.4896, 125.19382), cv::Point2f(950.6367, 250.94124)};
    // std::vector<cv::Point2f> frame1_kp_pts_test = {cv::Point2f(275, 150), cv::Point2f(275, 186), cv::Point2f(350, 151), cv::Point2f(340, 208), cv::Point2f(310, 226), cv::Point2f(387, 219), cv::Point2f(488, 193), cv::Point2f(433, 254), cv::Point2f(496, 269), cv::Point2f(551, 259), cv::Point2f(661, 150), cv::Point2f(658, 174), cv::Point2f(713, 155), cv::Point2f(731, 168), cv::Point2f(779, 158), cv::Point2f(822, 133), cv::Point2f(816, 125), cv::Point2f(881, 155), cv::Point2f(912, 128), cv::Point2f(929, 109), cv::Point2f(949, 125), cv::Point2f(951, 251)};

    // tum checkerboard, 63개 (.445969.png)
    // std::vector<cv::Point2f> frame1_kp_pts_test = {cv::Point2f(188.50336, 254.28859), cv::Point2f(201.61862, 256.00037), cv::Point2f(214.82784, 257.61743), cv::Point2f(227.99135, 259.2137), cv::Point2f(240.9269, 260.97116), cv::Point2f(253.95947, 262.65988), cv::Point2f(266.96304, 264.2749), cv::Point2f(279.88977, 265.71893), cv::Point2f(292.7736, 267.5034), cv::Point2f(186.67285, 266.95377), cv::Point2f(199.6509, 268.7468), cv::Point2f(212.82109, 270.5306), cv::Point2f(225.98372, 272.1499), cv::Point2f(239.08856, 273.56796), cv::Point2f(252.16556, 275.35953), cv::Point2f(265.0837, 277.1004), cv::Point2f(278.1214, 278.60114), cv::Point2f(291.15082, 280.2634), cv::Point2f(184.32774, 280.10443), cv::Point2f(197.61137, 281.6256), cv::Point2f(210.93588, 283.37775), cv::Point2f(224.28981, 284.8726), cv::Point2f(237.33499, 286.56235), cv::Point2f(250.51984, 288.32553), cv::Point2f(263.5544, 289.79327), cv::Point2f(276.68262, 291.35413), cv::Point2f(289.65073, 293.04395), cv::Point2f(182.41861, 292.95807), cv::Point2f(195.58974, 294.77478), cv::Point2f(208.92451, 296.49942), cv::Point2f(222.20618, 298.08133), cv::Point2f(235.44948, 299.5277), cv::Point2f(248.69556, 301.25595), cv::Point2f(261.8787, 302.814), cv::Point2f(275.05862, 304.41858), cv::Point2f(288.23535, 306.05096), cv::Point2f(180.09937, 306.39767), cv::Point2f(193.39595, 308.14496), cv::Point2f(206.92966, 309.59625), cv::Point2f(220.48015, 311.04932), cv::Point2f(233.58061, 312.8338), cv::Point2f(246.79042, 314.53445), cv::Point2f(260.04233, 316.07474), cv::Point2f(273.36606, 317.4822), cv::Point2f(286.52466, 319.16614), cv::Point2f(177.803, 319.75412), cv::Point2f(191.4469, 321.35947), cv::Point2f(204.9484, 322.82416), cv::Point2f(218.24597, 324.51074), cv::Point2f(231.55283, 326.2864), cv::Point2f(245.17305, 327.6438), cv::Point2f(258.54037, 329.15945), cv::Point2f(271.64615, 330.8213), cv::Point2f(284.77243, 332.54663), cv::Point2f(175.71579, 333.29483), cv::Point2f(189.30305, 334.76346), cv::Point2f(202.7548, 336.44424), cv::Point2f(216.32068, 338.13422), cv::Point2f(229.76256, 339.5763), cv::Point2f(243.20312, 341.05753), cv::Point2f(256.59732, 342.72125), cv::Point2f(269.98282, 344.40897), cv::Point2f(283.34937, 345.8332)};
    // std::vector<cv::Point2f> frame1_kp_pts_test = {cv::Point2f(189, 254), cv::Point2f(202, 256), cv::Point2f(215, 258), cv::Point2f(228, 259), cv::Point2f(241, 261), cv::Point2f(254, 263), cv::Point2f(267, 264), cv::Point2f(280, 266), cv::Point2f(293, 268), cv::Point2f(187, 267), cv::Point2f(200, 269), cv::Point2f(213, 271), cv::Point2f(226, 272), cv::Point2f(239, 274), cv::Point2f(252, 275), cv::Point2f(265, 277), cv::Point2f(278, 279), cv::Point2f(291, 280), cv::Point2f(184, 280), cv::Point2f(198, 282), cv::Point2f(211, 283), cv::Point2f(224, 285), cv::Point2f(237, 287), cv::Point2f(251, 288), cv::Point2f(264, 290), cv::Point2f(277, 291), cv::Point2f(290, 293), cv::Point2f(182, 293), cv::Point2f(196, 295), cv::Point2f(209, 296), cv::Point2f(222, 298), cv::Point2f(235, 300), cv::Point2f(249, 301), cv::Point2f(262, 303), cv::Point2f(275, 304), cv::Point2f(288, 306), cv::Point2f(180, 306), cv::Point2f(193, 308), cv::Point2f(207, 310), cv::Point2f(220, 311), cv::Point2f(234, 313), cv::Point2f(247, 315), cv::Point2f(260, 316), cv::Point2f(273, 317), cv::Point2f(287, 319), cv::Point2f(178, 320), cv::Point2f(191, 321), cv::Point2f(205, 323), cv::Point2f(218, 325), cv::Point2f(232, 326), cv::Point2f(245, 328), cv::Point2f(259, 329), cv::Point2f(272, 331), cv::Point2f(285, 333), cv::Point2f(176, 333), cv::Point2f(189, 335), cv::Point2f(203, 336), cv::Point2f(216, 338), cv::Point2f(230, 340), cv::Point2f(243, 341), cv::Point2f(257, 343), cv::Point2f(270, 344), cv::Point2f(283, 346)};

    // frame2
    std::vector<cv::Point2f> frame2_kp_pts_test = {cv::Point2f(65, 90), cv::Point2f(270, 149), cv::Point2f(434, 208), cv::Point2f(436, 163), cv::Point2f(498, 191), cv::Point2f(668, 151), cv::Point2f(720, 136), cv::Point2f(827, 125), cv::Point2f(939, 108), cv::Point2f(977, 324)};
    // frame3
    std::vector<cv::Point2f> frame3_kp_pts_test = {cv::Point2f(51, 88), cv::Point2f(265, 149), cv::Point2f(432, 210), cv::Point2f(437, 164), cv::Point2f(500, 193), cv::Point2f(672, 151), cv::Point2f(726, 136), cv::Point2f(838, 125), cv::Point2f(954, 106), cv::Point2f(1035, 346)};
    // frame4
    std::vector<cv::Point2f> frame4_kp_pts_test = {cv::Point2f(37, 85), cv::Point2f(260, 148), cv::Point2f(431, 211), cv::Point2f(439, 164), cv::Point2f(502, 193), cv::Point2f(677, 151), cv::Point2f(733, 136), cv::Point2f(851, 123), cv::Point2f(972, 104),    cv::Point2f(653, 124)};

    std::vector<std::vector<cv::Point2f>> matches_vec = {frame0_kp_pts_test, frame1_kp_pts_test, frame2_kp_pts_test, frame3_kp_pts_test, frame4_kp_pts_test};

    // std::vector<cv::Point2f> prev_frame_kp_pts = {cv::Point2f(51, 88), cv::Point2f(265, 149), cv::Point2f(432, 210), cv::Point2f(437, 164), cv::Point2f(500, 193), cv::Point2f(672, 151), cv::Point2f(726, 136), cv::Point2f(838, 125), cv::Point2f(954, 106)};
    // std::vector<cv::Point2f> curr_frame_kp_pts = {cv::Point2f(37, 85), cv::Point2f(260, 148), cv::Point2f(431, 211), cv::Point2f(439, 164), cv::Point2f(502, 193), cv::Point2f(677, 151), cv::Point2f(733, 136), cv::Point2f(851, 123), cv::Point2f(972, 104)};
    int prev_frame_idx = config_file["prev_frame_idx"];
    std::vector<cv::Point2f> prev_frame_kp_pts = matches_vec[prev_frame_idx];
    std::vector<cv::Point2f> curr_frame_kp_pts = matches_vec[prev_frame_idx + 1];

    // std::vector<cv::Point2f> prev_frame_kp_pts_temp;
    // std::vector<cv::Point2f> curr_frame_kp_pts_temp;
    // for (int i = 0; i < prev_frame_kp_pts.size(); i++) {
    //     if (i % 2 == 0) {
    //         prev_frame_kp_pts_temp.push_back(prev_frame_kp_pts[i]);
    //         curr_frame_kp_pts_temp.push_back(curr_frame_kp_pts[i]);
    //     }
    // }
    // prev_frame_kp_pts = prev_frame_kp_pts_temp;
    // curr_frame_kp_pts = curr_frame_kp_pts_temp;
    cv::Mat mask;
    cv::Mat essential_mat = cv::findEssentialMat(prev_frame_kp_pts, curr_frame_kp_pts, intrinsic, cv::RANSAC, 0.999, 1.0, mask);
    std::cout << "essential matrix:\n" << essential_mat << std::endl;

    int essential_inlier = countMask(mask);
    std::cout << "essential inlier: " << essential_inlier << std::endl;

    // draw keypoints
    drawKeypoints(p_id, prev_image, prev_frame_kp_pts, c_id, curr_image, curr_frame_kp_pts, mask, 10);
    // drawKeypoints(p_id, prev_image, image0_kp_pts, c_id, curr_image, image1_kp_pts, num_matches);


    //**========== 3. Motion estimation ==========**//
    cv::Mat R, t;
    cv::recoverPose(essential_mat, prev_frame_kp_pts, curr_frame_kp_pts, intrinsic, R, t, mask);
    // recoverPose(essential_mat, curr_frame_kp_pts, prev_frame_kp_pts, intrinsic, R, t, mask);

    int pose_inlier = countMask(mask);
    std::cout << "pose inlier: " << pose_inlier << std::endl;

    Eigen::Matrix3d rotation_mat;
    Eigen::Vector3d translation_mat;
    cv::cv2eigen(R, rotation_mat);
    cv::cv2eigen(t, translation_mat);

    relative_pose.linear() = rotation_mat.transpose();
    relative_pose.translation() = - rotation_mat.transpose() * translation_mat;
    poses.push_back(relative_pose);
    std::cout << "relative pose:\n" << relative_pose.matrix() << std::endl;


    //**========== 4. Triangulation ==========**//
    std::vector<Eigen::Vector3d> landmarks0, landmarks, landmarks_cv;
    std::vector<Eigen::Vector4d> landmarks0_4D, landmarks_4D, landmarks_cv_4D;
    // triangulate(intrinsic, prev_frame_kp_pts, curr_frame_kp_pts, relative_pose, landmarks0, landmarks0_4D);
    triangulate2(intrinsic, prev_frame_kp_pts, curr_frame_kp_pts, relative_pose, mask, landmarks, landmarks_4D);
    // triangulate2(intrinsic, frame1_kp_pts_test, frame2_kp_pts_test, relative_pose, landmarks);
    // triangulate2(intrinsic, image0_kp_pts, image1_kp_pts, relative_pose, landmarks);
    // for (int i = 0; i < landmarks.size(); i++) {
    //     std::cout << "landmark[" << i << "]" << " " << landmarks[i] << std::endl;
    // }


    // opencv triangulation
    cv::Mat points4D;
    cv::Mat prev_proj_mat, curr_proj_mat;
    cv::Mat prev_pose, curr_pose;
    Eigen::MatrixXd pose_temp;
    pose_temp = Eigen::MatrixXd::Identity(3, 4);
    cv::eigen2cv(pose_temp, prev_pose);
    pose_temp = relative_pose.inverse().matrix().block<3, 4>(0, 0);
    cv::eigen2cv(pose_temp, curr_pose);
    prev_proj_mat = intrinsic * prev_pose;
    curr_proj_mat = intrinsic * curr_pose;
    std::cout << "prev_proj: \n" << prev_proj_mat << std::endl;
    std::cout << "curr_proj: \n" << curr_proj_mat << std::endl;

    cv::triangulatePoints(prev_proj_mat, curr_proj_mat, prev_frame_kp_pts, curr_frame_kp_pts, points4D);

    // homogeneous -> 3D
    for (int i = 0; i < points4D.cols; i++) {
        cv::Mat point_homo = points4D.col(i);
        Eigen::Vector3d point_3d(point_homo.at<float>(0) / point_homo.at<float>(3),
                                point_homo.at<float>(1) / point_homo.at<float>(3),
                                point_homo.at<float>(2) / point_homo.at<float>(3));
        Eigen::Vector4d point_3d_homo(point_homo.at<float>(0), point_homo.at<float>(1), point_homo.at<float>(2), point_homo.at<float>(3));
        landmarks_cv.push_back(point_3d);
        landmarks_cv_4D.push_back(point_3d_homo);
    }

    for (int i = 0; i < landmarks_cv.size(); i++) {
        std::cout << "cv::Point3d(" << landmarks_cv[i].transpose() << ")" << std::endl;
    }


    // calculate reprojection error & save the images
    double reproj_error = calcReprojectionError(intrinsic, p_id, prev_image, prev_frame_kp_pts, c_id, curr_image, curr_frame_kp_pts, mask, relative_pose, landmarks, 62);
    // double reproj_error = calcReprojectionError(intrinsic, p_id, prev_image, image0_kp_pts, c_id, curr_image, image1_kp_pts, mask, relative_pose, landmarks, num_matches);
    // std::cout << "reprojection error: " << reproj_error << std::endl;


    // //**========== 5. Optimize ==========**//
    // optimize(intrinsic, prev_frame_kp_pts, Eigen::Isometry3d::Identity(), curr_frame_kp_pts, relative_pose, landmarks, mask, true);
    // std::cout << "=========================== after optimization ===============================================" << std::endl;
    // double reproj_error2 = calcReprojectionError(intrinsic, p_id, prev_image, prev_frame_kp_pts, c_id, curr_image, curr_frame_kp_pts, mask, relative_pose, landmarks, 63);


    //**========== End ==========**//
    // Evaluate
    std::vector<Eigen::Isometry3d> gt_poses, aligned_poses;
    double rpe_rot, rpe_trans;
    loadGT(config_file["gt_path"], p_id, gt_poses);
    Eigen::Isometry3d aligned_est_pose = alignPose(gt_poses, relative_pose);
    aligned_poses.push_back(aligned_est_pose);
    calcRPE_rt(gt_poses, aligned_est_pose, rpe_rot, rpe_trans);

    std::cout << "RPEr: " << rpe_rot << std::endl;
    std::cout << "RPEt: " << rpe_trans << std::endl;


    // log
    logTrajectory(std::vector<Eigen::Isometry3d>{relative_pose});

    // visualize
    // displayPoses(gt_poses, poses, aligned_poses);
    displayFramesAndLandmarks(poses, landmarks);

    return 0;
}