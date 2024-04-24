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

/*#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/StereoFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>*/
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
                    cv::Mat mask,
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

void triangulate2(cv::Mat intrinsic, std::vector<cv::Point2f> img0_kp_pts, std::vector<cv::Point2f> img1_kp_pts, Eigen::Isometry3d &cam1_pose, const cv::Mat &mask, std::vector<Eigen::Vector3d> &landmarks) {
    Eigen::Matrix3d camera_intrinsic;
    cv::cv2eigen(intrinsic, camera_intrinsic);
    Eigen::MatrixXd prev_proj = Eigen::MatrixXd::Identity(3, 4);
    Eigen::MatrixXd curr_proj = Eigen::MatrixXd::Identity(3, 4);

    prev_proj = camera_intrinsic * prev_proj;
    curr_proj = camera_intrinsic * curr_proj * cam1_pose.inverse().matrix();

    for (int i = 0; i < img0_kp_pts.size(); i++) {
        if (mask.at<unsigned char>(i) != 1) {
            landmarks.push_back(Eigen::Vector3d(0, 0, 0));
            continue;
        }
        std::cout << "triangulate " << i << std::endl;

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

    // for (int l = 0; l < prev_frame_id; l++) {
    for (int l = 0; l < 3; l++) {
        std::getline(gt_poses_file, line);
    }

    for (int i = 0; i < 2; i++) {
        std::getline(gt_poses_file, line);
        std::stringstream ssline(line);

        // // KITTI format
        // ssline
        //     >> r11 >> r12 >> r13 >> t1
        //     >> r21 >> r22 >> r23 >> t2
        //     >> r31 >> r32 >> r33 >> t3;

        // std::cout << "gt_pose[" << prev_frame_id + i << "]: "
        //             << r11 << " " << r12 << " " << r13 << " " << t1
        //             << r21 << " " << r22 << " " << r23 << " " << t2
        //             << r31 << " " << r32 << " " << r33 << " " << t3 << std::endl;

        // TUM format
        ssline >> time_stamp
                >> t1 >> t2 >> t3
                >> qx >> qy >> qz >> qw;

        // Eigen::Matrix3d rotation_mat;
        // rotation_mat << r11, r12, r13,
        //                 r21, r22, r23,
        //                 r31, r32, r33;
        Eigen::Quaterniond quaternion(qw, qx, qy, qz);
        Eigen::Matrix3d rotation_mat(quaternion);

        Eigen::Vector3d translation_mat;
        translation_mat << t1, t2, t3;

        Eigen::Isometry3d gt_pose = Eigen::Isometry3d::Identity();
        gt_pose.linear() = rotation_mat;
        gt_pose.translation() = translation_mat;

        std::cout << "gt_pose: \n" << gt_pose.matrix() << std::endl;
        gt_poses.push_back(gt_pose);
    }
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

/*void optimizeFrames(std::vector<std::shared_ptr<Frame>> &frames, bool verbose) {
    // create a graph
    gtsam::NonlinearFactorGraph graph;

    // stereo camera calibration object
    gtsam::Cal3_S2::shared_ptr K(
        new gtsam::Cal3_S2(frames[0]->pCamera_->fx_,
                                frames[0]->pCamera_->fy_,
                                frames[0]->pCamera_->s_,
                                frames[0]->pCamera_->cx_,
                                frames[0]->pCamera_->cy_));

    // create initial values
    gtsam::Values initial_estimates;

    // 1. Add Values and Factors
    const auto measurement_noise = gtsam::noiseModel::Isotropic::Sigma(2, 1.0);  // std of 1px.

    // insert values and factors of the frames
    std::vector<int> frame_pose_map;  // frame_pose_map[pose_idx] = frame_id
    std::vector<int> landmark_idx_id_map;  // landmark_map[landmark_idx] = landmark_id
    std::map<int, int> landmark_id_idx_map;  // landmark_map[landmark_id] = landmark_idx

    int landmark_idx = 0;
    int landmarks_cnt = 0;
    for (int frame_idx = 0; frame_idx < frames.size(); frame_idx++) {
        std::shared_ptr<Frame> pFrame = frames[frame_idx];
        gtsam::Pose3 frame_pose = gtsam::Pose3(gtsam::Rot3(pFrame->pose_.rotation()), gtsam::Point3(pFrame->pose_.translation()));
        // insert initial value of the frame pose
        initial_estimates.insert(gtsam::Symbol('x', frame_idx), frame_pose);

        for (const auto pLandmark : pFrame->landmarks_) {
            // insert initial value of the landmark
            std::map<int, int>::iterator landmark_map_itr = landmark_id_idx_map.find(pLandmark->id_);
            if (landmark_map_itr == landmark_id_idx_map.end()) {  // new landmark
                landmark_idx = landmarks_cnt;

                initial_estimates.insert<gtsam::Point3>(gtsam::Symbol('l', landmark_idx), gtsam::Point3(pLandmark->point_3d_));
                landmark_idx_id_map.push_back(pLandmark->id_);
                landmark_id_idx_map[pLandmark->id_] = landmark_idx;

                landmarks_cnt++;
            }
            else {
                landmark_idx = landmark_map_itr->second;
            }
            // 2D measurement
            cv::Point2f measurement_cv = pFrame->keypoints_[pLandmark->observations_.find(pFrame->id_)->second].pt;
            gtsam::Point2 measurement(measurement_cv.x, measurement_cv.y);
            // add measurement factor
            graph.emplace_shared<gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>>(measurement, measurement_noise,
                                                                                                            gtsam::Symbol('x', frame_idx), gtsam::Symbol('l', landmark_idx),
                                                                                                            K);
        }
    }


    // 2. prior factors
    gtsam::Pose3 first_pose = gtsam::Pose3(gtsam::Rot3(frames[0]->pose_.rotation()), gtsam::Point3(frames[0]->pose_.translation()));
    // gtsam::Pose3 second_pose = gtsam::Pose3(gtsam::Rot3(frames[1]->pose_.rotation()), gtsam::Point3(frames[1]->pose_.translation()));
    // add a prior on the first pose
    const auto pose_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << gtsam::Vector3::Constant(0.3), gtsam::Vector3::Constant(0.1)).finished());  // std of 0.3m for x,y,z, 0.1rad for r,p,y
    graph.addPrior(gtsam::Symbol('x', 0), first_pose, pose_noise);
    // // add a prior on the second pose (for scale)
    // graph.addPrior(gtsam::Symbol('x', 1), second_pose, pose_noise);
    // add a prior on the first landmark (for scale)
    auto point_noise = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);
    graph.addPrior(gtsam::Symbol('l', 0), frames[0]->landmarks_[0]->point_3d_, point_noise);

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
    for (int frame_idx = 0; frame_idx < frames.size(); frame_idx++) {
        std::shared_ptr<Frame> pFrame = frames[frame_idx];
        pFrame->pose_ = result.at<gtsam::Pose3>(gtsam::Symbol('x', frame_idx)).matrix();

        std::shared_ptr<Frame> pPrev_frame = pFrame->pPrevious_frame_.lock();
        pFrame->relative_pose_ = pPrev_frame->pose_.inverse() * pFrame->pose_;

        // Recover Landmark point
        for (int j = 0; j < pFrame->landmarks_.size(); j++) {
            std::shared_ptr<Landmark> pLandmark = pFrame->landmarks_[j];
            std::map<int, int>::iterator landmark_map_itr = landmark_id_idx_map.find(pLandmark->id_);
            if (landmark_map_itr != landmark_id_idx_map.end()) {
                pLandmark->point_3d_ = result.at<gtsam::Point3>(gtsam::Symbol('l', landmark_map_itr->second));
            }
        }
    }
    if (verbose) {
        graph.print("graph print:\n");
        result.print("optimization result:\n");
    }
}
*/

int countMask(const cv::Mat &mask) {
    int count = 0;
    for (int i = 0; i < 10; i++) {
        if (mask.at<unsigned char>(i) == 1) {
            count++;
        }
    }
    return count;
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
    // std::vector<cv::Point2f> frame0_kp_pts_test = {cv::Point2f(89.0, 92.0), cv::Point2f(279.65207, 147.67671), cv::Point2f(440.3918, 203.43727), cv::Point2f(434.0, 161.0), cv::Point2f(486.80063, 190.52914), cv::Point2f(661.3766, 148.82133), cv::Point2f(709.0, 136.0), cv::Point2f(807.58307, 125.778046), cv::Point2f(916.523, 110.59789), cv::Point2f(907.6637, 294.75223)};
    // tum checkerboard, 63개 (.409720.png)
    std::vector<cv::Point2f> frame0_kp_pts_test = {cv::Point2f(185.98361, 255.33424), cv::Point2f(199.0621, 257.06512), cv::Point2f(211.93787, 258.88757), cv::Point2f(225.30211, 260.8278), cv::Point2f(238.0471, 262.44495), cv::Point2f(251.32112, 264.19113), cv::Point2f(263.93756, 265.82367), cv::Point2f(277.21017, 267.4266), cv::Point2f(289.7167, 269.32452), cv::Point2f(183.60417, 268.46786), cv::Point2f(196.75133, 270.04303), cv::Point2f(210.08305, 271.65692), cv::Point2f(223.33095, 273.50418), cv::Point2f(236.30656, 275.05783), cv::Point2f(249.50627, 276.86057), cv::Point2f(262.3463, 278.86392), cv::Point2f(275.50507, 280.47092), cv::Point2f(288.29028, 282.06607), cv::Point2f(181.45612, 281.2323), cv::Point2f(194.626, 282.87787), cv::Point2f(207.89014, 284.67996), cv::Point2f(221.0085, 286.65448), cv::Point2f(233.89561, 288.1843), cv::Point2f(247.44759, 289.8412), cv::Point2f(260.40274, 291.54437), cv::Point2f(273.57608, 293.13394), cv::Point2f(286.5793, 294.89896), cv::Point2f(178.94438, 294.63147), cv::Point2f(192.10973, 296.19675), cv::Point2f(205.64519, 297.79102), cv::Point2f(218.89854, 299.5839), cv::Point2f(232.19884, 301.0378), cv::Point2f(245.59622, 302.8427), cv::Point2f(258.5278, 304.77774), cv::Point2f(271.6318, 306.38046), cv::Point2f(284.8339, 308.03354), cv::Point2f(176.39893, 307.6902), cv::Point2f(190.0887, 309.26962), cv::Point2f(203.64032, 311.12424), cv::Point2f(216.75804, 312.74728), cv::Point2f(229.92258, 314.5272), cv::Point2f(243.43706, 316.25223), cv::Point2f(256.5954, 317.77914), cv::Point2f(269.90576, 319.37866), cv::Point2f(283.29272, 321.0969), cv::Point2f(174.2677, 320.8919), cv::Point2f(187.97784, 322.7443), cv::Point2f(201.19373, 324.6455), cv::Point2f(214.4658, 326.1613), cv::Point2f(228.03183, 327.8519), cv::Point2f(241.62836, 329.47943), cv::Point2f(254.76276, 330.86276), cv::Point2f(267.80664, 332.7646), cv::Point2f(281.32922, 334.5889), cv::Point2f(172.07285, 334.64755), cv::Point2f(185.34409, 336.56198), cv::Point2f(198.8366, 338.05148), cv::Point2f(212.45972, 339.69293), cv::Point2f(225.9317, 341.32623), cv::Point2f(239.38484, 342.8262), cv::Point2f(252.78525, 344.6163), cv::Point2f(266.17078, 346.4669), cv::Point2f(279.53818, 347.9511)};
    // std::vector<cv::Point2f> frame0_kp_pts_test = {cv::Point2f(186, 255), cv::Point2f(199, 257), cv::Point2f(212, 259), cv::Point2f(225, 261), cv::Point2f(238, 262), cv::Point2f(251, 264), cv::Point2f(264, 266), cv::Point2f(277, 267), cv::Point2f(290, 269), cv::Point2f(184, 268), cv::Point2f(197, 270), cv::Point2f(210, 272), cv::Point2f(223, 274), cv::Point2f(236, 275), cv::Point2f(250, 277), cv::Point2f(262, 279), cv::Point2f(276, 280), cv::Point2f(288, 282), cv::Point2f(181, 281), cv::Point2f(195, 283), cv::Point2f(208, 285), cv::Point2f(221, 287), cv::Point2f(234, 288), cv::Point2f(247, 290), cv::Point2f(260, 292), cv::Point2f(274, 293), cv::Point2f(287, 295), cv::Point2f(179, 295), cv::Point2f(192, 296), cv::Point2f(206, 298), cv::Point2f(219, 300), cv::Point2f(232, 301), cv::Point2f(246, 303), cv::Point2f(259, 305), cv::Point2f(272, 306), cv::Point2f(285, 308), cv::Point2f(176, 308), cv::Point2f(190, 309), cv::Point2f(204, 311), cv::Point2f(217, 313), cv::Point2f(230, 315), cv::Point2f(243, 316), cv::Point2f(257, 318), cv::Point2f(270, 319), cv::Point2f(283, 321), cv::Point2f(174, 321), cv::Point2f(188, 323), cv::Point2f(201, 325), cv::Point2f(214, 326), cv::Point2f(228, 328), cv::Point2f(242, 329), cv::Point2f(255, 331), cv::Point2f(268, 333), cv::Point2f(281, 335), cv::Point2f(172, 335), cv::Point2f(185, 337), cv::Point2f(199, 338), cv::Point2f(212, 340), cv::Point2f(226, 341), cv::Point2f(239, 343), cv::Point2f(253, 345), cv::Point2f(266, 346), cv::Point2f(280, 348)};

    // frame1
    // std::vector<cv::Point2f> frame1_kp_pts_test = {cv::Point2f(275, 150),cv::Point2f(340, 208), cv::Point2f(439, 206), cv::Point2f(488, 193), cv::Point2f(642, 124), cv::Point2f(816, 125), cv::Point2f(929, 109), cv::Point2f(943, 308), cv::Point2f(779, 158), cv::Point2f(881, 155)};
    // std::vector<cv::Point2f> frame1_kp_pts_test = {cv::Point2f(77, 92), cv::Point2f(275, 149), cv::Point2f(435, 207), cv::Point2f(435, 163), cv::Point2f(496, 190), cv::Point2f(664, 150), cv::Point2f(714, 136), cv::Point2f(817, 126), cv::Point2f(925, 109), cv::Point2f(933, 307)};
    // subpixel, 기본 10개
    // std::vector<cv::Point2f> frame1_kp_pts_test = {cv::Point2f(76.435524, 92.402176), cv::Point2f(274.63477, 149.50064), cv::Point2f(439.3174, 206.34424), cv::Point2f(435.0, 163.0), cv::Point2f(487.71252, 192.5296), cv::Point2f(664.4662, 149.73691), cv::Point2f(714.8803, 135.63571), cv::Point2f(816.14087, 125.12716), cv::Point2f(928.7381, 109.08112), cv::Point2f(942.7255, 307.60168)};
    // tum checkerboard, 63개 (.445969.png)
    std::vector<cv::Point2f> frame1_kp_pts_test = {cv::Point2f(188.50336, 254.28859), cv::Point2f(201.61862, 256.00037), cv::Point2f(214.82784, 257.61743), cv::Point2f(227.99135, 259.2137), cv::Point2f(240.9269, 260.97116), cv::Point2f(253.95947, 262.65988), cv::Point2f(266.96304, 264.2749), cv::Point2f(279.88977, 265.71893), cv::Point2f(292.7736, 267.5034), cv::Point2f(186.67285, 266.95377), cv::Point2f(199.6509, 268.7468), cv::Point2f(212.82109, 270.5306), cv::Point2f(225.98372, 272.1499), cv::Point2f(239.08856, 273.56796), cv::Point2f(252.16556, 275.35953), cv::Point2f(265.0837, 277.1004), cv::Point2f(278.1214, 278.60114), cv::Point2f(291.15082, 280.2634), cv::Point2f(184.32774, 280.10443), cv::Point2f(197.61137, 281.6256), cv::Point2f(210.93588, 283.37775), cv::Point2f(224.28981, 284.8726), cv::Point2f(237.33499, 286.56235), cv::Point2f(250.51984, 288.32553), cv::Point2f(263.5544, 289.79327), cv::Point2f(276.68262, 291.35413), cv::Point2f(289.65073, 293.04395), cv::Point2f(182.41861, 292.95807), cv::Point2f(195.58974, 294.77478), cv::Point2f(208.92451, 296.49942), cv::Point2f(222.20618, 298.08133), cv::Point2f(235.44948, 299.5277), cv::Point2f(248.69556, 301.25595), cv::Point2f(261.8787, 302.814), cv::Point2f(275.05862, 304.41858), cv::Point2f(288.23535, 306.05096), cv::Point2f(180.09937, 306.39767), cv::Point2f(193.39595, 308.14496), cv::Point2f(206.92966, 309.59625), cv::Point2f(220.48015, 311.04932), cv::Point2f(233.58061, 312.8338), cv::Point2f(246.79042, 314.53445), cv::Point2f(260.04233, 316.07474), cv::Point2f(273.36606, 317.4822), cv::Point2f(286.52466, 319.16614), cv::Point2f(177.803, 319.75412), cv::Point2f(191.4469, 321.35947), cv::Point2f(204.9484, 322.82416), cv::Point2f(218.24597, 324.51074), cv::Point2f(231.55283, 326.2864), cv::Point2f(245.17305, 327.6438), cv::Point2f(258.54037, 329.15945), cv::Point2f(271.64615, 330.8213), cv::Point2f(284.77243, 332.54663), cv::Point2f(175.71579, 333.29483), cv::Point2f(189.30305, 334.76346), cv::Point2f(202.7548, 336.44424), cv::Point2f(216.32068, 338.13422), cv::Point2f(229.76256, 339.5763), cv::Point2f(243.20312, 341.05753), cv::Point2f(256.59732, 342.72125), cv::Point2f(269.98282, 344.40897), cv::Point2f(283.34937, 345.8332)};
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

    std::vector<cv::Point2f> prev_frame_kp_pts_temp;
    std::vector<cv::Point2f> curr_frame_kp_pts_temp;
    for (int i = 0; i < prev_frame_kp_pts.size(); i++) {
        if (i % 2 == 0) {
            prev_frame_kp_pts_temp.push_back(prev_frame_kp_pts[i]);
            curr_frame_kp_pts_temp.push_back(curr_frame_kp_pts[i]);
        }
    }
    prev_frame_kp_pts = prev_frame_kp_pts_temp;
    curr_frame_kp_pts = curr_frame_kp_pts_temp;

    cv::Mat mask;
    cv::Mat essential_mat = cv::findEssentialMat(curr_frame_kp_pts, prev_frame_kp_pts, intrinsic, cv::RANSAC, 0.999, 1.0, mask);

    int essential_inlier = countMask(mask);
    std::cout << "essential inlier: " << essential_inlier << std::endl;

    // draw keypoints
    drawKeypoints(p_id, prev_image, prev_frame_kp_pts, c_id, curr_image, curr_frame_kp_pts, mask, 10);
    // drawKeypoints(p_id, prev_image, image0_kp_pts, c_id, curr_image, image1_kp_pts, num_matches);


    //**========== 3. Motion estimation ==========**//
    cv::Mat R, t;
    cv::recoverPose(essential_mat, curr_frame_kp_pts, prev_frame_kp_pts, intrinsic, R, t, mask);
    // cv::recoverPose(essential_mat, image1_kp_pts, image0_kp_pts, intrinsic, R, t, mask);

    int pose_inlier = countMask(mask);
    std::cout << "pose inlier: " << pose_inlier << std::endl;

    Eigen::Matrix3d rotation_mat;
    Eigen::Vector3d translation_mat;
    cv::cv2eigen(R, rotation_mat);
    cv::cv2eigen(t, translation_mat);

    relative_pose.linear() = rotation_mat;
    relative_pose.translation() = translation_mat;
    poses.push_back(relative_pose);


    //**========== 4. Triangulation ==========**//
    std::vector<Eigen::Vector3d> landmarks;
    triangulate2(intrinsic, prev_frame_kp_pts, curr_frame_kp_pts, relative_pose, mask, landmarks);
    // triangulate2(intrinsic, frame1_kp_pts_test, frame2_kp_pts_test, relative_pose, landmarks);
    // triangulate2(intrinsic, image0_kp_pts, image1_kp_pts, relative_pose, landmarks);

    // calculate reprojection error & save the images
    double reproj_error = calcReprojectionError(intrinsic, p_id, prev_image, prev_frame_kp_pts, c_id, curr_image, curr_frame_kp_pts, mask, relative_pose, landmarks, 63);
    // double reproj_error = calcReprojectionError(intrinsic, p_id, prev_image, image0_kp_pts, c_id, curr_image, image1_kp_pts, mask, relative_pose, landmarks, num_matches);
    // std::cout << "reprojection error: " << reproj_error << std::endl;

    // Evaluate
    std::vector<Eigen::Isometry3d> gt_poses, aligned_poses;
    loadGT(config_file["gt_path"], p_id, gt_poses);

    // log
    logTrajectory(std::vector<Eigen::Isometry3d>{relative_pose});

    // visualize
    displayPoses(gt_poses, poses, aligned_poses);
    displayFramesAndLandmarks(poses, landmarks);

    return 0;
}