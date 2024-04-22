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
    std::vector<cv::Point2f> frame0_kp_pts_test = {cv::Point2f(89, 92), cv::Point2f(280, 148), cv::Point2f(437, 204), cv::Point2f(434, 161), cv::Point2f(495, 188), cv::Point2f(660, 150), cv::Point2f(709, 136), cv::Point2f(808, 127), cv::Point2f(913, 110), cv::Point2f(899, 294)};
    // frame1
    std::vector<cv::Point2f> frame1_kp_pts_test = {cv::Point2f(77, 92), cv::Point2f(275, 149), cv::Point2f(435, 207), cv::Point2f(435, 163), cv::Point2f(496, 190), cv::Point2f(664, 150), cv::Point2f(714, 136), cv::Point2f(817, 126), cv::Point2f(925, 109), cv::Point2f(933, 307)};
    // frame2
    std::vector<cv::Point2f> frame2_kp_pts_test = {cv::Point2f(65, 90), cv::Point2f(270, 149), cv::Point2f(434, 208), cv::Point2f(436, 163), cv::Point2f(498, 191), cv::Point2f(668, 151), cv::Point2f(720, 136), cv::Point2f(827, 125), cv::Point2f(939, 108), cv::Point2f(977, 324)};
    // frame3
    std::vector<cv::Point2f> frame3_kp_pts_test = {cv::Point2f(51, 88), cv::Point2f(265, 149), cv::Point2f(432, 210), cv::Point2f(437, 164), cv::Point2f(500, 193), cv::Point2f(672, 151), cv::Point2f(726, 136), cv::Point2f(838, 125), cv::Point2f(954, 106), cv::Point2f(1035, 346)};
    // frame4
    std::vector<cv::Point2f> frame4_kp_pts_test = {cv::Point2f(37, 85), cv::Point2f(260, 148), cv::Point2f(431, 211), cv::Point2f(439, 164), cv::Point2f(502, 193), cv::Point2f(677, 151), cv::Point2f(733, 136), cv::Point2f(851, 123), cv::Point2f(972, 104),    cv::Point2f(653, 124)};

    std::vector<std::vector<cv::Point2f>> matches_vec = {frame0_kp_pts_test, frame1_kp_pts_test, frame2_kp_pts_test, frame3_kp_pts_test, frame4_kp_pts_test};

    int prev_frame_idx = config_file["prev_frame_idx"];
    std::vector<cv::Point2f> prev_frame_kp_pts = {cv::Point2f(51, 88), cv::Point2f(265, 149), cv::Point2f(432, 210), cv::Point2f(437, 164), cv::Point2f(500, 193), cv::Point2f(672, 151), cv::Point2f(726, 136), cv::Point2f(838, 125), cv::Point2f(954, 106)};
    std::vector<cv::Point2f> curr_frame_kp_pts = {cv::Point2f(37, 85), cv::Point2f(260, 148), cv::Point2f(431, 211), cv::Point2f(439, 164), cv::Point2f(502, 193), cv::Point2f(677, 151), cv::Point2f(733, 136), cv::Point2f(851, 123), cv::Point2f(972, 104)};
    // std::vector<cv::Point2f> prev_frame_kp_pts = matches_vec[prev_frame_idx];
    // std::vector<cv::Point2f> curr_frame_kp_pts = matches_vec[prev_frame_idx + 1];

    cv::Mat mask;
    cv::Mat essential_mat = cv::findEssentialMat(curr_frame_kp_pts, prev_frame_kp_pts, intrinsic, cv::RANSAC, 0.999, 1.0, mask);

    int essential_inlier = 0;
    for (int i = 0; i < 10; i++) {
        if (mask.at<unsigned char>(i) == 1) {
            essential_inlier++;
        }
    }
    std::cout << "essential inlier: " << essential_inlier << std::endl;

    // draw keypoints
    drawKeypoints(p_id, prev_image, prev_frame_kp_pts, c_id, curr_image, curr_frame_kp_pts, mask, 10);
    // drawKeypoints(p_id, prev_image, image0_kp_pts, c_id, curr_image, image1_kp_pts, num_matches);


    //**========== 3. Motion estimation ==========**//
    cv::Mat R, t;
    cv::recoverPose(essential_mat, curr_frame_kp_pts, prev_frame_kp_pts, intrinsic, R, t, mask);
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
    triangulate2(intrinsic, prev_frame_kp_pts, curr_frame_kp_pts, relative_pose, mask, landmarks);
    // triangulate2(intrinsic, frame1_kp_pts_test, frame2_kp_pts_test, relative_pose, landmarks);
    // triangulate2(intrinsic, image0_kp_pts, image1_kp_pts, relative_pose, landmarks);

    // calculate reprojection error & save the images
    double reproj_error = calcReprojectionError(intrinsic, p_id, prev_image, prev_frame_kp_pts, c_id, curr_image, curr_frame_kp_pts, mask, relative_pose, landmarks, 10);
    // double reproj_error = calcReprojectionError(intrinsic, p_id, prev_image, image0_kp_pts, c_id, curr_image, image1_kp_pts, mask, relative_pose, landmarks, num_matches);
    std::cout << "reprojection error: " << reproj_error << std::endl;

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