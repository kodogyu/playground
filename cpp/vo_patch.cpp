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
void displayPoses(const std::vector<Eigen::Isometry3d> &poses);

// response comparison, for list sorting
bool compare_response(cv::KeyPoint first, cv::KeyPoint second)
{
  if (first.response < second.response) return true;
  else return false;
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

    // good matches
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < image_matches01_vec.size(); i++) {
        if (image_matches01_vec[i][0].distance < image_matches01_vec[i][1].distance * des_dist_thresh) {  // prev -> curr match에서 좋은가?
            int image1_keypoint_idx = image_matches01_vec[i][0].trainIdx;
            if (image_matches10_vec[image1_keypoint_idx][0].distance < image_matches10_vec[image1_keypoint_idx][1].distance * des_dist_thresh) {  // curr -> prev match에서 좋은가?
                if (image_matches01_vec[i][0].queryIdx == image_matches10_vec[image1_keypoint_idx][0].trainIdx)
                    good_matches.push_back(image_matches01_vec[i][0]);
            }
        }
    }

    std::cout << "original features for image" + std::to_string(p_id) + "&" + std::to_string(c_id) + ": " << image_matches01_vec.size() << std::endl;
    std::cout << "good features for image" + std::to_string(p_id) + "&" + std::to_string(c_id) + ": " << good_matches.size() << std::endl;

    cv::Mat image_matches;
    cv::drawMatches(prev_image, prev_image_keypoints, curr_image, curr_image_keypoints, good_matches, image_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), cv::Mat(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imwrite("vo_patch/inter_frames/(vo_patch)frame"
            + std::to_string(p_id)
            + "&"
            + std::to_string(c_id)
            + "_kp_matches(raw).png", image_matches);

    // RANSAC
    std::vector<cv::KeyPoint> image0_kps;
    std::vector<cv::KeyPoint> image1_kps;
    std::vector<cv::Point2f> image0_kp_pts;
    std::vector<cv::Point2f> image1_kp_pts;
    std::vector<cv::Point2f> top_image0_kp_pts;
    std::vector<cv::Point2f> top_image1_kp_pts;
    for (auto match : good_matches) {
        image0_kp_pts.push_back(prev_image_keypoints[match.queryIdx].pt);
        image1_kp_pts.push_back(curr_image_keypoints[match.trainIdx].pt);
        image0_kps.push_back(prev_image_keypoints[match.queryIdx]);
        image1_kps.push_back(curr_image_keypoints[match.trainIdx]);
    }
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

    logTrajectory(std::vector<Eigen::Isometry3d>{relative_pose});
    displayPoses(poses);

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


void displayPoses(const std::vector<Eigen::Isometry3d> &poses) {
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