// visual odometry with 2 images.

#include <iostream>

#include <opencv2/opencv.hpp>

#include <gtsam/geometry/Pose3.h>

int main(int argc, char** argv) {
    //** Image load **//
    if (argc != 5) {
        std::cout << "Usage: visual_odometry_example image1_left image1_right image2_left image2_right" << std::endl;
        return 1;
    }

    cv::Mat image1_left, image1_right, image2_left, image2_right;
    image1_left = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    image1_right = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    image2_left = cv::imread(argv[3], cv::IMREAD_GRAYSCALE);
    image2_right = cv::imread(argv[4], cv::IMREAD_GRAYSCALE);

    //** Feature extraction **//
    std::vector<cv::KeyPoint> img1_left_kps;
    cv::Mat img1_left_descriptors;
    // std::vector<cv::KeyPoint> img1_right_kps;
    // cv::Mat img1_right_descriptors;
    std::vector<cv::KeyPoint> img2_left_kps;
    cv::Mat img2_left_descriptors;
    std::vector<cv::KeyPoint> img2_right_kps;
    cv::Mat img2_right_descriptors;
    // create orb feature extractor
    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    orb->detectAndCompute(image1_left, cv::Mat(), img1_left_kps, img1_left_descriptors);
    // orb->detectAndCompute(image1_right, cv::Mat(), img1_right_kps, img1_right_descriptors);
    orb->detectAndCompute(image2_left, cv::Mat(), img2_left_kps, img2_left_descriptors);
    orb->detectAndCompute(image2_right, cv::Mat(), img2_right_kps, img2_right_descriptors);

    //** Feature matching **//
    // create a matcher
    cv::Ptr<cv::DescriptorMatcher> orb_matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
    // image1 left & right (template matching)
    int templ_width = 30, templ_height = 30;
    int stripe_width = 50, stripe_height = 40;
    cv::Rect roi_templ, roi_stripe;
    cv::Mat templ, stripe;
    std::vector<int> img1_kp_map;
    std::vector<cv::Point> img1_right_kps_pts;
    for (int i = 0; i < img1_left_kps.size(); i++) {
        cv::KeyPoint img1_left_kp = img1_left_kps[i];
        if (img1_left_kp.pt.x < stripe_width/2 || img1_left_kp.pt.x + stripe_width/2 > image1_left.cols ||
            img1_left_kp.pt.y < stripe_height/2 || img1_left_kp.pt.y + stripe_height/2 > image1_left.rows) {
                continue;
            }
        // template
        roi_templ = cv::Rect(cv::Point(img1_left_kp.pt) - cv::Point(templ_width/2, templ_height/2),
                        cv::Point(img1_left_kp.pt) + cv::Point(templ_width/2, templ_height/2));
        templ = cv::Mat(image1_left, roi_templ);
        // stripe
        roi_stripe = cv::Rect(cv::Point(img1_left_kp.pt) - cv::Point(stripe_width/2, stripe_height/2),
                        cv::Point(img1_left_kp.pt) + cv::Point(stripe_width/2, stripe_height/2));
        stripe = cv::Mat(image1_right, roi_stripe);

        // match template
        cv::Mat result;
        cv::matchTemplate(stripe, templ, result, CV_TM_SQDIFF_NORMED);
        // find best matching point
        double minval, maxval;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(result, &minval, &maxval, &minLoc, &maxLoc);
        img1_kp_map.push_back(i);  // img1_kp_map size = amount of right keypoints
        img1_right_kps_pts.push_back(minLoc + cv::Point(img1_left_kp.pt) - cv::Point(stripe_width/2, stripe_height/2) + cv::Point(templ_width/2, templ_height/2));
    }

    // image1 left & image2 left
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

    // // image2 left & right (matcher matching)
    // std::vector<std::vector<cv::DMatch>> matches_img2_vec;
    // orb_matcher->knnMatch(img2_left_descriptors, img2_right_descriptors, matches_img2_vec, 2);

    // std::vector<cv::DMatch> matches_img2;
    // for (int i = 0; i < matches_img2_vec.size(); i++) {
    //     if (matches_img2_vec[i][0].distance < matches_img2_vec[i][1].distance * dist_thresh) {
    //         matches_img2.push_back(matches_img2_vec[i][0]);
    //     }
    // }
    // std::cout << "original features for image2: " << matches_img2_vec.size() << std::endl;
    // std::cout << "good features for image2: " << matches_img2.size() << std::endl;

    // draw matches
    cv::Mat left_image, right_image;
    cv::cvtColor(image1_left, left_image, cv::COLOR_GRAY2BGR);
    cv::cvtColor(image1_right, right_image, cv::COLOR_GRAY2BGR);

    cv::Mat img1_matches;
    cv::hconcat(left_image, right_image, img1_matches);

    for (int i = 0; i < img1_right_kps_pts.size(); i++) {
        cv::Point left_kp_pt;
        cv::Point right_kp_pt;

        left_kp_pt = img1_left_kps[img1_kp_map[i]].pt;
        right_kp_pt = img1_right_kps_pts[i];
        right_kp_pt.x += left_image.cols;

        cv::rectangle(img1_matches,
            left_kp_pt - cv::Point(5, 5),
            left_kp_pt + cv::Point(5, 5),
            cv::Scalar(0, 255, 0));  // green
        cv::rectangle(img1_matches,
            right_kp_pt - cv::Point(5, 5),
            right_kp_pt + cv::Point(5, 5),
            cv::Scalar(0, 255, 0));  // green
        cv::line(img1_matches, left_kp_pt, right_kp_pt, cv::Scalar(0, 255, 0));
    }

    cv::Mat img1_2_matches, img2_matches;
    cv::drawMatches(image1_left, img1_left_kps,
                    image2_left, img2_left_kps,
                    matches_img1_2, img1_2_matches);
    // cv::drawMatches(image1_left, img2_left_kps,
    //                 image1_right, img2_right_kps,
    //                 matches_img2, img2_matches);

    cv::imshow("image1 matches: " + std::to_string(img1_right_kps_pts.size()), img1_matches);
    cv::imshow("image1&2 matches", img1_2_matches);
    // cv::imshow("image2 matches", img2_matches);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    //** Triangulation **//
    // camera intrinsic parameters
    double fx = 620.070096090849;
    double fy = 618.2102185572654;
    double cx = 325.29844703787114;
    double cy = 258.48711395621467;

    std::vector<gtsam::Point3> keypoints_3d;
    std::vector<cv::Point2f> img2_left_kp_pts;
    std::vector<int> img1_2_kp_map;
    cv::Mat spare_img1;
    cv::cvtColor(image1_left, spare_img1, cv::COLOR_GRAY2BGR);

    for (int i = 0; i < matches_img1_2.size(); i++) {
        for (int j = 0; j < img1_kp_map.size(); j++) {
            if (matches_img1_2[i].queryIdx == img1_kp_map[j]) {
                // keypoint in camera coordinate (normalized image plane)
                gtsam::Vector3 versor;
                cv::KeyPoint keypoint = img1_left_kps[matches_img1_2[i].queryIdx];  // image1 keypoint
                versor[0] = (keypoint.pt.x - cx) / fx;
                versor[1] = (keypoint.pt.y - cy) / fx;
                versor[2] = 1;
                // disparity to depth
                cv::Point img1_left_kp_pt = keypoint.pt;
                cv::Point img1_right_kp_pt = img1_right_kps_pts[j];
                float disparity = img1_left_kp_pt.x - img1_right_kp_pt.x;
                if (disparity > 0) {
                    float depth = fx * 0.008 / disparity;  // baseline = 0.008m (=8mm)
                    std::cout << "depth: " << depth << std::endl;
                    std::cout << "point: " << img1_left_kp_pt << std::endl;
                    cv::circle(spare_img1, img1_left_kp_pt, 3, cv::Scalar(0, 0, 5 * i), -1);
                    // 3D keypoint in camera coordinate
                    gtsam::Point3 keypoint_3d;
                    keypoint_3d = versor * depth;
                    keypoints_3d.push_back(keypoint_3d);
                    // corresponding image2 keypoint
                    cv::KeyPoint img2_keypoint = img2_left_kps[matches_img1_2[i].trainIdx];
                    img2_left_kp_pts.push_back(img2_keypoint.pt);
                    img1_2_kp_map.push_back(matches_img1_2[i].queryIdx);
                }
            }
        }
    }
    std::cout << "3D keypoint object points: " << keypoints_3d.size() << std::endl;

    cv::imshow("spare image 1", spare_img1);

    // draw matches
    cv::Mat image1_left_copy, image2_left_copy;
    cv::cvtColor(image1_left, image1_left_copy, cv::COLOR_GRAY2BGR);
    cv::cvtColor(image2_left, image2_left_copy, cv::COLOR_GRAY2BGR);

    cv::Mat img1_2_matches_temp;
    cv::hconcat(image1_left_copy, image2_left_copy, img1_2_matches_temp);

    // for (int i = 0; i < img2_left_kp_pts.size(); i++) {
    for (int i = 0; i < 10; i++) {
        cv::Point img1_kp_pt;
        cv::Point img2_kp_pt;

        img1_kp_pt = img1_left_kps[img1_2_kp_map[i]].pt;
        img2_kp_pt = img2_left_kp_pts[i];
        img2_kp_pt.x += image1_left_copy.cols;

        cv::rectangle(img1_2_matches_temp,
            img1_kp_pt - cv::Point(5, 5),
            img1_kp_pt + cv::Point(5, 5),
            cv::Scalar(0, 255, 0));  // green
        cv::rectangle(img1_2_matches_temp,
            img2_kp_pt - cv::Point(5, 5),
            img2_kp_pt + cv::Point(5, 5),
            cv::Scalar(0, 255, 0));  // green
        cv::line(img1_2_matches_temp, img1_kp_pt, img2_kp_pt, cv::Scalar(0, 255, 0));
    }
    cv::imshow("image 1 & 2 matches", img1_2_matches_temp);
    cv::waitKey(0);
    cv::destroyAllWindows();


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

    // distCoeffs.at<float>(0,0) = 0;   // rvec: [0.7220963272379515; 0.4954686910202425; -1.216350494376585]
    // distCoeffs.at<float>(1,0) = 0;   // tvec: [1.862240918769173; 0; 0.09360853660961244]
    // distCoeffs.at<float>(2,0) = 0;
    // distCoeffs.at<float>(3,0) = 0;
    distCoeffs.at<float>(0,0) = 0.14669700865145466;    // rvec: [0.8978477333458742; 0.09755148042756377; -0.6906326389118941]
    distCoeffs.at<float>(1,0) = -0.2735315348568459;    // tvec: [0.5493038210961142; -0.637639893508527; 6.432556693713797]
    distCoeffs.at<float>(2,0) = 0.007300675413449662;
    distCoeffs.at<float>(3,0) = -0.003734002028256388;

    std::vector<cv::Point3f> keypoints_3d_cv;
    for (int i = 0 ; i < keypoints_3d.size(); i++) {
        cv::Point3f point;
        point.x = keypoints_3d[i].x();
        point.y = keypoints_3d[i].y();
        point.z = keypoints_3d[i].z();

        keypoints_3d_cv.push_back(point);
    }

    std::vector<cv::Point3f> object_points(keypoints_3d_cv.begin(), keypoints_3d_cv.begin() + 4);
    std::vector<cv::Point2f> image_points(img2_left_kp_pts.begin(), img2_left_kp_pts.begin() + 4);
    cv::solvePnP(object_points,
                image_points,
                cameraMatrix, distCoeffs,
                rvec, tvec);

    std::cout << "rvec: " << rvec << std::endl;
    std::cout << "tvec: " << tvec << std::endl;

    // GTSAM optimize


    return 0;
}