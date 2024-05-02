#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

#include <iostream>

void visualize3d(std::vector<cv::Point3d> points3D, cv::Mat camera_pose) {
    cv::viz::Viz3d window; //creating a Viz window

    //Displaying the Coordinate Origin (0,0,0)
    window.showWidget("coordinate", cv::viz::WCoordinateSystem(10));

    // camera pose
    cv::Mat axis_O, axis_X, axis_Y, axis_Z;
    axis_O = camera_pose * cv::Vec4d(0, 0, 0, 1);
    axis_X = camera_pose * cv::Vec4d(1, 0, 0, 1);
    axis_Y = camera_pose * cv::Vec4d(0, 1, 0, 1);
    axis_Z = camera_pose * cv::Vec4d(0, 0, 1, 1);
    cv::Point3d pAxis_O(axis_O), pAxis_X(axis_X), pAxis_Y(axis_Y), pAxis_Z(axis_Z);
    std::cout << "axis_O: " << pAxis_O << std::endl;
    std::cout << "axis_X: " << pAxis_X << std::endl;
    std::cout << "axis_Y: " << pAxis_Y << std::endl;
    std::cout << "axis_Z: " << pAxis_Z << std::endl;

    cv::viz::WLine wAxis_X(pAxis_O, pAxis_X * 10, cv::viz::Color::red());
    cv::viz::WLine wAxis_Y(pAxis_O, pAxis_Y * 10, cv::viz::Color::blue());
    cv::viz::WLine wAxis_Z(pAxis_O, pAxis_Z * 10, cv::viz::Color::green());
    wAxis_X.setRenderingProperty(cv::viz::LINE_WIDTH, 4.0);
    wAxis_Y.setRenderingProperty(cv::viz::LINE_WIDTH, 4.0);
    wAxis_Z.setRenderingProperty(cv::viz::LINE_WIDTH, 4.0);
    window.showWidget("cam_axis_x", wAxis_X);
    window.showWidget("cam_axis_y", wAxis_Y);
    window.showWidget("cam_axis_z", wAxis_Z);

    std::vector<cv::Point3d> points_test;
    points_test.push_back(cv::Point3d(0.0, 0.0, 0.0));
    points_test.push_back(cv::Point3d(-10.1, -10.1, 10.1));
    points_test.push_back(cv::Point3d(-50.2, -50.2, 50.2));
    points_test.push_back(cv::Point3d(-100, -100, 100));
    points_test.push_back(cv::Point3d(-200, -200, 200));
    points_test.push_back(cv::Point3d(-300, -300, 300));
    cv::viz::WCloud wPoints_test(points_test, cv::viz::Color::white());
    wPoints_test.setRenderingProperty(cv::viz::POINT_SIZE, 10);
    window.showWidget("points test", wPoints_test);

    // Point cloud widget
    cv::viz::WCloud points3D_widget(points3D, cv::viz::Color::blue());
    points3D_widget.setRenderingProperty(cv::viz::POINT_SIZE, 10);

    //Displaying the 3D points in green
    window.showWidget("points", points3D_widget);
    window.spin();
}

cv::Mat decomposeEssentialMat(cv::Mat essential_mat) {
    cv::Mat U, S, VT;
    cv::SVD::compute(essential_mat, S, U, VT);

    // OpenCV decomposeEssentialMat()
    cv::Mat W = (cv::Mat_<double>(3, 3) << 0, 1, 0,
                                            -1, 0, 0,
                                            0, 0, 1);

    // hand function.
    // cv::Mat W = (cv::Mat_<double>(3, 3) << 0, -1, 0,
    //                                         1, 0, 0,
    //                                         0, 0, 1);

    std::vector<cv::Mat> pose_candidates(4);
    cv::hconcat(std::vector<cv::Mat>{U * W * VT, U.col(2)}, pose_candidates[0]);
    cv::hconcat(std::vector<cv::Mat>{U * W * VT, -U.col(2)}, pose_candidates[1]);
    cv::hconcat(std::vector<cv::Mat>{U * W.t() * VT, U.col(2)}, pose_candidates[2]);
    cv::hconcat(std::vector<cv::Mat>{U * W.t() * VT, -U.col(2)}, pose_candidates[3]);

    for (int i = 0; i < pose_candidates.size(); i++) {
        std::cout << "pose_candidate [" << i << "]: \n" << pose_candidates[i] << std::endl;
    }

    return pose_candidates[3];
}

int main() {
    std::cout << CV_VERSION << std::endl;

    // std::string prev_image_file = "/home/kodogyu/github_repos/sfm/data/nutellar2/nutella13.jpg";
    // std::string curr_image_file = "/home/kodogyu/github_repos/sfm/data/nutellar2/nutella14.jpg";
    std::string prev_image_file = "/home/kodogyu/playground/cpp/vo_patch/source_frames/000000.png";
    std::string curr_image_file = "/home/kodogyu/playground/cpp/vo_patch/source_frames/000001.png";

    // read images
    cv::Mat prev_image_color = cv::imread(prev_image_file, cv::IMREAD_COLOR);
    cv::Mat curr_image_color = cv::imread(curr_image_file, cv::IMREAD_COLOR);

    cv::Mat prev_image_gray, curr_image_gray;
    cv::cvtColor(prev_image_color, prev_image_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(curr_image_color, curr_image_gray, cv::COLOR_BGR2GRAY);

    std::cout << "prev_image size: " << prev_image_gray.size << std::endl;
    std::cout << "curr_image size: " << curr_image_gray.size << std::endl;

    // feature extraction
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(0, 3, 0.04, 10.0, 1.6);
    // cv::Ptr<cv::ORB> orb = cv::ORB::create();
    std::vector<cv::KeyPoint> prev_image_keypoints, curr_image_keypoints;

    cv::Mat prev_image_descriptors;
    cv::Mat curr_image_descriptors;
    sift->detectAndCompute(prev_image_gray, cv::Mat(), prev_image_keypoints, prev_image_descriptors);
    sift->detectAndCompute(curr_image_gray, cv::Mat(), curr_image_keypoints, curr_image_descriptors);
    // orb->detectAndCompute(prev_image, cv::Mat(), prev_image_keypoints, prev_image_descriptors);
    // orb->detectAndCompute(curr_image, cv::Mat(), curr_image_keypoints, curr_image_descriptors);
    std::cout << "prev image keypoint size: " << prev_image_keypoints.size() << std::endl;
    std::cout << "curr image keypoint size: " << curr_image_keypoints.size() << std::endl;
    std::cout << "prev image desciptor size: " << prev_image_descriptors.size << std::endl;
    std::cout << "curr image desciptor size: " << curr_image_descriptors.size << std::endl;


    // draw features

    // cv::Mat kp_image;
    // cv::cvtColor(prev_image_gray, kp_image, cv::COLOR_GRAY2BGR);
    // for (int i = 0; i < 10; i++) {
    //     cv::KeyPoint kp = prev_image_keypoints[i];
    //     cv::circle(kp_image, kp.pt, 10, cv::Scalar(0, 255, 0), 3);
    // }
    // cv::Mat resized_kp_image;
    // cv::resize(kp_image, resized_kp_image, kp_image.size()/2);
    // cv::imshow("kp_image", resized_kp_image);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    std::vector<cv::KeyPoint> prev_image_keypoints_100, curr_image_keypoints_100;
    prev_image_keypoints_100.assign(prev_image_keypoints.begin(), prev_image_keypoints.begin() + 100);
    curr_image_keypoints_100.assign(curr_image_keypoints.begin(), curr_image_keypoints.begin() + 100);

    cv::Mat prev_kp_image, curr_kp_image;
    cv::drawKeypoints(prev_image_color, prev_image_keypoints_100, prev_kp_image);
    cv::drawKeypoints(curr_image_color, curr_image_keypoints_100, curr_kp_image);
    cv::imshow("prev_kp_image", prev_kp_image);
    cv::imshow("curr_kp_image", curr_kp_image);
    cv::waitKey(0);
    cv::destroyAllWindows();



    // feature matching
    cv::Ptr<cv::BFMatcher> bf_matcher = cv::BFMatcher::create(cv::NORM_L2);
    // cv::Ptr<cv::BFMatcher> bf_matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);

    std::vector<std::vector<cv::DMatch>> matches;
    bf_matcher->knnMatch(prev_image_descriptors, curr_image_descriptors, matches, 2);  // prev -> curr matches
    std::cout << "matches length: " << matches.size() << std::endl;

    // good matches
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < matches.size(); i++) {
        if (matches[i][0].distance < matches[i][1].distance * 0.8) {
            good_matches.push_back(matches[i][0]);
        }
        if (i < 10){
            std::cout << matches[i][0].distance << ", " << matches[i][1].distance << std::endl;
        }
    }
    std::cout << "good matches size: " << good_matches.size() << std::endl;

    // essential matrix
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

    cv::Mat mask;
    // double skew = 0.0215878;
    // cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << 3092.8, skew, 2016,
    //                                                 0, 3092.8, 1512,
    //                                                 0, 0, 1);

// # # KITTI
// # fx: 718.856
// # fy: 718.856
// # s: 0.0
// # cx: 607.1928
// # cy: 185.2157
    cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << 718.856, 0, 607.1928,
                                                    0, 718.856, 185.2157,
                                                    0, 0, 1);
    cv::Mat essential_mat = cv::findEssentialMat(image0_kp_pts, image1_kp_pts, intrinsic, cv::RANSAC, 0.999, 1.0, 500, mask);
    std::cout << "essential matrix: \n" << essential_mat << std::endl;

    // relative pose
    cv::Mat relative_pose = decomposeEssentialMat(essential_mat);
    // cv::Mat R1, R2, t1;
    // cv::decomposeEssentialMat(essential_mat, R1, R2, t1);
    // std::cout << "=====OpenCV decomposition=====" << std::endl;
    // std::cout << "R1:\n" << R1 << std::endl;
    // std::cout << "R2:\n" << R2 << std::endl;
    // std::cout << "t:\n" << t1 << std::endl;

    // cv::Mat R, t;
    // cv::recoverPose(essential_mat, image0_kp_pts, image1_kp_pts, intrinsic, R, t, mask);
    // cv::Mat relative_pose1;
    // cv::hconcat(std::vector<cv::Mat>{R, -t}, relative_pose1);
    // std::cout << "R:\n" << R << std::endl;
    // std::cout << "t:\n" << t << std::endl;
    // std::cout << "relative pose1:\n" << relative_pose1 << std::endl;
    // std::cout << "intrinsic:\n" << intrinsic << std::endl;

    cv::Mat camera_matrix = intrinsic * relative_pose;
    // cv::Mat camera_matrix1 = intrinsic * relative_pose1;
    std::cout << "camera Matrix:\n" << camera_matrix << std::endl;
    // std::cout << "camera Matrix1:\n" << camera_matrix1 << std::endl;

    // needed points
    std::vector<cv::Point2f> prev_image_valid_pts, curr_image_valid_pts;
    for (int i = 0; i < mask.rows; i++) {
        if (mask.at<unsigned char>(i) == 1) {
            prev_image_valid_pts.push_back(image0_kp_pts[i]);
            curr_image_valid_pts.push_back(image1_kp_pts[i]);
        }
    }

    // triangulate points
    // cv::Mat prev_projection_mat = (cv::Mat_<double>(3, 4) << 1, 0, 0, 0,
    //                                                         0, 1, 0, 0,
    //                                                         0, 0, 1, 0);
    cv::Mat prev_projection_mat = intrinsic * cv::Mat::eye(cv::Size(4, 3), CV_64F);
    // cv::Mat curr_projection_mat = camera_matrix1.clone();
    cv::Mat curr_projection_mat = camera_matrix.clone();

    cv::Mat points4D;
    cv::triangulatePoints(prev_projection_mat, curr_projection_mat, prev_image_valid_pts, curr_image_valid_pts, points4D);

    std::vector<cv::Point3d> points3D;
    for (int i = 0; i < points4D.cols; i++) {
        cv::Mat point3D_mat = points4D.col(i).rowRange(0, 3) / points4D.at<double>(3, i);
        // cv::Mat point3D_mat = points4D.col(i).rowRange(0, 3);
        cv::Point3d point3d(point3D_mat);

        points3D.push_back(point3d);
    }

    for (int i = 0; i < 10; i++) {
            std::cout << " point 3d: " << points3D[i] << std::endl;
    }
    visualize3d(points3D, relative_pose);

    // reproject
    std::vector<cv::Point2d> projected_points, curr_projected_points;
    cv::Mat rvec;
    cv::Rodrigues(cv::Mat::eye(cv::Size(3, 3), CV_64FC1), rvec);
    cv::projectPoints(points3D, rvec, cv::Vec3d::zeros(), intrinsic, cv::Mat(), projected_points);

    cv::Mat r_vec;
    // cv::Rodrigues(R, r_vec);
    cv::Rodrigues(relative_pose.rowRange(0, 3).colRange(0, 3), r_vec);
    std::cout << "rotation matrix:\n" << relative_pose.rowRange(0, 3).colRange(0, 3) << std::endl;
    // cv::projectPoints(points3D, r_vec, t, intrinsic, cv::Mat(), curr_projected_points);
    cv::projectPoints(points3D, r_vec, relative_pose.col(3), intrinsic, cv::Mat(), curr_projected_points);
    std::cout << "projected" << std::endl;

    for (int i = 0; i < 300; i++) {
        cv::Point2f pt = projected_points[i];
        cv::Point2f pt2 = curr_projected_points[i];

        cv::circle(prev_image_color, pt, 3, cv::Scalar(0, 255, 0), 1);
        cv::circle(curr_image_color, pt2, 3, cv::Scalar(0, 255, 0), 1);

        if (i < 10) {
            std::cout << pt2.x << ", " << pt2.y << std::endl;
        }
    }
    cv::imshow("projected", prev_image_color);
    cv::imshow("projected2", curr_image_color);
    cv::waitKey(0);

    return 0;
}