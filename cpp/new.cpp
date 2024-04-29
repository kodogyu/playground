#include <opencv2/opencv.hpp>

#include <iostream>

int main() {
    std::string prev_image_file = "/home/kodogyu/github_repos/sfm/data/nutellar2/nutella13.jpg";
    std::string curr_image_file = "/home/kodogyu/github_repos/sfm/data/nutellar2/nutella14.jpg";

    // read images
    cv::Mat prev_image = cv::imread(prev_image_file, cv::IMREAD_GRAYSCALE);
    cv::Mat curr_image = cv::imread(curr_image_file, cv::IMREAD_GRAYSCALE);
    std::cout << "prev_image size: " << prev_image.size << std::endl;
    std::cout << "curr_image size: " << curr_image.size << std::endl;

    // feature extraction
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    sift->save("new_file.txt");
    std::vector<cv::KeyPoint> prev_image_keypoints;
    std::vector<cv::KeyPoint> curr_image_keypoints;
    cv::Mat prev_image_descriptors;
    cv::Mat curr_image_descriptors;
    sift->detectAndCompute(prev_image, cv::Mat(), prev_image_keypoints, prev_image_descriptors);
    sift->detectAndCompute(curr_image, cv::Mat(), curr_image_keypoints, curr_image_descriptors);

    // feature matching
    cv::Ptr<cv::BFMatcher> bf_matcher = cv::BFMatcher::create();

    std::vector<std::vector<cv::DMatch>> matches;
    bf_matcher->knnMatch(prev_image_descriptors, curr_image_descriptors, matches, 2);  // prev -> curr matches
    std::cout << "match length: " << matches.size() << std::endl;

    // good matches
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < matches.size(); i++) {
        if (matches[i][0].distance < matches[i][1].distance * 0.95) {
            good_matches.push_back(matches[i][0]);
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
    cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << 3092.8, 0, 2016,
                                                    0, 3092.8, 1512,
                                                    0, 0, 1);
    cv::Mat essential_mat = cv::findEssentialMat(image0_kp_pts, image1_kp_pts, intrinsic, cv::RANSAC, 0.999, 1.0, 500, mask);
    std::cout << "essential matrix: \n" << essential_mat << std::endl;


    return 0;
}