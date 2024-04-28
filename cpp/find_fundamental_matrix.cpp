#include <Eigen/Dense>
#include <opencv4/opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::string image0_file = "vo_patch/source_frames/000000.png";
    std::string image1_file = "vo_patch/source_frames/000001.png";
    cv::Mat image0 = cv::imread(image0_file, cv::IMREAD_GRAYSCALE);
    cv::Mat image1 = cv::imread(image1_file, cv::IMREAD_GRAYSCALE);

    // feature matches
    std::vector<cv::Point2f> image0_kp_pts_test = {cv::Point2f(280, 148), cv::Point2f(280, 184), cv::Point2f(352, 149), cv::Point2f(344, 205), cv::Point2f(315, 223), cv::Point2f(392, 216), cv::Point2f(487, 191), cv::Point2f(436, 249), cv::Point2f(499, 263), cv::Point2f(551, 254), cv::Point2f(658, 149), cv::Point2f(655, 173), cv::Point2f(708, 155), cv::Point2f(725, 168), cv::Point2f(771, 157), cv::Point2f(813, 134), cv::Point2f(808, 126), cv::Point2f(870, 155), cv::Point2f(901, 129), cv::Point2f(917, 111), cv::Point2f(937, 126), cv::Point2f(916, 244)};
    std::vector<cv::Point2f> image1_kp_pts_test = {cv::Point2f(275, 150), cv::Point2f(275, 186), cv::Point2f(350, 151), cv::Point2f(340, 208), cv::Point2f(310, 226), cv::Point2f(387, 219), cv::Point2f(488, 193), cv::Point2f(433, 254), cv::Point2f(496, 269), cv::Point2f(551, 259), cv::Point2f(661, 150), cv::Point2f(658, 174), cv::Point2f(713, 155), cv::Point2f(731, 168), cv::Point2f(779, 158), cv::Point2f(822, 133), cv::Point2f(816, 125), cv::Point2f(881, 155), cv::Point2f(912, 128), cv::Point2f(929, 109), cv::Point2f(949, 125), cv::Point2f(951, 251)};

    // matrix W
    int num_keypoints = image0_kp_pts_test.size();
    Eigen::MatrixXf W = Eigen::MatrixXf::Zero(num_keypoints, 9);
    // std::cout << "Matrix W:\n" << W.matrix() << std::endl;

    for (int i = 0; i < num_keypoints; i++) {
        Eigen::Vector3f p(image0_kp_pts_test[i].x, image0_kp_pts_test[i].y, 1);
        Eigen::Vector3f q(image1_kp_pts_test[i].x, image1_kp_pts_test[i].y, 1);

        Eigen::Matrix3f pq = p * q.transpose();
        Eigen::Matrix<float, 1, 9> pq_vec = pq.reshaped().transpose();
        // std::cout << "pq[" << i << "] : " << pq_vec << std::endl;

        W.block<1, 9>(i, 0) = pq_vec;
    }
    // std::cout << "Matrix W:\n" << W.matrix() << std::endl;

    // SVD
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<float, 9, 1> fundamental_vec = svd.matrixV().col(8);

    // rank 2
    std::cout << "fundamental_vec: \n" << fundamental_vec.matrix() << std::endl;
    Eigen::Matrix3f fundamental_mat = fundamental_vec.reshaped(3, 3);

    Eigen::JacobiSVD<Eigen::Matrix3f> f_svd(fundamental_mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXf fundamental_U = f_svd.matrixU();
    Eigen::MatrixXf fundamental_S = f_svd.singularValues().asDiagonal();
    Eigen::MatrixXf fundamental_V = f_svd.matrixV();

    std::cout << "fundamental_S: \n" << fundamental_S.matrix() << std::endl;
    fundamental_S(2, 2) = 0;
    std::cout << "fundamental_S: \n" << fundamental_S.matrix() << std::endl;

    // Fundamental Matrix
    fundamental_mat = fundamental_U * fundamental_S * fundamental_V;
    std::cout << "Fundamental Matrix: \n" << fundamental_mat.matrix() << std::endl;


    // draw epilines
    cv::Mat image0_bgr, image1_bgr;
    cv::cvtColor(image0, image0_bgr, cv::COLOR_GRAY2BGR);
    cv::cvtColor(image1, image1_bgr, cv::COLOR_GRAY2BGR);

    for (int i = 0; i < num_keypoints; i++) {
        Eigen::Vector3f p(image0_kp_pts_test[i].x, image0_kp_pts_test[i].y, 1);
        Eigen::Vector3f p_prime(image1_kp_pts_test[i].x, image1_kp_pts_test[i].y, 1);
        Eigen::Vector3f epiline0 = fundamental_mat * p_prime;
        Eigen::Vector3f epiline1 = fundamental_mat * p;
        std::cout << "[" << i << "]" << std::endl;
        std::cout << "p: " << p.transpose() << std::endl;
        std::cout << "p-prime: " << p_prime.transpose() << std::endl;
        std::cout << "epiline0: " << epiline0.transpose() << std::endl;
        std::cout << "epiline1: " << epiline1.transpose() << std::endl;

        // image0
        double img0_x_0 = 0;
        double img0_y_0 = -1 * (epiline0(0) * img0_x_0 + epiline0(2)) / epiline0(1);
        double img0_x_1 = image0.cols;
        double img0_y_1 = -1 * (epiline0(0) * img0_x_1 + epiline0(2)) / epiline0(1);
        std::cout << "(" << img0_x_0 << ", " << img0_y_0 << "), " << "(" << img0_x_1 << ", " << img0_y_1 << ")" << std::endl;

        // draw keypoint
        cv::rectangle(image0_bgr,
                        image0_kp_pts_test[i] - cv::Point2f(5, 5),
                        image0_kp_pts_test[i] + cv::Point2f(5, 5),
                        cv::Scalar(0, 255, 0));  // green

        // draw epipline
        cv::line(image0_bgr, cv::Point2d(img0_x_0, img0_y_0), cv::Point2d(img0_x_1, img0_y_1), cv::Scalar(0, 0, 255));


        // image1
        double img1_x_0 = 0;
        double img1_y_0 = -1 * (epiline1(0) * img1_x_0 + epiline1(2)) / epiline1(1);
        double img1_x_1 = image0.cols;
        double img1_y_1 = -1 * (epiline1(0) * img1_x_1 + epiline1(2)) / epiline1(1);
        std::cout << "(" << img1_x_0 << ", " << img1_y_0 << "), " << "(" << img1_x_1 << ", " << img1_y_1 << ")" << std::endl;

        // draw keypoint
        cv::rectangle(image1_bgr,
                        image1_kp_pts_test[i] - cv::Point2f(5, 5),
                        image1_kp_pts_test[i] + cv::Point2f(5, 5),
                        cv::Scalar(0, 255, 0));  // green

        // draw epipline
        cv::line(image1_bgr, cv::Point2d(img1_x_0, img1_y_0), cv::Point2d(img1_x_1, img1_y_1), cv::Scalar(0, 0, 255));
    }

    // image show
    cv::imshow("image0", image0_bgr);
    cv::imshow("image1", image1_bgr);
    cv::waitKey(0);
    cv::destroyAllWindows();


    return 0;
}