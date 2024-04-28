#include <Eigen/Dense>

#include <opencv4/opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <pangolin/pangolin.h>

#include <iostream>
#include <fstream>


void triangulate2(cv::Mat intrinsic, std::vector<cv::Point2f> img0_kp_pts, std::vector<cv::Point2f> img1_kp_pts, Eigen::Isometry3d &cam1_pose, /*const cv::Mat &mask,*/ std::vector<Eigen::Vector3d> &landmarks) {
    Eigen::Matrix3d camera_intrinsic;
    cv::cv2eigen(intrinsic, camera_intrinsic);
    Eigen::MatrixXd prev_proj = Eigen::MatrixXd::Identity(3, 4);
    Eigen::MatrixXd curr_proj = Eigen::MatrixXd::Identity(3, 4);

    prev_proj = camera_intrinsic * prev_proj;
    curr_proj = camera_intrinsic * curr_proj * cam1_pose.inverse().matrix();

    for (int i = 0; i < img0_kp_pts.size(); i++) {
        // if (mask.at<unsigned char>(i) != 1) {
        //     continue;
        // }

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

int triangulate2_count(cv::Mat intrinsic, std::vector<cv::Point2f> img0_kp_pts, std::vector<cv::Point2f> img1_kp_pts, Eigen::Isometry3d &cam1_pose/*, const cv::Mat &mask*/ /*, std::vector<Eigen::Vector3d> &landmarks*/) {
    Eigen::Matrix3d camera_intrinsic;
    cv::cv2eigen(intrinsic, camera_intrinsic);
    Eigen::MatrixXd prev_proj = Eigen::MatrixXd::Identity(3, 4);
    Eigen::MatrixXd curr_proj = Eigen::MatrixXd::Identity(3, 4);

    prev_proj = camera_intrinsic * prev_proj;
    curr_proj = camera_intrinsic * curr_proj * cam1_pose.inverse().matrix();

    int positive_cnt = 0;
    for (int i = 0; i < img0_kp_pts.size(); i++) {
        // if (mask.at<unsigned char>(i) != 1) {
        //     continue;
        // }

        Eigen::Matrix4d A;
        A.row(0) = img0_kp_pts[i].x * prev_proj.row(2) - prev_proj.row(0);
        A.row(1) = img0_kp_pts[i].y * prev_proj.row(2) - prev_proj.row(1);
        A.row(2) = img1_kp_pts[i].x * curr_proj.row(2) - curr_proj.row(0);
        A.row(3) = img1_kp_pts[i].y * curr_proj.row(2) - curr_proj.row(1);

        Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector4d point_3d_homo = svd.matrixV().col(3);
        Eigen::Vector3d point_3d = point_3d_homo.head(3) / point_3d_homo[3];
        // landmarks.push_back(point_3d);

        Eigen::Vector4d cam1_point_3d_homo = cam1_pose.inverse() * point_3d_homo;
        Eigen::Vector3d cam1_point_3d = cam1_point_3d_homo.head(3) / cam1_point_3d_homo[3];
        // std::cout << "landmark(world) z: " << point_3d.z() << ", (camera) z: " << cam1_point_3d.z() << std::endl;
        if (point_3d.z() > 0 && cam1_point_3d.z() > 0 && point_3d.z() < 70) {
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

    // cv::imwrite("images/find_fundamental_matrix/frame" + std::to_string(p_id) + "_" + std::to_string(num_matches) + "_proj.png", image0_copy);
    // cv::imwrite("images/find_fundamental_matrix/frame" + std::to_string(c_id) + "_" + std::to_string(num_matches) + "_proj.png", image1_copy);

    cv::Mat image_concat;
    cv::vconcat(image0_copy, image1_copy, image_concat);
    cv::imwrite("images/find_fundamental_matrix/frame" + std::to_string(p_id) + "&frame" + std::to_string(c_id) + "_" + std::to_string(num_matches) + "_proj.png", image_concat);

    std::cout << "pose inliers: " << inlier_cnt << std::endl;
    std::cout << "inlier reprojected error: " << ((inlier_reproj_error0 + inlier_reproj_error1) / inlier_cnt) / 2 << std::endl;

    double reprojection_error = ((reproj_error0 + reproj_error1) / landmarks.size()) / 2;
    return reprojection_error;
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

        std::cout << "gt_pose[" << prev_frame_id + i << "]: "
                    << r11 << " " << r12 << " " << r13 << " " << t1
                    << r21 << " " << r22 << " " << r23 << " " << t2
                    << r31 << " " << r32 << " " << r33 << " " << t3 << std::endl;

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
    pangolin::CreateWindowAndBind("find Fundamental matrix", 1024, 768);
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
    pangolin::CreateWindowAndBind("find Fundamental matrix2", 1024, 768);
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


int main() {
    std::string image0_file = "vo_patch/source_frames/000000.png";
    std::string image1_file = "vo_patch/source_frames/000001.png";
    cv::Mat image0 = cv::imread(image0_file, cv::IMREAD_GRAYSCALE);
    cv::Mat image1 = cv::imread(image1_file, cv::IMREAD_GRAYSCALE);

    // feature matches
    std::vector<cv::Point2f> image0_kp_pts_test = {cv::Point2f(280, 148), cv::Point2f(280, 184), cv::Point2f(352, 149), cv::Point2f(344, 205), cv::Point2f(315, 223), cv::Point2f(392, 216), cv::Point2f(487, 191), cv::Point2f(436, 249), cv::Point2f(499, 263), cv::Point2f(551, 254), cv::Point2f(658, 149), cv::Point2f(655, 173), cv::Point2f(708, 155), cv::Point2f(725, 168), cv::Point2f(771, 157), cv::Point2f(813, 134), cv::Point2f(808, 126), cv::Point2f(870, 155), cv::Point2f(901, 129), cv::Point2f(917, 111), cv::Point2f(937, 126), cv::Point2f(916, 244)};
    std::vector<cv::Point2f> image1_kp_pts_test = {cv::Point2f(275, 150), cv::Point2f(275, 186), cv::Point2f(350, 151), cv::Point2f(340, 208), cv::Point2f(310, 226), cv::Point2f(387, 219), cv::Point2f(488, 193), cv::Point2f(433, 254), cv::Point2f(496, 269), cv::Point2f(551, 259), cv::Point2f(661, 150), cv::Point2f(658, 174), cv::Point2f(713, 155), cv::Point2f(731, 168), cv::Point2f(779, 158), cv::Point2f(822, 133), cv::Point2f(816, 125), cv::Point2f(881, 155), cv::Point2f(912, 128), cv::Point2f(929, 109), cv::Point2f(949, 125), cv::Point2f(951, 251)};

    std::vector<cv::Point2f> image0_kp_pts_temp;
    std::vector<cv::Point2f> image1_kp_pts_temp;

    image0_kp_pts_temp = {image0_kp_pts_test[0], image0_kp_pts_test[1], image0_kp_pts_test[3], image0_kp_pts_test[5], image0_kp_pts_test[7], image0_kp_pts_test[10], image0_kp_pts_test[13], image0_kp_pts_test[15], image0_kp_pts_test[18], image0_kp_pts_test[21]};
    image1_kp_pts_temp = {image1_kp_pts_test[0], image1_kp_pts_test[1], image1_kp_pts_test[3], image1_kp_pts_test[5], image1_kp_pts_test[7], image1_kp_pts_test[10], image1_kp_pts_test[13], image1_kp_pts_test[15], image1_kp_pts_test[18], image1_kp_pts_test[21]};

    // matrix W
    // int num_keypoints = image0_kp_pts_test.size();
    int num_keypoints = image0_kp_pts_temp.size();
    Eigen::MatrixXf W = Eigen::MatrixXf::Zero(num_keypoints, 9);
    // std::cout << "Matrix W:\n" << W.matrix() << std::endl;

    // concat points
    Eigen::MatrixXf P(3, num_keypoints), P_prime(3, num_keypoints);
    for (int i = 0; i < num_keypoints; i++) {
        // Eigen::Vector3f p(image0_kp_pts_test[i].x, image0_kp_pts_test[i].y, 1);
        // Eigen::Vector3f p_prime(image1_kp_pts_test[i].x, image1_kp_pts_test[i].y, 1);
        Eigen::Vector3f p(image0_kp_pts_temp[i].x, image0_kp_pts_temp[i].y, 1);
        Eigen::Vector3f p_prime(image1_kp_pts_temp[i].x, image1_kp_pts_temp[i].y, 1);

        P.block<3, 1>(0, i) = p;
        P_prime.block<3, 1>(0, i) = p_prime;
    }

    // mean, SSE
    Eigen::Vector3f P_mean = P.rowwise().mean();
    Eigen::Vector3f P_prime_mean = P_prime.rowwise().mean();
    std::cout << "P_mean: \n" << P_mean << std::endl;
    std::cout << "P_prime_mean: \n" << P_prime_mean << std::endl;

    Eigen::Vector3f P_sse = Eigen::Vector3f::Zero();
    Eigen::Vector3f P_prime_sse = Eigen::Vector3f::Zero();
    for (int i = 0; i < num_keypoints; i++) {
        Eigen::Vector3f P_e = P.col(i) - P_mean;
        Eigen::Vector3f P_prime_e = P_prime.col(i) - P_prime_mean;

        P_sse += (P_e.array() * P_e.array()).matrix();
        P_prime_sse += (P_prime_e.array() * P_prime_e.array()).matrix();
    }
    std::cout << "P_sse: \n" << P_sse << std::endl;
    std::cout << "P_prime_sse: \n" << P_prime_sse << std::endl;

    // scale
    Eigen::Vector3f P_scale = Eigen::sqrt(2*num_keypoints / P_sse.array());
    Eigen::Vector3f P_prime_scale = Eigen::sqrt(2*num_keypoints / P_prime_sse.array());
    std::cout << "P_scale: \n" << P_scale << std::endl;
    std::cout << "P_prime_scale: \n" << P_prime_scale << std::endl;

    // transform
    Eigen::Matrix3f P_transform;
    Eigen::Matrix3f P_prime_transform;
    P_transform << P_scale(0), 0, P_scale(0) * -P_mean(0),
                    0, P_scale(1), P_scale(1) * -P_mean(1),
                    0, 0, 1;
    P_prime_transform << P_prime_scale(0), 0, P_prime_scale(0) * -P_prime_mean(0),
                    0, P_prime_scale(1), P_prime_scale(1) * -P_prime_mean(1),
                    0, 0, 1;
    std::cout << "P_transform: \n" << P_transform.matrix() << std::endl;
    std::cout << "P_prime_transform: \n" << P_prime_transform.matrix() << std::endl;

    // Normalize keypoints
    for (int i = 0; i < num_keypoints; i++) {
        Eigen::Vector3f p = P_transform * P.col(i);
        Eigen::Vector3f p_prime = P_prime_transform * P_prime.col(i);

        Eigen::Matrix3f p_p_prime = p * p_prime.transpose();
        Eigen::Matrix<float, 1, 9> p_p_prime_vec = p_p_prime.reshaped().transpose();
        // std::cout << "pq[" << i << "] : " << pq_vec << std::endl;

        W.block<1, 9>(i, 0) = p_p_prime_vec;
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
    fundamental_mat = fundamental_U * fundamental_S * fundamental_V.transpose();
    std::cout << "Fundamental Matrix: \n" << fundamental_mat.matrix() << std::endl;

    // Re-transform
    fundamental_mat = P_transform.transpose() * fundamental_mat * P_prime_transform;
    fundamental_mat = fundamental_mat.array() / fundamental_mat(2, 2);
    std::cout << "Fundamental Matrix: \n" << fundamental_mat.matrix() << std::endl;

    // draw epilines
    cv::Mat image0_bgr, image1_bgr;
    cv::cvtColor(image0, image0_bgr, cv::COLOR_GRAY2BGR);
    cv::cvtColor(image1, image1_bgr, cv::COLOR_GRAY2BGR);

    num_keypoints = image0_kp_pts_test.size();
    for (int i = 0; i < num_keypoints; i++) {
        Eigen::Vector3f p(image0_kp_pts_test[i].x, image0_kp_pts_test[i].y, 1);
        Eigen::Vector3f p_prime(image1_kp_pts_test[i].x, image1_kp_pts_test[i].y, 1);
        Eigen::Vector3f epiline0 = fundamental_mat * p_prime;
        Eigen::Vector3f epiline1 = fundamental_mat.transpose() * p;
        // std::cout << "[" << i << "]" << std::endl;
        // std::cout << "p: " << p.transpose() << std::endl;
        // std::cout << "p-prime: " << p_prime.transpose() << std::endl;
        // std::cout << "epiline0: " << epiline0.transpose() << std::endl;
        // std::cout << "epiline1: " << epiline1.transpose() << std::endl;

        // image0
        double img0_x_0 = 0;
        double img0_y_0 = -1 * (epiline0(0) * img0_x_0 + epiline0(2)) / epiline0(1);
        double img0_x_1 = image0.cols;
        double img0_y_1 = -1 * (epiline0(0) * img0_x_1 + epiline0(2)) / epiline0(1);
        // std::cout << "(" << img0_x_0 << ", " << img0_y_0 << "), " << "(" << img0_x_1 << ", " << img0_y_1 << ")" << std::endl;

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
        // std::cout << "(" << img1_x_0 << ", " << img1_y_0 << "), " << "(" << img1_x_1 << ", " << img1_y_1 << ")" << std::endl;

        // draw keypoint
        cv::rectangle(image1_bgr,
                        image1_kp_pts_test[i] - cv::Point2f(5, 5),
                        image1_kp_pts_test[i] + cv::Point2f(5, 5),
                        cv::Scalar(0, 255, 0));  // green

        // draw epipline
        cv::line(image1_bgr, cv::Point2d(img1_x_0, img1_y_0), cv::Point2d(img1_x_1, img1_y_1), cv::Scalar(0, 0, 255));

        // image show
        cv::imshow("image0", image0_bgr);
        cv::imshow("image1", image1_bgr);
        cv::waitKey(50);
    }
    while (cv::waitKey(0) != 27) {}
    cv::destroyAllWindows();

    // std::cout << "\n============OpenCV Fundamental matrix============" << std::endl;
    // cv::Mat mask;
    // cv::Mat fundamental_mat_cv = cv::findFundamentalMat(image0_kp_pts_test, image1_kp_pts_test, mask, cv::RANSAC, 3.0, 0.99);
    // std::cout << "fundamental matrix:\n" << fundamental_mat_cv << std::endl;

    // int mask_cnt = 0;
    // for (int i = 0; i < mask.rows; i++) {
    //     if (mask.at<unsigned char>(i) != 1) {
    //         std::cout << i << std::endl;
    //     }
    //     else {
    //         mask_cnt++;
    //     }
    // }
    // std::cout << "mask count: " << mask_cnt << std::endl;

    // cv::Mat lines0, lines1;
    // cv::computeCorrespondEpilines(image1_kp_pts_test, 2, fundamental_mat_cv, lines0);
    // cv::computeCorrespondEpilines(image0_kp_pts_test, 1, fundamental_mat_cv, lines1);

    // Eigen::Matrix3f fundamental_mat_cv_eigen;
    // cv::cv2eigen(fundamental_mat_cv, fundamental_mat_cv_eigen);

    // for (int i = 0; i < num_keypoints; i++) {
    //     Eigen::Vector3f p(image0_kp_pts_test[i].x, image0_kp_pts_test[i].y, 1);
    //     Eigen::Vector3f p_prime(image1_kp_pts_test[i].x, image1_kp_pts_test[i].y, 1);
    //     Eigen::Vector3f epiline0 = fundamental_mat_cv_eigen * p_prime;
    //     Eigen::Vector3f epiline1 = fundamental_mat_cv_eigen.transpose() * p;

    //     // image0
    //     double img0_x_0 = 0;
    //     double img0_y_0 = -1 * (epiline0(0) * img0_x_0 + epiline0(2)) / epiline0(1);
    //     double img0_x_1 = image0.cols;
    //     double img0_y_1 = -1 * (epiline0(0) * img0_x_1 + epiline0(2)) / epiline0(1);
    //     // std::cout << "(" << img0_x_0 << ", " << img0_y_0 << "), " << "(" << img0_x_1 << ", " << img0_y_1 << ")" << std::endl;

    //     // draw keypoint
    //     cv::rectangle(image0_bgr,
    //                     image0_kp_pts_test[i] - cv::Point2f(5, 5),
    //                     image0_kp_pts_test[i] + cv::Point2f(5, 5),
    //                     cv::Scalar(255, 0, 0));  // green

    //     // draw epipline
    //     cv::line(image0_bgr, cv::Point2d(img0_x_0, img0_y_0), cv::Point2d(img0_x_1, img0_y_1), cv::Scalar(255, 0, 0));
    //     cv::imshow("image0_cv", image0_bgr);
    //     cv::waitKey(50);
    // }



    // Essential matrix
    // KITTI
    float fx = 718.856;
    float fy = 718.856;
    float s = 0.0;
    float cx = 607.1928;
    float cy = 185.2157;

    Eigen::Matrix3f intrinsic;
    intrinsic << fx, s, cx,
                0, fy, cy,
                0, 0, 1;
    std::cout << "intrinsic:\n" << intrinsic << std::endl;
    cv::Mat intrinsic_cv;
    cv::eigen2cv(intrinsic, intrinsic_cv);

    Eigen::Matrix3f essential_mat = intrinsic.transpose() * fundamental_mat * intrinsic;
    // Eigen::Matrix3f essential_mat = intrinsic.transpose() * fundamental_mat_cv_eigen * intrinsic;
    std::cout << "essential matrix:\n" << essential_mat << std::endl;

    // SVD
    Eigen::JacobiSVD<Eigen::Matrix3f> e_svd(essential_mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f essential_U = e_svd.matrixU();
    Eigen::Matrix3f essential_S = Eigen::Vector3f(1, 1, 0).asDiagonal();
    Eigen::Matrix3f essential_V = e_svd.matrixV();
    essential_mat = essential_U * essential_S * essential_V.transpose();
    std::cout << "essential matrix:\n" << essential_mat << std::endl;

    cv::Mat essential_mat_cv(cv::Size(3, 3), CV_32FC1);
    cv::eigen2cv(essential_mat, essential_mat_cv);


    // Motion estimation
    std::vector<Eigen::Isometry3d> poses, poses_answer;

    // // recover Pose
    // cv::Mat R0, t0;
    // cv::Mat mask = cv::Mat::ones(num_keypoints, 1, CV_8UC1);
    // int cv_recPos_inlier = cv::recoverPose(essential_mat_cv, image0_kp_pts_test, image1_kp_pts_test, intrinsic_cv, R0, t0, mask);
    // std::cout << "opencv recoverPose() info\n inliers: " << cv_recPos_inlier << std::endl;
    // std::cout << " rotation: \n" << R0 << std::endl;
    // std::cout << " translation: \n" << t0 << std::endl;

    // Eigen::Matrix3d rotation_mat;
    // Eigen::Vector3d translation_mat;
    // Eigen::Isometry3d relative_pose;
    // cv::cv2eigen(R0, rotation_mat);
    // cv::cv2eigen(t0, translation_mat);

    // relative_pose.linear() = rotation_mat;
    // relative_pose.translation() = translation_mat;

    // Decompose essential matrix
    std::cout << "decomposing essential matrix" << std::endl;
    cv::Mat R1, R2, t;
    cv::Mat mask = cv::Mat::ones(num_keypoints, 1, CV_8UC1);
    cv::decomposeEssentialMat(essential_mat_cv, R1, R2, t);

    Eigen::Matrix3d rotation_mat;
    Eigen::Vector3d translation_mat;
    Eigen::Isometry3d relative_pose;

    std::vector<Eigen::Isometry3d> rel_pose_candidates(4, Eigen::Isometry3d::Identity());
    int valid_point_cnts[4];
    for (int k = 0; k < 4; k++) {
        if (k == 0) {
            cv::cv2eigen(R1, rotation_mat);
            cv::cv2eigen(t, translation_mat);
        }
        else if (k == 1) {
            cv::cv2eigen(R2, rotation_mat);
            cv::cv2eigen(t, translation_mat);
        }
        else if (k == 2) {
            cv::cv2eigen(R1, rotation_mat);
            cv::cv2eigen(-t, translation_mat);
        }
        else if (k == 3) {
            cv::cv2eigen(R2, rotation_mat);
            cv::cv2eigen(-t, translation_mat);
        }
        rel_pose_candidates[k].linear() = rotation_mat;
        rel_pose_candidates[k].translation() = translation_mat;
        valid_point_cnts[k] = triangulate2_count(intrinsic_cv, image0_kp_pts_test, image1_kp_pts_test, rel_pose_candidates[k]);
        poses.push_back(rel_pose_candidates[k]);
    }
    int max_cnt = 0, max_idx = 0;
    for (int k = 0; k < 4; k++) {
        std::cout << "cnt[" << k << "]: " << valid_point_cnts[k] << std::endl;
        if (valid_point_cnts[k] > max_cnt) {
            max_cnt = valid_point_cnts[k];
            max_idx = k;
        }
    }
    std::cout << "max idx: " << max_idx << std::endl;
    relative_pose = rel_pose_candidates[max_idx];
    std::cout << "relative pose:\n " << relative_pose.matrix() << std::endl;
    poses_answer = {Eigen::Isometry3d::Identity(), relative_pose};

    // triangulate
    std::cout << "triangulating..." << std::endl;
    std::vector<Eigen::Vector3d> landmarks;
    triangulate2(intrinsic_cv, image0_kp_pts_test, image1_kp_pts_test, relative_pose, landmarks);

    // reprojection error
    std::cout << "reprojection error..." << std::endl;
    double reprojection_error = calcReprojectionError(intrinsic_cv,
                                                        0, image0, image0_kp_pts_test,
                                                        1, image1, image1_kp_pts_test,
                                                        mask, relative_pose, landmarks, image0_kp_pts_test.size());
    std::cout << "reprojection error: " << reprojection_error << std::endl;


    // Display
    std::vector<Eigen::Isometry3d> gt_poses, aligned_poses;
    loadGT("/home/kodogyu/Datasets/KITTI/dataset/poses/00.txt", 0, gt_poses);

    displayPoses(gt_poses, poses, aligned_poses);
    displayFramesAndLandmarks(poses_answer, landmarks);

    return 0;
}