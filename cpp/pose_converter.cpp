#include <iostream>
#include <Eigen/Dense>
#include <fstream>

int main() {
    std::string file_path = "/home/kodogyu/Datasets/TUM/rgbd_dataset_freiburg3_checkerboard_large/groundtruth.txt";
    std::cout << "pose file path: " << file_path << std::endl;

    std::ifstream gt_poses_file(file_path);
    int no_frame;
    double time_stamp;
    double r11, r12, r13, r21, r22, r23, r31, r32, r33, t1, t2, t3;
    double qw, qx, qy, qz;
    std::string line;
    std::vector<Eigen::Isometry3d> gt_poses;

    int offset = 3;
    int num_frames = 2;
    for (int l = 0; l < offset; l++) {
        std::getline(gt_poses_file, line);
    }

    for (int i = 0; i < num_frames; i++) {
        std::getline(gt_poses_file, line);
        std::stringstream ssline(line);
        // // KITTI format
        // ssline
        //     >> r11 >> r12 >> r13 >> t1
        //     >> r21 >> r22 >> r23 >> t2
        //     >> r31 >> r32 >> r33 >> t3;

        // // KITTI-360 format
        // ssline >> no_frame
        //         >> r11 >> r12 >> r13 >> t1
        //         >> r21 >> r22 >> r23 >> t2
        //         >> r31 >> r32 >> r33 >> t3;

        // TUM format
        ssline >> time_stamp
                >> t1 >> t2 >> t3
                >> qx >> qy >> qz >> qw;

        // Rotation
        // Eigen::Matrix3d rotation_mat;
        // rotation_mat << r11, r12, r13,
        //                 r21, r22, r23,
        //                 r31, r32, r33;
        Eigen::Quaterniond quaternion(qw, qx, qy, qz);
        Eigen::Matrix3d rotation_mat(quaternion);

        // Translation
        Eigen::Vector3d translation_mat;
        translation_mat << t1, t2, t3;

        Eigen::Isometry3d gt_pose;
        gt_pose.linear() = rotation_mat;
        gt_pose.translation() = translation_mat;

        std::cout << "[" << i << "]\n" << gt_pose.matrix() << std::endl;
        gt_poses.push_back(gt_pose);
    }

    Eigen::Isometry3d rel_pose = gt_poses[0].inverse() * gt_poses[1];
    std::cout << "[relative pose]\n" << rel_pose.matrix() << std::endl;

    return 0;
}