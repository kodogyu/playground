#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <vector>

Eigen::Isometry3d rotatePose(Eigen::Isometry3d pose) {
    Eigen::Matrix3d rotation;
    rotation << 0, 1, 0,
                0, 0, -1,
                -1, 0, 0;

    Eigen::Isometry3d rotation_transform = Eigen::Isometry3d::Identity();
    rotation_transform.linear() = rotation;

    Eigen::Isometry3d result = rotation_transform * pose;
    return result;
}

int main() {
    std::string file_path = "/home/kodogyu/swc_capstone/system_test/gt_trajectory.txt";
    std::cout << "pose file path: " << file_path << std::endl;

    std::ifstream gt_poses_file(file_path);
    int no_frame;
    double time_stamp;
    double r11, r12, r13, r21, r22, r23, r31, r32, r33, t1, t2, t3;
    double qw, qx, qy, qz;
    std::string line;
    std::vector<Eigen::Isometry3d> gt_poses;

    int offset = 1;
    int num_frames = 5;
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
        // with quaternion
        // Eigen::Matrix3d rotation_mat;
        // rotation_mat << r11, r12, r13,
        //                 r21, r22, r23,
        //                 r31, r32, r33;

        // without quaternion
        Eigen::Quaterniond quaternion(qw, qx, qy, qz);
        Eigen::Matrix3d rotation_mat(quaternion);

        // Translation
        Eigen::Vector3d translation_mat;
        translation_mat << t1, t2, t3;

        Eigen::Isometry3d gt_pose = Eigen::Isometry3d::Identity();
        gt_pose.linear() = rotation_mat;
        gt_pose.translation() = translation_mat;

        std::cout << "[" << i << "]\n" << gt_pose.matrix() << std::endl;
        gt_poses.push_back(gt_pose);
    }

    int reference_frame = 0;
    int target_frame = 2;
    Eigen::Isometry3d rel_pose = gt_poses[reference_frame].inverse() * gt_poses[target_frame];
    std::cout << "[relative pose from " << reference_frame << " -> " << target_frame << "]\n" << rel_pose.matrix() << std::endl;


    // rotate poses
    std::cout << "---------- rotate poses ----------" << std::endl;
    Eigen::Isometry3d rotated_pose;
    std::vector<Eigen::Isometry3d> rotated_poses;

    for (int i = 0; i < gt_poses.size(); i++) {
        rotated_pose = rotatePose(gt_poses[i]);
        std::cout << "[" << i << "]\n" << rotated_pose.matrix() << std::endl;

        rotated_poses.push_back(rotated_pose);
    }

    Eigen::Isometry3d rel_pose2 = rotated_poses[reference_frame].inverse() * rotated_poses[target_frame];
    std::cout << "[relative pose from " << reference_frame << " -> " << target_frame << "]\n" << rel_pose2.matrix() << std::endl;

    return 0;
}