#include <Eigen/Dense>
#include <iostream>

int main() {
    // rotation, translation vector
    std::vector<double> target_pose_vec_12 = {0.999998,  0.000527263,  -0.00206693,   -0.0469029,
                                            -0.000529651,     0.999999,  -0.00115486,   -0.0283993,
                                              0.00206632,   0.00115596,     0.999997,   0.00206632,
                                                       0,            0,            0,            1};
    std::vector<double> orig_r_vec_9;
    std::vector<double> orig_t_vec_3;

    // std::vector<double> sample_pose_vec_12 = {0.999989697051137, 0.002036503656136008, 0.004056900840972814, -0.0008027808832581021,
    //                                 -0.002018493190442878, 0.9999881120860248, -0.004438622744511449, 0.003304891601216703,
    //                                 0.004065891884332017, -0.004430388186886486, -0.9999819199284047, 0.999994216600455};
    std::vector<double> sample_pose_vec_12 = {0.9999949254423817, 0.002034693955030138, 0.002451348607314396, -0.0008027808832581021,
 -0.002040017452757473, 0.9999955622708887, 0.002171121213014298, 0.003304891601216703,
 -0.002446920161685563, -0.00217611098947632, 0.9999946385469692, 0.999994216600455};
    std::vector<double> test_r_vec_9;
    std::vector<double> test_t_vec_3;



    // vector to matrix
    Eigen::Matrix3d orig_rotation_mat, test_rotation_mat;
    Eigen::Vector3d orig_translation_mat, test_translation_mat;

    orig_rotation_mat << target_pose_vec_12[0], target_pose_vec_12[1], target_pose_vec_12[2],
                    target_pose_vec_12[4], target_pose_vec_12[5], target_pose_vec_12[6],
                    target_pose_vec_12[8], target_pose_vec_12[9], target_pose_vec_12[10];
    orig_translation_mat << target_pose_vec_12[3], target_pose_vec_12[7], target_pose_vec_12[8];

    test_rotation_mat << sample_pose_vec_12[0], sample_pose_vec_12[1], sample_pose_vec_12[2],
                    sample_pose_vec_12[4], sample_pose_vec_12[5], sample_pose_vec_12[6],
                    sample_pose_vec_12[8], sample_pose_vec_12[9], sample_pose_vec_12[10];
    test_translation_mat << sample_pose_vec_12[3], sample_pose_vec_12[7], sample_pose_vec_12[8];

    // eigen pose
    Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d target_pose;  // gt
    Eigen::Isometry3d sample_pose;  // estimate

    target_pose.linear() = orig_rotation_mat;
    target_pose.translation() = orig_translation_mat;
    sample_pose.linear() = test_rotation_mat;
    sample_pose.translation() = test_translation_mat;

    // relative pose
    Eigen::Isometry3d relative_pose = target_pose.inverse() * sample_pose;

    // RPEt
    double rpe_t = relative_pose.translation().norm();

    std::cout << "target pose: \n" << target_pose.matrix() << std::endl;
    std::cout << "sample pose: \n" << sample_pose.matrix() << std::endl;
    std::cout << "relative pose: \n" << relative_pose.matrix() << std::endl;
    std::cout << "RPEt = " << rpe_t << std::endl;

    return 0;
}