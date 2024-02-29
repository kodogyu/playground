//
// Created by gaoxiang on 19-5-4.
//

#include <gflags/gflags.h>
#include "myslam/visual_odometry.h"

// DEFINE_string(config_file, "/home/kodogyu/playground/visual_odometry/config/l515_stereo.yaml", "config file path");
int main(int argc, char **argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    std::string config_file = "/home/kodogyu/playground/visual_odometry/config/l515_stereo.yaml";

    myslam::VisualOdometry::Ptr vo(
        new myslam::VisualOdometry(config_file));
    vo->Init();
    vo->Run();

    return 0;
}
