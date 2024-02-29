#include <iostream>
#include <opencv2/opencv.hpp>
// #include <cv_bridge/cv_bridge.h>

using namespace std;

int main() {
    std::string filepath = "/home/kodogyu/playground/visual_odometry/config/l515_stereo.yaml";
    cv::FileStorage fs = cv::FileStorage(filepath.c_str(), cv::FileStorage::READ);
    std::string s;

    fs["dataset_dir"] >> s;

    cout << "opencv version: " << CV_VERSION << endl;
    cout << "dataset dir: " << s << endl;

    return 0;
}