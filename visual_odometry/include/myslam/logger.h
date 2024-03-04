#ifndef MYSLAM_LOGGER_H
#define MYSLAM_LOGGER_H

#include <fstream>

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/feature.h"

namespace myslam {

class Logger {
  public:
    typedef std::shared_ptr<Logger> Ptr;

    Logger();

    ~Logger();

    void logPose(const SE3 &pose);

    void logImage(const std::string filename, const cv::Mat image);

    void logFeatureMatchImages(const Frame::Ptr frame);

  public:
    std::string log_path_;

    std::ofstream log_file_;
};

}  // namespace myslam

#endif  // MYSLAM_LOGGER_H