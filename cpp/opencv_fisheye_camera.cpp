#include <opencv2/opencv.hpp>
#include <opencv2/ccalib/omnidir.hpp>

#include <boost/format.hpp>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "usage: " << argv[0] << "num_frames [image_dir]" << std::endl;
        return 1;
    }

    // camera parameters
    cv::Mat K, xi, D, idx;
    K = (cv::Mat_<double>(3, 3) << 1.3363220825849971e+03, 0, 7.1694323510126321e+02,
                                    0, 1.3357883350012958e+03, 7.0576498308221585e+02,
                                    0, 0, 1);
    xi = (cv::Mat_<double>(1, 1) << 2.2134047507854890e+00);
    D = (cv::Mat_<double>(4, 1) << 1.6798235660113681e-02, 1.6798235660113681e-02, 4.2223943394772046e-04, 4.2462134260997584e-04);

    // image read
    std::string image_dir = "/home/kodogyu/Datasets/KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync/image_02/data_rgb/";
    if (argc == 3) { image_dir = argv[2]; }
    boost::format fmt("%010d.png");
    cv::Mat image;

    for (int i = 0; i < atoi(argv[1]); i++) {
        std::cout << "file: " << fmt % i << std::endl;
        image = cv::imread(image_dir + (fmt % i).str(), cv::IMREAD_GRAYSCALE);

        // undistort
        cv::Mat perspective, cylindrical, stereographic, longlati;
        cv::Size new_size(640, 480);
        cv::Mat Knew_perspective = (cv::Mat_<double>(3, 3) << new_size.width/4, 0, new_size.width/2,
                                                0, new_size.height/4, new_size.height/2,
                                                0, 0, 1);
        cv::Mat Knew_others = (cv::Mat_<double>(3, 3) << new_size.width/3.1415, 0, 0,
                                                0, new_size.height/3.1415, 0,
                                                0, 0, 1);
        cv::omnidir::undistortImage(image, perspective, K, D, xi, cv::omnidir::RECTIFY_PERSPECTIVE, Knew_perspective, new_size);
        cv::omnidir::undistortImage(image, cylindrical, K, D, xi, cv::omnidir::RECTIFY_CYLINDRICAL, Knew_others, new_size);
        cv::omnidir::undistortImage(image, stereographic, K, D, xi, cv::omnidir::RECTIFY_STEREOGRAPHIC, Knew_others, new_size);
        cv::omnidir::undistortImage(image, longlati, K, D, xi, cv::omnidir::RECTIFY_LONGLATI, Knew_others, new_size);

        // visualize
        cv::imshow("perspective", perspective);
        cv::imshow("cylindrical", cylindrical);
        cv::imshow("stereographic", stereographic);
        cv::imshow("longlati", longlati);

        cv::waitKey(0);
        // cv::destroyAllWindows();
    }

    return 0;
}