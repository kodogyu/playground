// g++ matchTemplate.cpp -o matchTemplate `pkg-config opencv --cflags --libs`
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    string image_name = "../ipynb/color_frames/color_image484.png";
    Mat input_image = imread(image_name, IMREAD_COLOR);
    
    Rect roi(Point2i(130, 380), Point2i(170, 420));
    Mat templ = Mat(input_image, roi);

    cout << "template size: " << templ.size << endl;
    cout << "input image size: " << input_image.size << endl;

    Mat templ_input;
    matchTemplate(templ, input_image, templ_input, CV_TM_SQDIFF_NORMED);
    Mat input_templ;
    matchTemplate(input_image, templ, input_templ, CV_TM_SQDIFF_NORMED);
    imshow("templ, input", templ_input);
    imshow("input templ", input_templ);
    waitKey(0);
    destroyAllWindows();

    return 0;
}