#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
using namespace cv;
using namespace std;
static void example1_help()
{
 cout
 << "--------------------------------------------------------------------------" << endl
 << "This program shows how to launch a 3D visualization window. You can stop event loop to continue executing. "
 << "You can access the same window via its name. You can run event loop for a given period of time. " << endl
 << "Usage:" << endl
 << "./launching_viz" << endl
 << endl;
}

void example1() {
 example1_help();
 viz::Viz3d myWindow("Viz Demo");

 myWindow.spin();

 cout << "First event loop is over" << endl;

 viz::Viz3d sameWindow = viz::getWindowByName("Viz Demo");

 sameWindow.spin();

 cout << "Second event loop is over" << endl;

 sameWindow.spinOnce(1, true);
 while(!sameWindow.wasStopped())
 {
    sameWindow.spinOnce(1, true);
 }

 cout << "Last event loop is over" << endl;
}

static void help2()
{
 cout
 << "--------------------------------------------------------------------------" << endl
 << "This program shows how to visualize a cube rotated around (1,1,1) and shifted "
 << "using Rodrigues vector." << endl
 << "Usage:" << endl
 << "./widget_pose" << endl
 << endl;
}
void example2()
{
 help2();
 viz::Viz3d myWindow("Coordinate Frame");
 myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());
 viz::WLine axis(Point3f(-1.0f,-1.0f,-1.0f), Point3f(1.0f,1.0f,1.0f));
 axis.setRenderingProperty(viz::LINE_WIDTH, 4.0);
 myWindow.showWidget("Line Widget", axis);
 viz::WCube cube_widget(Point3f(0.5,0.5,0.0), Point3f(0.0,0.0,-0.5), true, viz::Color::blue());
 cube_widget.setRenderingProperty(viz::LINE_WIDTH, 4.0);
 myWindow.showWidget("Cube Widget", cube_widget);
 Mat rot_vec = Mat::zeros(1,3,CV_32F);
 float translation_phase = 0.0, translation = 0.0;
 while(!myWindow.wasStopped())
 {
 /* Rotation using rodrigues */
 rot_vec.at<float>(0,0) += (float)CV_PI * 0.01f;
 rot_vec.at<float>(0,1) += (float)CV_PI * 0.01f;
 rot_vec.at<float>(0,2) += (float)CV_PI * 0.01f;
 translation_phase += (float)CV_PI * 0.01f;
 translation = sin(translation_phase);
 Mat rot_mat;
 Rodrigues(rot_vec, rot_mat);
 Affine3f pose(rot_mat, Vec3f(translation, translation, translation));
 myWindow.setWidgetPose("Cube Widget", pose);
 myWindow.spinOnce(1, true);
 }
}

void example3() {
    std::vector<cv::Point3d> pts3d = {cv::Point3d(5, 5, 10)};
    viz::Viz3d window; //creating a Viz window
    //Displaying the Coordinate Origin (0,0,0)
    window.showWidget("coordinate", viz::WCoordinateSystem(100));
    //Displaying the 3D points in green
    window.showWidget("points", viz::WCloud(pts3d, viz::Color::green()));
    window.spin();
}

int main()
{
    example3();

    return 0;
}