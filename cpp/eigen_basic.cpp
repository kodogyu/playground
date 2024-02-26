// g++ eigen.cpp -I/usr/include/eigen3 -o eigen
#include <iostream>
#include <Eigen/Dense>

int main() {
    // Vector
    Eigen::Vector2d vec2;
    vec2 << 5, 10;
    
    std::cout << vec2[0] << std::endl;

    return 0;
}