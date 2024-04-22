#include <iostream>
#include <boost/format.hpp>
#include <string>

int main() {
    for (int i = 0; i < 10; i++) {
        std::cout << boost::format("hello %05d") % i << std::endl;
    }

    return 0;
}