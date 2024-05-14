#include <vector>
#include <iostream>

int main() {
    std::vector<int> a = {1, 2, 3 ,4 , 5};
    std::vector<int> b;

    b.insert(b.begin(), a.begin(), a.begin() + 5);

    for (auto elem : b) {
        std::cout << elem << std::endl;
    }

    std::cout << "----------" << std::endl;
    std::cout << "a[0]: " << a[0] << std::endl;
    std::cout << "a[-1]: " << a[-1] << std::endl;

    return 0;
}