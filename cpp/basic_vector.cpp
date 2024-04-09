#include <vector>
#include <iostream>

int main() {
    std::vector<int> a = {1, 2, 3 ,4 , 5};
    std::vector<int> b;

    b.insert(b.begin(), a.begin(), a.begin() + 5);

    for (auto elem : b) {
        std::cout << elem << std::endl;
    }

    return 0;
}