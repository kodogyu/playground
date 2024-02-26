#include <iostream>
#include <unordered_map>

int main() {
    std::unordered_map<int, std::string> myMap = {
        {1, "One"},
        {2, "Two"},
        {3, "Three"}
    };

    // 마지막 요소에 접근
    auto firstElement = myMap.begin();
    firstElement++;
    
    // 키와 값을 출력
    std::cout << "Key of first element: " << firstElement->first << std::endl;
    std::cout << "Value of first element: " << firstElement->second << std::endl;

    return 0;
}
