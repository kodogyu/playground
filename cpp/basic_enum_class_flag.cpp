#include <iostream>

enum Type{
    HELLO = 1,
    WORLD = 2,
    MARK = 4,
    HELLO_WORLD = HELLO + WORLD
};

int main() {

    Type type = Type::HELLO_WORLD;

    std::cout << "type & HELLO: " << (type & Type::HELLO) << std::endl;
    std::cout << "type & WORLD: " << (type & Type::WORLD) << std::endl;
    std::cout << "type & MARK: " << (type & Type::MARK) << std::endl;

    if (type & Type::HELLO) {
        std::cout << "hello ";
    }
    if (type & Type::WORLD) {
        std::cout << "world ";
    }
    if (type & Type::MARK) {
        std::cout << "! ";
    }
    std::cout << std::endl;
}