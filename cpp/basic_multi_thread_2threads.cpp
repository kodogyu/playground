#include <iostream>
#include <thread>
#include <mutex>
#include <unistd.h>  // sleep()


void frequent_thread_func(std::mutex* m) {
    printf("frequent thread activated\n");
    std::unique_lock<std::mutex> lock(*m);  // lock
    printf("frequent_thread: locked mutex\n");
    while (true) {
        printf("frequent_thread: frequent thread!\n");

        sleep(3);  // sleep 3 seconds
    }
    
}

void infinite_thread_func(std::mutex* m) {
    printf("infinite thread activated\n");

    std::unique_lock<std::mutex> lock(*m);  // lock
    printf("infinite_thread: locked mutex\n");

    while (true) {}  // do nothing
    printf("This should not be printed\n");
}

int main() {
    std::mutex m;
    std::thread infinite_thread(infinite_thread_func, &m);
    std::thread frequent_thread(frequent_thread_func, &m);

    frequent_thread.join();
    infinite_thread.join();

    return 0;
}