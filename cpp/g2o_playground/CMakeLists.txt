# g2o 
LIST(APPEND CMAKE_MODULE_PATH /home/kodogyu/libraries/g2o/cmake_modules)
set(G2O_ROOT /usr/local)
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIR})

message(STATUS "g2o include path: ${G2O_INCLUDE_DIR}")

add_executable(circle_fit circle_fit.cpp)
set_target_properties(circle_fit PROPERTIES OUTPUT_NAME circle_fit)
# target_link_libraries(circle_fit ${G2O_LIBS})
target_link_libraries(circle_fit g2o_core g2o_stuff g2o_solver_dense)