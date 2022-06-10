cmake_minimum_required(VERSION 3.4)

# Set the project name and version
project(myproject)

# Path variables
set(CASADI_DIR /home/shamil/Desktop/phd/code/casadi)

# Append path of the Franka library
list(APPEND CMAKE_PREFIX_PATH ${CASADI_DIR})
list(APPEND CMAKE_MODULE_PATH ${CASADI_DIR}/cmake)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
# add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)

# Dependencies
find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(casadi REQUIRED)

# Adding libraries
include_directories(${CASADI_DIR}/include)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
# include_directories(${CASADI_INCLUDE_DIR})

# Add executables
add_executable(main src/main.cpp)

# Link executables with libraries
target_link_libraries(main pinocchio::pinocchio casadi)