cmake_minimum_required(VERSION 2.8.11)
project(okvis_timing)

add_library(${PROJECT_NAME}
  src/Timer.cpp
  src/NsecTimeUtilities.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC okvis_util)
