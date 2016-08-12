cmake_minimum_required(VERSION 2.8.11)
project(okvis_util)

# nothing to build for now...
add_library(${PROJECT_NAME} STATIC src/dependency-tracker.cc)
