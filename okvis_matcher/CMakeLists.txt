cmake_minimum_required(VERSION 2.8.3)
project(okvis_matcher)

# build the library 
add_library(${PROJECT_NAME}
  src/DenseMatcher.cpp
  src/MatchingAlgorithm.cpp
  src/ThreadPool.cpp
  include/okvis/DenseMatcher.hpp
  include/okvis/MatchingAlgorithm.hpp
  include/okvis/ThreadPool.hpp
)

# and link it
target_link_libraries(${PROJECT_NAME} 
  PUBLIC okvis_util 
)

SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-fPIC")
