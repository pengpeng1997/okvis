cmake_minimum_required(VERSION 2.8.11)

project(okvis_time)

IF(libvisensor_FOUND)
  add_definitions(-DHAVE_LIBVISENSOR)
  MESSAGE(STATUS "Found libvisensor. Setting HAVE_LIBVISENSOR flag.")
ENDIF()

# build the library
add_library(${PROJECT_NAME}
  src/Time.cpp
  src/Duration.cpp)


SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-fPIC")
  