cmake_minimum_required(VERSION 2.8.11)

project(okvis_cv)

# require Eigen
find_package( Eigen REQUIRED )
include_directories(${EIGEN_INCLUDE_DIR}) 

# require OpenCV
find_package( OpenCV COMPONENTS core highgui imgproc features2d REQUIRED )
include_directories(BEFORE ${OpenCV_INCLUDE_DIRS}) 

# build the library
add_library(${PROJECT_NAME} STATIC 
  src/CameraBase.cpp
  src/NCameraSystem.cpp
)

# and link it
target_link_libraries(${PROJECT_NAME}  
  PUBLIC okvis_util
  PUBLIC okvis_kinematics 
  PUBLIC okvis_time
  PRIVATE ${OpenCV_LIBRARIES} 
  PRIVATE ${BRISK_LIBRARIES}
)
