cmake_minimum_required(VERSION 2.8.11)
project(okvis_multisensor_processing)

if(NOT DO_TIMING)
  add_definitions(-DDEACTIVATE_TIMERS)
  message(STATUS "Deactivating timers.")
endif()

# generate the configure header file
configure_file(../cmake/okvisConfig.hpp.in "${CMAKE_CURRENT_BINARY_DIR}/okvisConfig.hpp" @ONLY)
message(STATUS ${CMAKE_CURRENT_BINARY_DIR}/okvisConfig.hpp)

# require Eigen
find_package( Eigen REQUIRED )
include_directories(${EIGEN_INCLUDE_DIR}) 

# require OpenCV
find_package( OpenCV COMPONENTS core highgui imgproc features2d REQUIRED )
include_directories(BEFORE ${OpenCV_INCLUDE_DIRS}) 

add_library(${PROJECT_NAME}
  src/ThreadedKFVio.cpp
  src/ImuFrameSynchronizer.cpp
  src/FrameSynchronizer.cpp
  src/VioVisualizer.cpp
  include/okvis/ThreadedKFVio.hpp
  include/okvis/ImuFrameSynchronizer.hpp
  include/okvis/FrameSynchronizer.hpp
  include/okvis/VioVisualizer.hpp
  include/okvis/threadsafe/ThreadsafeQueue.hpp
  ../cmake/okvisConfig.hpp.in
  okvisConfig.hpp
)

# and link it                     
target_link_libraries(${PROJECT_NAME}
   PUBLIC okvis_time
   PUBLIC okvis_util
   PUBLIC okvis_kinematics 
   PUBLIC okvis_cv 
   PUBLIC okvis_common
   PUBLIC okvis_ceres 
   PUBLIC okvis_frontend
   PRIVATE ${GLOG_LIBRARIES}
)

# export config
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_BINARY_DIR}/okvisConfig.hpp)
