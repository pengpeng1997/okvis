cmake_minimum_required(VERSION 2.8.11)
project(okvis_frontend)

if(NOT DO_TIMING)
  add_definitions(-DDEACTIVATE_TIMERS)
  message(STATUS "Deactivating timers.")
endif()    

# require Eigen
find_package( Eigen REQUIRED )
include_directories(${EIGEN_INCLUDE_DIR}) 

# build the library
add_library(${PROJECT_NAME} 
  src/Frontend.cpp
  src/VioKeyframeWindowMatchingAlgorithm.cpp
  src/stereo_triangulation.cpp
  src/ProbabilisticStereoTriangulator.cpp
  src/FrameNoncentralAbsoluteAdapter.cpp
  src/FrameRelativeAdapter.cpp
  include/okvis/Frontend.hpp
  include/okvis/VioKeyframeWindowMatchingAlgorithm.hpp
  include/okvis/triangulation/stereo_triangulation.hpp
  include/okvis/triangulation/ProbabilisticStereoTriangulator.hpp
  include/opengv/absolute_pose/FrameNoncentralAbsoluteAdapter.hpp
  include/opengv/relative_pose/FrameRelativeAdapter.hpp
  include/opengv/sac_problems/absolute_pose/FrameAbsolutePoseSacProblem.hpp
  include/opengv/sac_problems/relative_pose/FrameRelativePoseSacProblem.hpp
  include/opengv/sac_problems/relative_pose/FrameRotationOnlySacProblem.hpp)

# and link it                 
target_link_libraries(${PROJECT_NAME}
  ${OPENGV_LIBRARY} 
  ${BRISK_LIBRARIES} 
  ${CERES_LIBRARIES}
  okvis_util
  okvis_cv 
  okvis_ceres 
  okvis_timing 
  okvis_matcher)
