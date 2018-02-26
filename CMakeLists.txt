cmake_minimum_required(VERSION 2.8.11)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations ") # Eigen raises deprecated declarations warning.

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11 -O3")
  
  
if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "arm*")
  message(STATUS "ARM processor detected, will attempt to use NEON.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
else()
  message(STATUS "Assuming SSE instructions available.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mssse3")
endif()


find_package(Eigen3   REQUIRED)
find_package( Eigen REQUIRED )
FIND_PACKAGE(Boost COMPONENTS filesystem system REQUIRED)
find_package(Glog REQUIRED)
find_package( brisk 2 REQUIRED )
find_package(SuiteSparse REQUIRED) #will be needed in any case...
find_package(CXSparse QUIET) #optional
find_package( Ceres 1.9 REQUIRED )
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)

add_definitions(-DUSE_OPENMP)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CMAKE_BINARY_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS}) 
include_directories(${EIGEN_INCLUDE_DIR}) 
include_directories(BEFORE ${GLOG_INCLUDE_DIRS}) 
include_directories(BEFORE ${BRISK_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_PATH})

include_directories(${OPENGV_INCLUDE_DIR}) 

include_directories(${CHOLMOD_INCLUDE_DIR})



# BUILD LOCAL DEPENDENCIES
include_directories(okvis_util/include)
add_subdirectory(okvis_util)

include_directories(okvis_kinematics/include)
add_subdirectory(okvis_kinematics)
add_dependencies(okvis_kinematics okvis_util)

include_directories(okvis_time/include)
add_subdirectory(okvis_time)
add_dependencies(okvis_time okvis_util)

include_directories(okvis_cv/include)
add_subdirectory(okvis_cv)
add_dependencies(okvis_cv okvis_util)

include_directories(okvis_common/include)
add_subdirectory(okvis_common)

include_directories(okvis_ceres/include)
add_subdirectory(okvis_ceres)
add_dependencies(okvis_ceres okvis_util)

include_directories(okvis_timing/include)
add_subdirectory(okvis_timing)

include_directories(okvis_matcher/include)
add_subdirectory(okvis_matcher)
add_dependencies(okvis_matcher okvis_util)

include_directories(okvis_frontend/include)
add_subdirectory(okvis_frontend)
add_dependencies(okvis_frontend okvis_util)

include_directories(okvis_multisensor_processing/include)
add_subdirectory(okvis_multisensor_processing)
add_dependencies(okvis_multisensor_processing okvis_util)


set(DEPS ${CERES_LIBRARIES} blas lapack gomp ${SUITESPARSE_LIBRARIES} ${Eigen_LIBS} ${GLOG_LIBRARY})




add_library(okvis-original  okvis_library.cpp)
target_link_libraries(okvis-original ${common_libraries}
  pthread
  okvis_util
  okvis_kinematics
  okvis_time
  okvis_cv 
  okvis_common
  okvis_ceres
  okvis_timing
  okvis_matcher
  okvis_frontend 
  okvis_multisensor_processing
  ${Boost_LIBRARIES}
  ${DEPS}
  )
  
set_target_properties(okvis-original PROPERTIES LINKER_LANGUAGE CXX)
SET_TARGET_PROPERTIES(okvis-original PROPERTIES COMPILE_FLAGS "-fPIC -fopenmp")


generate_slam_library(okvis-original "-lgomp -fopenmp" )

  
