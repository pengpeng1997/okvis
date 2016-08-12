cmake_minimum_required(VERSION 2.8.11)

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=gnu++11 -O3")

if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "arm*")
  message(STATUS "ARM processor detected, will attempt to use NEON.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
else()
  message(STATUS "Assuming SSE instructions available.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mssse3")
endif()

get_directory_property(defs COMPILE_DEFINITIONS)

#list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include_directories(${CMAKE_BINARY_DIR}/include)
link_directories(${CMAKE_BINARY_DIR}/lib)

# require Eigen
find_package( Eigen REQUIRED )
include_directories(${EIGEN_INCLUDE_DIR}) 



FIND_PACKAGE(Boost COMPONENTS filesystem system REQUIRED)
include_directories(${Boost_INCLUDE_DIRS}) 



# glog
find_package(Glog REQUIRED)
include_directories(BEFORE ${GLOG_INCLUDE_DIRS}) 

# brisk
find_package( brisk 2 REQUIRED )
include_directories(BEFORE ${BRISK_INCLUDE_DIRS})


# SuitesSparse
find_package(SuiteSparse REQUIRED QUIET) #will be needed in any case...
find_package(CXSparse QUIET) #optional

# ceres
find_package( Ceres 1.9 REQUIRED )
include_directories(BEFORE ${CERES_INCLUDE_DIRS})

# OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
  message(STATUS "OpenMP found")
  add_definitions(-DUSE_OPENMP)
else ()
  message(STATUS "OpenMP NOT found")
endif ()

#OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#set(OpenGV_LIBRARIES opengv)
#set_target_properties(opengv PROPERTIES IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/lib/libopengv.a)
#add_dependencies(${OpenGV_LIBRARIES} opengv_external)
include_directories(${OPENGV_INCLUDE_DIR}) 

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
add_dependencies(okvis_cv brisk okvis_util)

include_directories(okvis_common/include)
add_subdirectory(okvis_common)

include_directories(okvis_ceres/include)
add_subdirectory(okvis_ceres)
add_dependencies(okvis_ceres ceres okvis_util)

include_directories(okvis_timing/include)
add_subdirectory(okvis_timing)

include_directories(okvis_matcher/include)
add_subdirectory(okvis_matcher)
add_dependencies(okvis_matcher okvis_util)

include_directories(okvis_frontend/include)
add_subdirectory(okvis_frontend)
add_dependencies(okvis_frontend opengv okvis_util)

include_directories(okvis_multisensor_processing/include)
add_subdirectory(okvis_multisensor_processing)
add_dependencies(okvis_multisensor_processing okvis_util)


# merge everything in the same lib  
add_library(okvis-original  
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
)
target_link_libraries(okvis-original
    pthread 
    ${Boost_LIBRARIES}
    ${CXSPARSE_LIBRARIES}   
    ${OPENGV_LIBRARY}
     gomp cholmod blas lapack
)
set_target_properties(okvis-original PROPERTIES LINKER_LANGUAGE CXX)

  
add_slambench_binary(okvis  main         original "" "-fopenmp" "-fopenmp;${GLOG_LIBRARY};okvis_common;okvis_multisensor_processing")    
  
  
  
  # :${OpenCV_LIBRARIES}:/home/toky/work/pamela/slambench2/deps/opengv/lib/libopengv.a::
  
  
  
  
