cmake_minimum_required(VERSION 3.10)
project(finale_programmazione_parallela LANGUAGES CXX CUDA)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include(FetchContent) # If not included already

FetchContent_Declare(sciplot_content
  GIT_REPOSITORY https://github.com/sciplot/sciplot.git
  GIT_TAG master)
  FetchContent_GetProperties(sciplot_content)
  if(NOT sciplot_content_POPULATED)
    FetchContent_Populate(sciplot_content)
  endif()
  include_directories(${sciplot_content_SOURCE_DIR})

#add_library(datalib STATIC src/dataManipulator.cpp)
#add_library(kmeans STATIC src/kmeans.cpp src/point.cpp)
#add_library(utils STATIC src/ioUtils.cpp src/plotUtils.cpp)
#target_include_directories(datalib PUBLIC include)
#target_include_directories(kmeans PUBLIC include)
#target_include_directories(utils PUBLIC include)
#if(WIN32)
#  target_include_directories(utils PUBLIC win_include)
#endif()

add_executable(finale src/main.cu)
target_link_libraries( finale ${OpenCV_LIBS} )
#target_link_libraries(datalib PUBLIC ${Boost_LIBRARIES})
#target_link_libraries(parziale PUBLIC datalib)
#target_link_libraries(parziale PUBLIC kmeans)
#target_link_libraries(parziale PUBLIC utils)
