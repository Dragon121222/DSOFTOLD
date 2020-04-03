cmake_minimum_required(VERSION 3.14)
project(DSOFT.exe)

#Add arrayfire
find_package(ArrayFire)

#Add GTK3
FIND_PACKAGE(PkgConfig REQUIRED)
PKG_CHECK_MODULES(GTK3 REQUIRED gtk+-3.0)

INCLUDE_DIRECTORIES(${GTK3_INCLUDE_DIRS})
LINK_DIRECTORIES(${GTK3_LIBRARY_DIRS})

#Add Boost Libraries 
find_package( Boost REQUIRED COMPONENTS program_options regex filesystem)
include_directories( ${Boost_INCLUDE_DIRS} )

#Add OpenGL Libraries
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )

#Add OpenCV
find_package( OpenCV REQUIRED )

set(CMAKE_CXX_STANDARD 14)

add_executable(exe DAPP/main.cpp DOPERATORS.cpp DNN/DNN.h DNN/DNN.cpp DNN/DSUDOKUGEN.h DNN/DSUDOKUGEN.cpp DCAD/DMESH.h DCAD/DMESH.cpp DCAD/DCAD.h DCAD/DCAD.cpp DAPP/DFSM.h DAPP/DFSM.cpp )

target_link_libraries(exe ArrayFire::afcpu ${GTK3_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${OpenCV_LIBS} ${Boost_LIBRARIES})