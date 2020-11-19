
# ORIGINAL CODE FROM UDACITY #
############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2020-11-19          </date>
# <author>  Daniel Angulo Suescun and Juan Guillermo Sierra from National University from Colombia </author>
# <email>  dangulos@unal.edu.co jgsierraa@unal.edu.co </email>
############################################################################

# collect source files

cmake_minimum_required(VERSION 2.8)
find_package(CUDA)
find_package(OpenCV REQUIRED)
file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (to_bw_files main.cpp)

CUDA_ADD_EXECUTABLE(to_bw ${to_bw_files} ${hdr} ${cu})
target_link_libraries(to_bw ${OpenCV_LIBS})