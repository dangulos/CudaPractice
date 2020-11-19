# cudacv-bw
Cuda with OpenCV (C/C++) simple resize conversion from 1280, 1920 and 4k resolutions to 480 in google colab

To run this code you have to execute the following commands
```bash
# Create the makefile linking to your libraries
!git clone https://github.com/dangulos/CudaPractice.git
%cd /content/CudaPractice/
!apt-get install -qq gcc-5 g++-5 -y
!ln -s /usr/bin/gcc-5 
!ln -s /usr/bin/g++-5 
# this command compiles the libraries
!cmake . -D CMAKE_C_COMPILER=/content/CudaPractice/gcc-5 -D CMAKE_CXX_COMPILER=/content/CudaPractice/g++-5 /content/CudaPractice/
!make
#run the build
!./to_bw 4kwallpaper.jpg
