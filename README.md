# cudacv-bw
Cuda with OpenCV (C/C++) simple resize conversion from 1280, 1920 and 4k resolutions to 480 in google colab

Code done by Daniel Angulo Suescun and Juan Guillermo Sierra from the National University in colombia for the class Parallel Computing

To run this code you have to execute the following commands
```bash
# Clone the repo into your colab
!git clone https://github.com/dangulos/CudaPractice.git
#enter the directory
%cd /content/CudaPractice/
!apt-get install -qq gcc-5 g++-5 -y
!ln -s /usr/bin/gcc-5 
!ln -s /usr/bin/g++-5 
# this command compiles the libraries
!cmake . -D CMAKE_C_COMPILER=/content/CudaPractice/gcc-5 -D CMAKE_CXX_COMPILER=/content/CudaPractice/g++-5 /content/CudaPractice/
!make
#run the build
!./to_bw your_image.jpg
