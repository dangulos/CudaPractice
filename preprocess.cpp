#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

cv::Mat originalImage;
cv::Mat resizeImage;

uchar4        *liberate_originalImage;
unsigned char *liberate_resizeImage;

size_t numRows() { return originalImage.rows; }
size_t numCols() { return originalImage.cols; }

//return types are void since any internal error will be handled by quitting
//no point in returning error codes...
//returns a pointer to an RGBA version of the input image
//and a pointer to the single channel grey-scale output
//on both the host and device
void preProcess(uchar4 **inputImage, unsigned char **outputImage,
                uchar4 **d_originalImage, unsigned char **d_resizeImage,
                const std::string &filename) {
  //context ok
  checkCudaErrors(cudaFree(0));

  cv::Mat image;
  
  image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
  if (image.empty()) {
    std::cerr << "Couldn't open file: " << filename << std::endl;
    exit(1);
  }
  // color format
  cv::cvtColor(image, originalImage, CV_BGR2RGBA);

  //allocate memory for the output
  //cols => 852
  //rows => 480
  resizeImage.create(480, 852, CV_8UC3);

  //This shouldn't ever happen given the way the images are created
  //at least based upon my limited understanding of OpenCV, but better to check
  if (!originalImage.isContinuous() || !resizeImage.isContinuous()) {
    std::cerr << "Images aren't continuous!! Exiting." << std::endl;
    exit(1);
  }

  *inputImage = (uchar4 *)originalImage.ptr<unsigned char>(0);
  *outputImage  = resizeImage.ptr<unsigned char>(0);

  const size_t numPixels = numRows() * numCols();
  const size_t numPixelsReduced = 480 * 852;
  //allocate memory on the device for both input and output
  checkCudaErrors(cudaMalloc(d_originalImage, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMalloc(d_resizeImage, sizeof(unsigned char) * numPixelsReduced * 3));
  checkCudaErrors(cudaMemset(*d_resizeImage, 0, 3 * numPixelsReduced  * sizeof(unsigned char))); //make sure no memory is left laying around

  //copy input array to the GPU
  checkCudaErrors(cudaMemcpy(*d_originalImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

  liberate_originalImage = *d_originalImage;
  liberate_resizeImage = *d_resizeImage;
}