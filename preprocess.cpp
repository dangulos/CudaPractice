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


/**
 * gets everything necessary for cuda to run correctly: loads the original image 
 * and creates the canvas for the resize one, checks for any errors, flaten the information
 * of the images and sets the necessary memory allocation
 * */
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

  //allocate memory for the resize image
  resizeImage.create(480, 852, CV_8UC3);

  // check if any error of creation of the images happen in opencv
  if (!originalImage.isContinuous() || !resizeImage.isContinuous()) {
    std::cerr << "Images aren't continuous!! Exiting." << std::endl;
    exit(1);
  }

  // set the directions of the data of respectie images
  *inputImage = (uchar4 *)originalImage.ptr<unsigned char>(0);
  *outputImage  = resizeImage.ptr<unsigned char>(0);

  // size of original image
  const size_t numPixels = numRows() * numCols();

  // size of resize image (always 480*852)
  const size_t numPixelsReduced = 480 * 852;

  //allocate memory on the device for both input and output
  checkCudaErrors(cudaMalloc(d_originalImage, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMalloc(d_resizeImage, sizeof(unsigned char) * numPixelsReduced * 3));
  checkCudaErrors(cudaMemset(*d_resizeImage, 0, 3 * numPixelsReduced  * sizeof(unsigned char))); //make sure no memory is left laying around

  //copy original image array to the GPU
  checkCudaErrors(cudaMemcpy(*d_originalImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

  // variables to liberate the memory allocated in main
  liberate_originalImage = *d_originalImage;
  liberate_resizeImage = *d_resizeImage;
}