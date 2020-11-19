#include <iostream>
#include "utils.h"
#include <string>
#include <stdio.h>

// Function calling the kernel to operate
void reduction(uchar4 *const d_originalImage,
               unsigned char *const d_resizeImage,
               size_t numRows, size_t numCols,
               int aBlockSize, int aGridSize);

// include the preprocess file
#include "preprocess.cpp"

int main(int argc, char **argv)
{

  uchar4 *h_originalImage, *d_originalImage;
  unsigned char *h_resizeImage, *d_resizeImage;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::string input_file;
  std::string output_file;

  //make sure the context initializes ok
  checkCudaErrors(cudaFree(0));

  switch (argc)
  {
  case 2:
    input_file = std::string(argv[1]);
    output_file = "output.png";
    break;
  case 3:
    std::cout << "3 options" << argv[1] << argv[2] << endl;
    break;
  case 4:
    std::cout << "4 options" << argv[1] << argv[2] << argv[3] << endl;
    break;
  case 5:
    std::cout << "5 options" << argv[1] << argv[2] << argv[3] << argv[4] << endl;
    break;
  default:
    std::cerr << "Usage: ./to_bw input_file [output_filename]" << std::endl;
    exit(1);
  }
  //load the image and give us our input and output pointers
  //preProcess(&h_rgbaImage, &h_resizeImage, &d_rgbaImage, &d_resizeImage, input_file);
  preProcess(&h_originalImage, &h_resizeImage, &d_originalImage, &d_resizeImage, input_file);

  //call the cuda code
  cudaEventRecord(start);
  reduction(d_originalImage, d_resizeImage, numRows(), numCols(), 1024, 100);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Time elapsed: %lf\n", milliseconds);

  //size_t numPixels = numRows()*numCols();
  //cols => 852
  //rows => 480
  size_t numPixels = 480 * 852;
  checkCudaErrors(cudaMemcpy(h_resizeImage, d_resizeImage, sizeof(unsigned char) * numPixels * 3, cudaMemcpyDeviceToHost));

  /* Output the grey image */
  cv::Mat output(480, 852, CV_8UC3, (void *)h_resizeImage);
  // Open the window
  // cv::namedWindow("to_bw");
  // Display the image m in this window
  // cv::imshow("to_bw", output);
  // cvWaitKey (0);
  // cvDestroyWindow ("to_bw");
  //output the image
  cv::imwrite(output_file.c_str(), output);

  /* Cleanup */
  cudaFree(liberate_originalImage);
  cudaFree(liberate_resizeImage);

  return 0;
}
