#include "utils.h"
#include <stdio.h>
#include <math.h>       

// Max Threads per block in GeForce 210
#define TxB 1024

__global__
void reduction_kernel(const uchar4* const rgbaImage,
                       unsigned char* const outputImage,
                       int numRows, int numCols, int totalThreads)
{

  int initIteration, endIteration;

  int id = (blockDim.x * blockIdx.x) + threadIdx.x;

  initIteration = (852*480/totalThreads) * id;

  if (id == totalThreads - 1)
    endIteration = 852*480;
  else
    endIteration = initIteration + ((852*480 / totalThreads));

  int index = 0;

  for (int aux = initIteration; aux < endIteration; aux++){
    int j = aux % 852;
    int i = (aux - j) / 852;
    index = aux * 3;
    int x = j * (numCols/852.0);
    int y = i * (numRows/480.0);
  
    int indexAux = (x + y * numCols);
    uchar4 px = rgbaImage[indexAux]; // thread pixel to process
    outputImage[index + 2] = px.x; 
    outputImage[index + 1] = px.y; 
    outputImage[index] = px.z;
    
  }
}

void reduction(uchar4 * const d_originalImage,
                  unsigned char* const d_resizeImage, size_t numRows, size_t numCols, int aBlockSize, int aGridSize)
{

  //cols => 852
  //rows => 480
  int totalThreads = aBlockSize * aGridSize;
  printf("Total threads %d", totalThreads);
  // long long int total_px = 852*480;  // total pixels
  // long int grids_n = ceil(total_px / TxB); // grids numer
  const dim3 blockSize(aBlockSize, 1, 1);
  const dim3 gridSize(aGridSize, 1, 1);
  reduction_kernel<<<gridSize, blockSize>>>(d_originalImage, d_resizeImage, numRows, numCols, totalThreads);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}