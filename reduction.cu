#include "utils.h"
#include <stdio.h>
#include <math.h>       

// Max Threads per block in GeForce 210
#define TxB 1024

__global__
void reduction_kernel(const uchar4* const originalImage,
                       unsigned char* const resizeImage,
                       int numRows, int numCols, int totalThreads)
{
  int aux = blockIdx.x * blockDim.x + threadIdx.x;
  int j = aux % 852;
  int i = (aux - j) / 852;
  int index = (j + i * 852) * 3;
  int x = j * (numCols/852.0);
  int y = i * (numRows/480.0);

  int indexAux = (x + y * numCols);
  uchar4 px = originalImage[indexAux]; // thread pixel to process
  resizeImage[index + 2] = px.x; 
  resizeImage[index + 1] = px.y; 
  resizeImage[index] = px.z;
}

void reduction(uchar4 * const d_originalImage,
                  unsigned char* const d_resizeImage, size_t numRows, size_t numCols, int aBlockSize, int aGridSize)
{

  
  int totalThreads = aBlockSize * aGridSize;

  // const dim3 blockSize(aBlockSize, 1, 1);
  // const dim3 gridSize(aGridSize, 1, 1);
  long long int total_px = 852*480;  // total pixels
  long int grids_n = ceil(total_px / aBlockSize); // grids numer
  const dim3 blockSize(aBlockSize, 1, 1);
  const dim3 gridSize(grids_n, 1, 1);
  reduction_kernel<<<gridSize, blockSize>>>(d_originalImage, d_resizeImage, numRows, numCols, totalThreads);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}