#include "utils.h"
#include <stdio.h>
#include <math.h>       /* ceil */

// Max Threads per block in GeForce 210
#define TxB 1024

__global__
void rgba_to_grey_kernel(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols, int totalThreads)
{

  int initIteration, endIteration;

  int id = (blockDim.x * blockIdx.x) + threadIdx.x;

  initIteration = (852*480/totalThreads) * id;

  if (id == totalThreads)
    endIteration = 852*480;
  else
    endIteration = initIteration + ((852*480 / totalThreads));

  int index = 0;

  for (int aux = initIteration; aux < endIteration; aux++){
    int j = aux % 852;
    int i = (aux - j) / 852;
    index = (j + i * 852) * 3;
    int x = j * (numCols/852.0);
    int y = i * (numRows/480.0);
  
    int indexAux = (x + y * numCols);
    uchar4 px = rgbaImage[indexAux]; // thread pixel to process
    greyImage[index + 2] = px.x; 
    greyImage[index + 1] = px.y; 
    greyImage[index] = px.z;
    
  }
}

void rgba_to_grey(uchar4 * const d_rgbaImage,
                  unsigned char* const d_greyImage, size_t numRows, size_t numCols, int aBlockSize, int aGridSize)
{

  // Since it does not matter the relative position of a pixel
  // the block - grid assign strategy will simply be to cover
  // all pixels secuencially in 'x' axis
  //cols => 852
  //rows => 480
  int totalThreads = aBlockSize * aGridSize;
  printf("Total threads %d", totalThreads);
  // long long int total_px = 852*480;  // total pixels
  // long int grids_n = ceil(total_px / TxB); // grids numer
  const dim3 blockSize(aBlockSize, 1, 1);
  const dim3 gridSize(aGridSize, 1, 1);
  rgba_to_grey_kernel<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols, totalThreads);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}