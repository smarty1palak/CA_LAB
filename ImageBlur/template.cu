#include<cuda.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include "wb.h"
#define BLUR_SIZE 1

//@@ INSERT CODE HERE

__global__ void blur( float * input, float * output, int  height, int width)
{

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x<height && y<width)
	{
	for(int k=0;k<3;k++)
	{
	float sum=0;
	int count=0;
	for(int i=x-BLUR_SIZE; i<= x+BLUR_SIZE; i++)
        {
		for(int j= y-BLUR_SIZE; j<=y+BLUR_SIZE;j++)
		{
			if(i>=0 && i<height && j>=0 && j<width)
			{
				count++;
				sum+=input[3*(i*width+j)+k];
			}
		}
	}
	output[3*(x*width+y)+k]=sum/count;
	}
	}
	else
            return ;
} 

int main(int argc, char *argv[]) {
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *deviceInputImageData;
  float *deviceOutputImageData;

  
  /* parse the input arguments */
  //@@ Insert code here
  wbArg_t args = wbArg_read(argc, argv);
  inputImageFile = wbArg_getInputFile(args, 0);

  inputImage = wbImport(inputImageFile);

  // The input image is in grayscale, so the number of channels
  // is 1
  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);

  // Since the image is monochromatic, it only contains only one channel
  outputImage = wbImage_new(imageWidth, imageHeight, 3);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputImageData,
             3*imageWidth * imageHeight * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData,
             3*imageWidth * imageHeight * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             3*imageWidth * imageHeight * sizeof(float),
             cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Compute, "Doing the computation on the GPU");
 
  dim3 block(32, 32, 1);
  dim3 grid(imageHeight/32 +1, imageWidth/32 +1, 1);

  blur<<<grid, block>>> (deviceInputImageData, deviceOutputImageData, imageHeight, imageWidth);


  wbTime_stop(Compute, "Doing the computation on the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             3*imageWidth * imageHeight * sizeof(float),
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(args, outputImage);

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
