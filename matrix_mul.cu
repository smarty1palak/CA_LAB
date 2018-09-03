#include<stdio.h>
#include<cuda.h>

# define M 1000
# define N 1000

__global__ void mult( int * a, int * b, int * c)
{
	unsigned int i= blockDim.x *blockIdx.x + threadIdx.x;
	unsigned int j= blockDim.y *blockIdx.y + threadIdx.y;
	int sum=0;
	if(i<M && j<N)
	{
		for(int k=0;k<N;k++)
		{
			sum+=(a[i*N+k]* b[k*N+j]);			
		}
		c[i*N+j]=sum;
	}
	else
           return ;
}

int check(int *a, int *b, int *c)
{
	for(int i=0;i<M;i++)
	{
		for(int j=0;j<N;j++)
		{
			int sum=0;
			for(int k=0;k<N;k++)
			{
				sum+=a[i*N+k]*b[k*N+j];
			}
			if(c[i*N+j]!=sum)
				return 0;
		}
	}
	return 1;
}

int main()
{
	int *h_a, *h_b, *h_c;
	int *d_a, *d_b, *d_c;

	// allocating memory on host	
	h_a = (int *)malloc(M * N * sizeof(int));
	h_b = (int *)malloc(M * N * sizeof(int));
	h_c = (int *)malloc(M * N * sizeof(int));
	
	//assigning random values to the array elements
	for(int i=0;i<M;i++)
	{
		for(int j=0;j<N;j++)
		{
			h_a[i*M+j]=1;
			h_b[i*M+j]=2;
		}
		
	}

	
	//assigning memory on the device	
	cudaMalloc((void **)&d_a, M*N*sizeof(int));
	cudaMalloc((void **)&d_b, M*N*sizeof(int));
	cudaMalloc((void **)&d_c, M*N*sizeof(int));

	//copying elements from host to device
	cudaMemcpy(d_a, h_a, M*N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, M*N*sizeof(int), cudaMemcpyHostToDevice);


	//declaring the number of blocks and number of threads per block
	dim3 threads(32,32);
	dim3 blocks(M/32+1, N/32+1);

	//calling the function and calculating the sum on device
	mult<<< blocks, threads >>>(d_a, d_b, d_c);

	//copying the result to host memory
	cudaMemcpy(h_c, d_c, M*N*sizeof(int), cudaMemcpyDeviceToHost);

	if(check(h_a, h_b, h_c))
		printf("Matrix multiplication is correct\n");
	else
		printf("Matrix multiplication is incorrect\n");

	cudaFree(d_a);
  	cudaFree(d_b);
  	cudaFree(d_c);

  	free(h_a);
  	free(h_b);
  	free(h_c);
	
}
