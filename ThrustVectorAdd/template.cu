#include<bits/stdc++.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace std;
int check( float *c, float *b, float *a, int n)
{
  for(int i=0;i<n;i++)
  {
    if(c[i] !=a[i] +b[i])
      return 0;
  }
  return 1;
}


int main(int argc, char *argv[]) {

  float *hostInput1 = NULL;
  float *hostInput2 = NULL;
  float *hostOutput = NULL;
  int length;

  /* parse the input arguments */
  //@@ Insert code here

  // Import host input data
  //@@ Read data from the raw files here
   FILE * a = fopen(argv[1], "r");
   FILE * b = fopen(argv[2], "r");
   FILE * c = fopen(argv[3], "r");
  
   fscanf( c, "%d", &length);
   
  //@@ Insert code here
  hostInput1 = (float *)malloc(length * sizeof(float));
  hostInput2 = (float *)malloc(length * sizeof(float));

  // Declare and allocate host output
  hostOutput = (float *) malloc(length * sizeof(float));
  //@@ Insert code here
	for(int i=0;i<length;i++)
   {
	fscanf( a, "%f", &hostInput1[i]); 
	fscanf( b, "%f", &hostInput2[i]); 
   }
  // Copy to device
  //@@ Insert code here
  thrust::device_vector<float> d_a(hostInput1,hostInput1+length);
  thrust::device_vector<float> d_b(hostInput2,hostInput2+length);
  thrust::device_vector<float> d_c(hostOutput,hostOutput+length);


  // Execute vector addition
  //@@ Insert Code here
  thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_c.begin(), thrust::plus<float>());
  /////////////////////////////////////////////////////////

  // Copy data back to host
  //@@ Insert code here
  thrust::copy(d_c.begin(), d_c.end(), hostOutput);
  if ( check(hostOutput, hostInput2, hostInput1, length))
    cout<<"Vector addition is correct\n";
  else
    cout<<"Vector addition is incorrect\n";

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  return 0;
}
