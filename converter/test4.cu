#include<cuda_runtime.h>
#include<stdio.h>
__global__
void something(int* a){
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	a[id] = 0;
}
int main(){
	int * a;
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cudaMalloc((void**)&a, 2049*sizeof(int));
	something<<<3,1024, 0, stream>>>(a);
	cudaFree(a);
	return 0;
}
