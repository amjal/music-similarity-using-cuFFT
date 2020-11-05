#include<cuda_runtime.h>
int main(){
	int ** ptr;
	ptr = (int**)malloc(10*sizeof(int*));
	int ** tmp;
	cudaMalloc((void**)&temp, 10*sizeof(int*));
	cudaMemcpy(ptr, tmp, 10*sizeof(int*), cudaMemcpyDeviceToHost);
