#include<cuda_runtime.h>
#include<cufft.h>
#include<cufftXt.h>
#include<stdio.h>
#include<string>
#include<math.h>
cufftComplex* read_file(std::string file_path, size_t * size, bool shrink){
	size_t size2 = 1000000;
	//*size = get_data_size(file_path);
	//shrink the sample into a power of 2 so that transformations are done fast
	//if(shrink)
	//	*size = (size_t)pow(2, (size_t)log2(*size));
	FILE* file;
	file = fopen(file_path.c_str(), "r");
	if(file == NULL){
		printf("Error: Couldn't open file %s\n", file_path.c_str());
		exit(EXIT_FAILURE);
	}
	cufftComplex* data_cufft = (cufftComplex*)malloc(*size*sizeof(cufftComplex));
	cufftComplex* chunk_cufft = (cufftComplex*)malloc(size2*sizeof(cufftComplex));
	int offset = 1000;
	unsigned char* data = (unsigned char*)malloc((*size +offset)*sizeof(char));
	fread(data, 1,(( *size)+offset),file);
	for(int i =0; i < *size; i ++){
		data_cufft[i].x = (float) data[i];	
		//we're dealing with real numbers so set phase to 0
		data_cufft[i].y = 0;
		if(i<size2){
			chunk_cufft[i].x = (float)data[i+offset];
			chunk_cufft[i].y = 0;
		}
		//printf("%f %f\n", data_cufft[i].x, chunk_cufft[i].x);

	}
	fclose(file);
		//getchar();

	cufftHandle plan1;
	cufftHandle plan2;
	cufftPlan1d(&plan1, (int)*size, CUFFT_R2C, 1);
	cufftPlan1d(&plan2,(int)size2, CUFFT_R2C, 1);

	cufftComplex* d_data_cufft;
	cufftComplex* d_chunk_cufft;
	cudaMalloc((void**)&d_data_cufft, *size*sizeof(cufftComplex));
	cudaMalloc((void**)&d_chunk_cufft, size2*sizeof(cufftComplex));
	cudaMemcpy(d_data_cufft, data_cufft, *size*sizeof(cufftComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(d_chunk_cufft, chunk_cufft, size2*sizeof(cufftComplex), cudaMemcpyHostToDevice);
	cufftExecR2C(plan1,(cufftReal*)d_data_cufft, d_data_cufft);
	cufftExecR2C(plan2,(cufftReal*)d_chunk_cufft, d_chunk_cufft);
	cudaMemcpy(data_cufft, d_data_cufft, (*size/2+1)*sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(chunk_cufft, d_chunk_cufft, (size2/2+1)*sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	for(int i =0; i <(size2/2+1); i ++){
		printf("x1:%f , y1:%f, abs: %f\n", data_cufft[i].x, data_cufft[i].y, sqrt(data_cufft[i].x*data_cufft[i].x + data_cufft[i].y*data_cufft[i].y));
		printf("x2:%f , y2:%f, abs: %f\n", chunk_cufft[i].x, chunk_cufft[i].y, sqrt(chunk_cufft[i].x*chunk_cufft[i].x + chunk_cufft[i].y*chunk_cufft[i].y));
		printf("\n");
	}
	return data_cufft;
}
int main(int argc, char* argv[]){
	size_t s = 1000000;
	read_file(argv[1], &s, false);
	return 0;
}
