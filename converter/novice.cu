#include<fstream>
#include<stdlib.h>
#include<string>
#include<stdio.h>
#include<omp.h>
#include<cuda_runtime.h>
#include<cufft.h>
#include<cufftXt.h>
#include<math.h>

char** sample_names;
size_t* sample_sizes;
int num_samples = 0;

__global__ 
void calc_LAD(cufftComplex* sample, cufftComplex* partial, int sample_size){
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	int sample_idx = id % sample_size;
	if(id < 4*sample_size ){
		float diff = abs(sqrt(sample[sample_idx].x*sample[sample_idx].x + sample[sample_idx].y*sample[sample_idx].y)
				- sqrt(partial[id].x*partial[id].x + partial[id].y*partial[id].y));
		partial[id].x = diff;
		partial[id].y = 0;
		__syncthreads();
		for(unsigned int s = sample_size/2; s>0; s>>=1){
			if(sample_idx < s)
				partial[id].x += partial[id+s].x;
			__syncthreads();
		}
		//now the results are in the num_threads first elements of the array so we'll need one memcpy in host code
		if(sample_idx ==0)
			partial[id/sample_size] = partial[id];
	}
}

int get_num_files(std::string files){
	int c=0;
	for(int i=0; i < files.length() ; i ++)
		if(files.at(i) == '\n')  c++;
	return c;
}

size_t get_data_size(std::string file_path){
	std::ifstream in(file_path.c_str(), std::ifstream::ate | std::ifstream::binary);
	size_t size = in.tellg();
	in.close();
	return size;
}
	
void copy_samples(char* argv[]){
	std::string songs_path(argv[2]);
	std::string command = "cp ";
	command.append(songs_path);
	command.append("/* ");
	command.append("./Converted");
	system(command.c_str());
}


std::string run_command(std::string cmd) {
	std::string data;
	FILE * stream;
	const int max_buffer = 256;
	char buffer[max_buffer];
	cmd.append(" 2>&1");
	
	stream = popen(cmd.c_str(), "r");
	if (stream) {
		while (!feof(stream))
			if (fgets(buffer, max_buffer, stream) != NULL) data.append(buffer);
				pclose(stream);
	}
	return data;
}

cufftComplex* read_file(std::string file_path, size_t * size, bool shrink, int downsampling_factor){
	*size = get_data_size(file_path);
	(*size)/=downsampling_factor;
	//shrink the sample into a power of 2 so that transformations are done fast
	if(shrink)
		*size = (size_t)pow(2, (size_t)log2(*size));
	FILE* file;
	file = fopen(file_path.c_str(), "r");
	if(file == NULL){
		printf("Error: Couldn't open file %s\n", file_path.c_str());
		exit(EXIT_FAILURE);
	}
	cufftComplex* data_cufft = (cufftComplex*)malloc(*size*sizeof(cufftComplex));
	unsigned char* data = (unsigned char*)malloc((*size*downsampling_factor)*sizeof(char));
	fread(data, 1, *size*downsampling_factor,file);
	for(int i =0; i < *size; i ++){
		data_cufft[i].x = (float) data[i*downsampling_factor];	
		//we're dealing with real numbers so set phase to 0
		data_cufft[i].y = 0;
	}
	fclose(file);
	return data_cufft;
}

cufftComplex** bring_samples_data(std::string supdir, std::string files){
	num_samples = get_num_files(files);
	cufftComplex**  all_data_cufft = (cufftComplex**)malloc(num_samples*sizeof(cufftComplex*));
	sample_names = (char**)malloc(num_samples*sizeof(char*));
	sample_sizes = (size_t*)malloc(num_samples*sizeof(size_t));
	
	std::string delimiter = "\n";
	size_t pos = 0;
	std::string file;
	int c = num_samples;
	float start = omp_get_wtime();		
	while ((pos = files.find(delimiter)) != std::string::npos){
		file = files.substr(0, pos);
		sample_names[--c] = (char*)malloc(file.length()*sizeof(char));
		strcpy(sample_names[c] , file.c_str());
		files.erase(0, pos + delimiter.length());
		std::string s = supdir;
		s.append(file);
		//sample_sizes[c] = get_data_size(s);
		all_data_cufft[c] = read_file(s, &sample_sizes[c], true, 1);
		printf("%s: data read\n", file.c_str());
	}
	float end = omp_get_wtime();
	printf("time elapsed: %f\n", end - start);
	return all_data_cufft;
}

void get_cuda_error(cudaError_t error, int line){
	if(error != cudaSuccess){
		printf("%s line: %d\n", cudaGetErrorString(error), line);
		exit(EXIT_FAILURE);
	}
}

void get_cufft_result(cufftResult_t result, int line){
	if(result != CUFFT_SUCCESS){
		printf("CUFFT error number %d at line: %d\n", result, line);
		exit(EXIT_FAILURE);
	}
}

int calc_num_threads(int sample_size){
	size_t work_area_size;
	size_t free_mem;
	cudaMemGetInfo(&free_mem, NULL);
	cufftEstimate1d(sample_size, CUFFT_R2C, 1, &work_area_size);
	//(x-1)*work_area_size + x*(sample_size/2+1)*sizeof(cufftComplex) < free_mem
	return min((size_t)(4),(free_mem + work_area_size)/(work_area_size + (sample_size/2+1)*sizeof(cufftComplex)));
}


int main(int argc, char*argv[]){
	//copy music to converter music directory
	copy_samples(argv);
	//run converter
	system("python3.6 ./Converter.py");

	//bring in sample songs' data to RAM
	std::string all_sample_data = run_command("ls ./Data");
	cufftComplex** all_samples_data = bring_samples_data("./Data/", all_sample_data);

	//get a list of all complete data files
	std::string command = "ls ";
	command.append(argv[1]);
	std::string all_complete_data = run_command(command);

	//traverse through complete data files and compare
	std::string delimiter = "\n";
	size_t pos = 0;
	std::string file;
	//This is main loop of the program
	while ((pos = all_complete_data.find(delimiter)) != std::string::npos){
		file = all_complete_data.substr(0, pos);
		all_complete_data.erase(0, pos + delimiter.length());
		std::string s = argv[1];
		s.append("/");
		s.append(file);

		/*There is a trick in which input should be aligned to cufftComplex upon memory allocation
		  and plane creation.I think for omptimization reasons. but when executing the plan you just cast
		  the input to cufftReal*/
		size_t data_size;	
		cufftComplex* complete_data = read_file(s, &data_size,true, 1);
		printf("%s: data read\n", file.c_str());

		cufftComplex* d_complete_data;
		cudaError_t error = cudaMalloc((void**)&d_complete_data, data_size*sizeof(cufftComplex));
		get_cuda_error(error, __LINE__);
				//size_t free_mem;
				//cudaMemGetInfo(&free_mem, NULL);
				//printf("free mem: %zu\n", free_mem);
	
		error = cudaMemcpy(d_complete_data, complete_data, data_size*sizeof(cufftComplex), cudaMemcpyHostToDevice);
		get_cuda_error(error, __LINE__);

		cufftHandle plan;
		cufftResult_t result;
		float min_lad;
		for(int sample_no =0; sample_no < num_samples ; sample_no++){
			min_lad = -1;
			cufftComplex* d_sample_data;
			
			error = cudaMalloc((void**)&d_sample_data, sample_sizes[sample_no]*sizeof(cufftComplex));
			get_cuda_error(error, __LINE__);

			error = cudaMemcpy(d_sample_data, all_samples_data[sample_no], sample_sizes[sample_no]*sizeof(cufftComplex), cudaMemcpyHostToDevice);
			get_cuda_error(error, __LINE__);

			result = cufftPlan1d(&plan, sample_sizes[sample_no], CUFFT_R2C,1);
			get_cufft_result(result, __LINE__);

			result = cufftExecR2C(plan, (cufftReal*)d_sample_data, d_sample_data);
			get_cufft_result(result, __LINE__);
			//printf("%s is transformed and ready to check\n", sample_names[sample_no]);

			error = cudaDeviceSynchronize();
			get_cuda_error(error, __LINE__);

			//now is time to compare the sample with the complete data	
			int num_threads = calc_num_threads(sample_sizes[sample_no]);

			//create different plans for different host threads, it's a necessity imposed by cuFFT thread safety
			cufftHandle* plans = (cufftHandle*)malloc((num_threads-1)*sizeof(cufftHandle));
			plans[0] = plan;
			for(int i=1; i < num_threads ; i++){
				result = cufftPlan1d(&plans[i], sample_sizes[sample_no], CUFFT_R2C, 1);
				get_cufft_result(result, __LINE__);
			}
	
			cufftComplex* d_partial_data_transformed;//This contains subsets of data that are being transformed in parallel
			error = cudaMalloc((void**)&d_partial_data_transformed,num_threads*(sample_sizes[sample_no]/2+1)*sizeof(cufftComplex));
			get_cuda_error(error, __LINE__);
			
			/*if the time-domain signal is somehow continuous(for sound waves it's not illogical to assume that), 
			  then windows with little space between them show approximately the same signals with similar FFTs
			  so by taking advantage of that we don't compare sample with every window in complete signal and put 
			  a little bit of padding called space between the windows.
			  */
			size_t ss = sample_sizes[sample_no];
			int space = ss/32;	
			cufftComplex* lad_results = (cufftComplex*) malloc(num_threads*sizeof(cufftComplex));
			bool *stopped = (bool*)malloc(num_threads*sizeof(bool));// this is for contorlling when to stop
			bool everybody_stopped= false;
			int num_stopped_threads = 0;
			#pragma omp parallel num_threads(num_threads)
			{
				int ID = omp_get_thread_num();
				for(int i = ID*space ; i < data_size && !everybody_stopped ; i+= num_threads*space){
					if(i +ss < data_size){// the last chunk might be small so we have to handle that
																//the thread needs to stay so we can pass the barrier
																// but it doesn't do any work
						cufftExecR2C(plans[ID], (cufftReal*)d_complete_data+i, d_partial_data_transformed + ID*(ss/2 +1));
						error = cudaDeviceSynchronize();
						get_cuda_error(error, __LINE__);
					}
					else{// now this thread has no work to do
						if(stopped[ID] == false){
							stopped[ID] = true;
							num_stopped_threads ++;
						}
					}
					#pragma omp barrier
					#pragma omp single
					{
						if(num_stopped_threads == num_threads)// all threads have reached chunks smaller than sample_size and 
							everybody_stopped = true;			// therefore have stopped working
						else{// there are active threads 
							int block_dim = 1024;
							int grid_dim = (num_threads*(ss/2+1)-1)/block_dim +1;
							calc_LAD<<<grid_dim, block_dim>>>(d_sample_data, d_partial_data_transformed, ss/2+1);
							error = cudaDeviceSynchronize();
							get_cuda_error(error, __LINE__);	
							cudaMemcpy(lad_results, d_partial_data_transformed, num_threads*sizeof(cufftComplex), cudaMemcpyDeviceToHost);
							int min_index =0;
							for(int j=0; j < num_threads; j++)
								if(abs(lad_results[j].x) < abs(lad_results[min_index].x))
									min_index = j;	
							if(min_lad == -1)
								min_lad = abs(lad_results[min_index].x);
							else if(lad_results[min_index].x<min_lad)
								min_lad = lad_results[min_index].x;
						}
					}
				}
					
			}
			if(min_lad < 10000000)
				printf("%s matched\n", sample_names[sample_no]);
				
			//printf("min_lad=%f\n", min_lad);
			for(int i=0; i < num_threads; i++)
				cufftDestroy(plans[i]);
			cudaFree(d_sample_data);
			cudaFree(d_partial_data_transformed);
		}	
		cudaFree(d_complete_data);
	}
	return 0;
}

