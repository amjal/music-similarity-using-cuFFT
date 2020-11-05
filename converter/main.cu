#include<fstream>
#include<stdlib.h>
#include<string>
#include<stdio.h>
#include<omp.h>
#include<cuda_runtime.h>
#include<cufft.h>
#include<cufftXt.h>
#include<math.h>

int downsampling_rate = 3;
/* since there is no device-wide synchronization mechanism in cuda, calculating the differences and summing them 
   are done in seperate kernels. Kernel calls imply implicit device-wide synchronization*/
void copySamples(char* sample_songs_directory);

/*this function will fetch sample data from RAM, copy sample data into GPU, and transform sample data and does this in a 
three way concurrent way*/
void prepareSamples(char* samples_directory, char*** sample_names, size_t** sample_sizes,
		unsigned int* num_samples, cufftComplex*** d_transformed_samples);

//runs command and returns its output as a string
std::string captureCommandOutput(std::string command);

//gets a list of files(from a command output) and returns how many file there are in that list
unsigned int getNumFiles(std::string files_list);

/*reads file with path 'path' and fill the information. Assuming the sampling rate for our data is 44kHz and that input
data are songs, furthermore assuming that frequency of musical sound doesn't have frequencies greater than 4.4kHz (which
is a good assumption), we can downsample our input file with up to a factor of 5 without loss of accuracy.
*/
void readFile(std::string path, cufftComplex** data, size_t* size, int downsampling_factor);

//This is a part of prepareSamples function
void fetchNFill(char* samples_directory, std::string sample_files_list, cufftComplex** samples, 
	  char*** sample_names, size_t** sample_sizes, unsigned int* pos);

//This is part of preapareSamples function, this one copys data from host to device and allocates needed data for FFT
void prepareTransform(cudaStream_t stream, cufftComplex** samples, size_t* sample_sizes, 
	unsigned int pos, cufftComplex** d_samples_shadow); 

//This too, is a part of prepareSamples function. It does the last part which is transforming samples
void transformSample(cudaStream_t stream, cufftComplex** d_samples_shadow, size_t* sample_sizes, unsigned int pos);

/*This is the second major function the program runs, it's made of 3 major parts that are run 3 way concurrently
given a data array and a sample, this function returns whether or not sample matches the array*/
bool findMatches(char* data_path, cufftComplex** d_transformed_samples, int sample_num, 
		size_t sample_size, unsigned int batch_size);

unsigned int calcBatchSize(size_t sample_size);
void checkCudaError(cudaError_t error, int line);
void checkCufftResult(cufftResult_t result, int line);
size_t getFileSize(std::string path, int downsampling_factor);
void fetchBatch(std::ifstream* in, size_t read_size, cufftComplex*data, int downsampling_factor);

__global__
void calcDiffs(cufftComplex** d_transformed_samples, int sample_num, cufftComplex* d_data_batch_transformed,
		size_t transformed_sample_size, int num_chunks_copied_to_device){
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	int sample_idx = id % transformed_sample_size;
	if(id < num_chunks_copied_to_device*transformed_sample_size){
		double sample_x = d_transformed_samples[sample_num][sample_idx].x/transformed_sample_size;
		double sample_y = d_transformed_samples[sample_num][sample_idx].y/transformed_sample_size;
		double chunk_x = d_data_batch_transformed[id].x/transformed_sample_size;
		double chunk_y = d_data_batch_transformed[id].y/transformed_sample_size;

		float diff = abs(sqrt(sample_x*sample_x + sample_y*sample_y) - sqrt(chunk_x*chunk_x +chunk_y*chunk_y));
		d_data_batch_transformed[id].x = diff;
		d_data_batch_transformed[id].y = 0;
	}
}

__global__ 
void intraSampleSum(cufftComplex* d_data_batch_transformed, size_t chunk_size){
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	if(id < chunk_size){
		for(int s = blockDim.x/2 ; s>0 ; s >>=1){
			if (tid < s && id + s < chunk_size)
				d_data_batch_transformed[id].x += d_data_batch_transformed[id + s].x;
			__syncthreads();
		}
	}
	
}

__global__
void aligner(cufftComplex* d_data_batch_transformed, int chunk_size, int num_chunks_copied_to_device){
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	int new_chunk_size = (chunk_size-1)/blockDim.x +1;
	int chunk_idx = id % new_chunk_size;
	int chunk_id = id/new_chunk_size;
	if(id < new_chunk_size * num_chunks_copied_to_device){
		d_data_batch_transformed[id] = d_data_batch_transformed[chunk_id*chunk_size + blockDim.x*chunk_idx];
	}
}

int main(int argc, char* argv[]){
	/*The program requires two arguments at runtime. The first one is the directory containing 
	  complete songs' data, the second is the directory containing sample .wavs*/ 
	if(argc != 3){
		printf("Error: incorrect parameters\n");
		exit(EXIT_FAILURE);
	}
	//we should first copy the .wav samples into the Converted folder so that the converter converts them into data. 
	copySamples(argv[2]);
	//run the converter to convert sample .wav files into .txt data
	system("python3.6 ./Converter.py");

	char** sample_names;
	size_t* sample_sizes;
	unsigned int num_samples;
	cufftComplex** d_transformed_samples;
	prepareSamples("./Data", &sample_names, &sample_sizes, &num_samples, &d_transformed_samples);

	std::string command = "ls ";
	command.append(argv[1]);
	std::string data_files_list = captureCommandOutput(command);
	unsigned int num_data = getNumFiles(data_files_list);
	char** data_paths = (char**)malloc(num_data*sizeof(char*));
	size_t* data_sizes = (size_t*)malloc(num_data*sizeof(size_t));
	//just put the file names into an array for convenience
	std::string delim = "\n";
	std::string data_file_name;
	int str_pos = 0;
	int c = 0;
	while((str_pos = data_files_list.find(delim)) != std::string::npos){
		data_file_name = data_files_list.substr(0,str_pos);
		data_files_list.erase(0, str_pos+ delim.length());
		std::string tmp(argv[1]);
		tmp.append("/");
		tmp.append(data_file_name);
		data_sizes[c] = getFileSize(tmp, downsampling_rate);
		data_paths[c] = (char*)malloc(tmp.length()*sizeof(char));
		strcpy(data_paths[c] , tmp.c_str());
		c++;
	}
	for(int sample_num =0; sample_num < num_samples; sample_num++){
		unsigned int batch_size = calcBatchSize(sample_sizes[sample_num]);
		bool match_found = false;
		int matched_data_num;
		for(int data_num =0; data_num< num_data; data_num++){
			if(findMatches(data_paths[data_num], d_transformed_samples, sample_num, sample_sizes[sample_num], 
					 batch_size)){
				match_found = true;	
				matched_data_num = data_num;
				break;
			}
		}
		if(match_found)
			printf("%s matches %s\n", sample_names[sample_num], data_paths[matched_data_num]);
		else printf("%s matches nothing\n",sample_names[sample_num]);
	}
	return 0;
}

void copySamples(char* sample_songs_directory){
	std::string samples_path(sample_songs_directory);
	std::string command = "cp ";
	command.append(samples_path);
	command.append("/* ");
	command.append("./Converted");
	system(command.c_str());
}

void prepareSamples(char* samples_directory, char*** sample_names, size_t** sample_sizes, unsigned int* num_samples,
		cufftComplex*** d_transformed_samples){
	//first get a list of all files in that directory
	std::string command = "ls ";
	command.append(samples_directory);
	std::string sample_files_list = captureCommandOutput(command);
	
	//find how many data files there are
	*num_samples = getNumFiles(sample_files_list);

	//do memory initializations 
	cufftComplex** samples = (cufftComplex**)malloc(*num_samples*sizeof(cufftComplex*));
	*sample_names = (char**)malloc(*num_samples*sizeof(char*));
	*sample_sizes = (size_t*)malloc(*num_samples*sizeof(size_t));
	//d_transformed_samples resides in host, points to a location in device
	cudaError_t error = cudaMalloc((void**)d_transformed_samples, *num_samples*sizeof(cufftComplex*));
	checkCudaError(error, __LINE__);
	/*d_samples_shadow resides in host, points to an array of pointers residing
	in host, those pointers point to locations in GPU*/
	cufftComplex** d_samples_shadow = (cufftComplex**)malloc(*num_samples*sizeof(cufftComplex*));
	//error = cudaMemcpy(d_samples_shadow, d_samples, *num_samples*sizeof(cufftComplex*), cudaMemcpyDeviceToHost);
	//checkCudaError(error, __LINE__);

	unsigned int t1_pos=0, t2_pos=0, t3_pos=0;
	cudaStream_t t2_stream, t3_stream;
	cudaStreamCreate(&t2_stream);
	cudaStreamCreate(&t3_stream);
	#pragma omp parallel num_threads(3)
	{
		#pragma omp sections
		{
			#pragma omp section
			{
				//fetch all data and fill the arrays
				fetchNFill(samples_directory, sample_files_list, samples, sample_names, 
						sample_sizes, &t1_pos); 
			}
			#pragma omp section
			{
				/*copy sample from CPU to GPU, note that this is some kind of producer consumer model where
				  the consumer(t2) has to wait for the producer(t1) to finish its job and then start its own job
				  in parallel with t1's next job. This way we can achieve a 3 way concurrency */
				for(; t2_pos<*num_samples; t2_pos++){
					while(t2_pos >= t1_pos);
					prepareTransform(t2_stream, samples, *sample_sizes, t2_pos, d_samples_shadow);
				}
			}	
			#pragma omp section
			{
				for(; t3_pos< *num_samples; t3_pos++){
					while(t3_pos >= t2_pos);
					transformSample(t3_stream, d_samples_shadow, *sample_sizes, t3_pos);
				}
			}
		}
	}	
	error = cudaMemcpy(*d_transformed_samples, d_samples_shadow, *num_samples*sizeof(cufftComplex*), cudaMemcpyHostToDevice); 
	checkCudaError(error,__LINE__);
	free(d_samples_shadow);
	for(int i=0; i<*num_samples; i++){
		error = cudaFreeHost(samples[i]);
		checkCudaError(error,__LINE__);
	}
	cudaStreamDestroy(t2_stream);
	cudaStreamDestroy(t3_stream);
	free(samples);	
}

void fetchNFill(char* samples_directory, std::string sample_files_list, cufftComplex** samples, 
	  char*** sample_names, size_t** sample_sizes, unsigned int* pos){
		std::string delim = "\n";
		std::string sample_name;
		int str_pos = 0;
		while((str_pos = sample_files_list.find(delim)) != std::string::npos){
			sample_name = sample_files_list.substr(0,str_pos);
			sample_files_list.erase(0, str_pos+ delim.length());
			
			(*sample_names)[(*pos)] = (char*)malloc(sample_name.length()*sizeof(char));
			strcpy((*sample_names)[(*pos)], sample_name.c_str());

			std::string full_path = samples_directory;
			full_path.append("/");
			full_path.append(sample_name);
			readFile(full_path, &(samples[(*pos)]), &(*sample_sizes)[(*pos)], downsampling_rate);
			printf("%s: data read, size: %zu\n", (*sample_names)[*pos], (*sample_sizes)[*pos]);
			(*pos) ++;
		}	
}

void prepareTransform(cudaStream_t stream, cufftComplex** samples, size_t* sample_sizes, unsigned int pos, 
		cufftComplex** d_samples_shadow ){
	size_t sample_size = sample_sizes[pos];
	cudaError_t error = cudaMalloc((void**)&(d_samples_shadow[pos]),sample_size*sizeof(cufftComplex)); 
	checkCudaError(error, __LINE__);
	error = cudaMemcpyAsync(d_samples_shadow[pos], samples[pos], sample_size*sizeof(cufftComplex), cudaMemcpyHostToDevice, stream);
	checkCudaError(error, __LINE__);
	error = cudaStreamSynchronize(stream);
	checkCudaError(error, __LINE__);
}

void transformSample(cudaStream_t stream, cufftComplex** d_samples_shadow, size_t* sample_sizes, unsigned int pos){
	cufftHandle plan;
	cufftCreate(&plan);
	cufftSetStream(plan, stream);
	size_t work_size;
	cufftResult_t result = cufftMakePlan1d(plan, sample_sizes[pos], CUFFT_R2C,1, &work_size);
	checkCufftResult(result, __LINE__);
	result = cufftExecR2C(plan, (cufftReal*)d_samples_shadow[pos], d_samples_shadow[pos]);
	checkCufftResult(result, __LINE__);
	cudaError_t error = cudaStreamSynchronize(stream);
	checkCudaError(error,__LINE__);
	cufftDestroy(plan);
}

bool findMatches(char* data_path, cufftComplex** d_transformed_samples, int sample_num, size_t sample_size,
		 unsigned int batch_size){
	size_t data_size = getFileSize(data_path, downsampling_rate);
	
	/* Why the array of size 3? because I have a 3 way concurrent algorithm and in order to run that
	   algorithm in different streams and make it actually concurrent, data accesses need to be independent*/
	cufftComplex* d_data_batch[3];
	cufftComplex* d_data_batch_transformed[3];
	cudaError_t error = cudaMalloc((void**)&(d_data_batch[0]), batch_size*sample_size*sizeof(cufftComplex));
	checkCudaError(error,__LINE__);
	error = cudaMalloc((void**)&(d_data_batch[1]), batch_size*sample_size*sizeof(cufftComplex));
	checkCudaError(error,__LINE__);
	error = cudaMalloc((void**)&(d_data_batch[2]), batch_size*sample_size*sizeof(cufftComplex));
	checkCudaError(error,__LINE__);
	error = cudaMalloc((void**)&(d_data_batch_transformed[0]), batch_size*(sample_size/2+1)*sizeof(cufftComplex));
	checkCudaError(error,__LINE__);
	error = cudaMalloc((void**)&(d_data_batch_transformed[1]), batch_size*(sample_size/2+1)*sizeof(cufftComplex));
	checkCudaError(error,__LINE__);
	error = cudaMalloc((void**)&(d_data_batch_transformed[2]), batch_size*(sample_size/2+1)*sizeof(cufftComplex));
	checkCudaError(error,__LINE__);
	/* At first iteration where sample_num = 0, the data is not in RAM so there must be an extra thread bringing data
	   from disk to RAM*/
	cufftComplex* data;
	cudaMallocHost((void**)&data, data_size*sizeof(cufftComplex));//pinned memory for stream
	//usually equals to batch_size but near end of array might be less, transformer needs this value to do the transform
	bool t3_stop = false;
	bool t4_stop = false;
	int num_chunks_copied_to_device[3];
	num_chunks_copied_to_device[0] = batch_size;
	num_chunks_copied_to_device[1] = batch_size;
	num_chunks_copied_to_device[2] = batch_size;
	size_t stride = sample_size/8; 
	size_t read_size = (batch_size-1)*stride + sample_size;
	int t1_pos=0, t2_pos=0, t3_pos=0, t4_pos=0; 
	cudaStream_t t2_stream, t3_stream, t4_stream;
	cudaStreamCreate(&t2_stream);
	cudaStreamCreate(&t3_stream);
	cudaStreamCreate(&t4_stream);
	bool match_found =false;
	#pragma omp parallel num_threads(4)
	{
		#pragma omp sections
		{
			#pragma omp section
			{
				FILE* file = fopen(data_path, "r");
				char* tmp = (char*)malloc(data_size*downsampling_rate*sizeof(char));
				fread(tmp, 1, downsampling_rate*data_size, file);
				size_t batch_offset=0;
				for(int i =0; i < data_size && (!match_found); i++){
					data[i].x = tmp[i*downsampling_rate];
					data[i].y = 0;
					if(i - batch_offset +1 == read_size){
						//printf("t1 at %d\n", t1_pos);
						batch_offset = i-sample_size+stride;
						t1_pos ++;
					}
				}
				free(tmp);
				fclose(file);
				t1_pos++;
			}
			#pragma omp section
			{
				size_t batch_offset=0; 
				int i;
				while((sample_num ==0 && t2_pos >= t1_pos) ||(t2_pos - t4_pos>2));// wait for t1 to finish data_size but only when t1 is active
				for(i =0; i + sample_size < data_size && (!match_found) ; i+=stride){
					error = cudaMemcpyAsync(d_data_batch[t2_pos%3]+(i-batch_offset)/stride*sample_size, &(data)[i], 
							sample_size*sizeof(cufftComplex), cudaMemcpyHostToDevice, t2_stream);
					checkCudaError(error, __LINE__);
					error = cudaStreamSynchronize(t2_stream);
					checkCudaError(error, __LINE__);
					if(i+sample_size - batch_offset == read_size){
						//printf("t2 at %d\n", t2_pos);
						batch_offset = i+stride;
						t2_pos++;
						while((sample_num ==0 && t2_pos >= t1_pos) || (t2_pos - t4_pos>2));// wait for t1 to finish data_size but only when t1 is active
					}
				}
				num_chunks_copied_to_device[t2_pos%3] = (i-batch_offset)/stride;
				t2_pos++;
				t3_stop = true;
			}	
			#pragma omp section
			{
				do{
					while(t3_pos >= t2_pos);
					//printf("t3 at %d\n", t3_pos);
					cufftHandle plan;
					cufftCreate(&plan);
					cufftSetStream(plan, t3_stream);
					size_t work_size;
					cufftResult_t result = cufftMakePlan1d(plan, sample_size,
							CUFFT_R2C, batch_size, &work_size);
					checkCufftResult(result, __LINE__);
					result = cufftExecR2C(plan, (cufftReal*)d_data_batch[t3_pos%3], d_data_batch_transformed[t3_pos%3]);
					checkCufftResult(result, __LINE__);
					error = cudaStreamSynchronize(t3_stream);
					checkCudaError(error, __LINE__);
					cufftDestroy(plan);
					t3_pos++;
				}while(!t3_stop && !match_found);
				t4_stop = true;
			}
			#pragma omp section
			{
				size_t transformed_sample_size;
				//float global_min_lad;
				int block_dim;
				int grid_dim;
				int which_one;
				int new_chunk_size;
				do{
					while(t4_pos>=t3_pos);
					//printf("t4 at %d\n", t4_pos);
					transformed_sample_size = (sample_size/2+1);
					block_dim = 1024;
					grid_dim = (num_chunks_copied_to_device[t4_pos%3]*transformed_sample_size-1)/block_dim +1;
					if(grid_dim ==0)
						break;
					which_one = t4_pos%3;
					calcDiffs<<<grid_dim, block_dim, 0, t4_stream>>>(d_transformed_samples, sample_num, 
							d_data_batch_transformed[which_one], transformed_sample_size,
							num_chunks_copied_to_device[which_one]);
					error = cudaGetLastError();
					checkCudaError(error, __LINE__);
					error = cudaStreamSynchronize(t4_stream);
					checkCudaError(error, __LINE__);
					cudaStream_t* sum_streams = (cudaStream_t*)malloc(num_chunks_copied_to_device[which_one]*
							sizeof(cudaStream_t));
					for(int i =0; i < num_chunks_copied_to_device[which_one]; i++)
						cudaStreamCreate(&(sum_streams[i]));
					for(int chunk_size = transformed_sample_size; chunk_size>1; chunk_size =(chunk_size -1)/block_dim +1){
						grid_dim = (chunk_size -1)/block_dim + 1;
						for(int chunk_num =0; chunk_num < num_chunks_copied_to_device[which_one]; chunk_num++){
							intraSampleSum<<<grid_dim, block_dim, 0, sum_streams[chunk_num]>>>
								(d_data_batch_transformed[which_one] + chunk_num*chunk_size, chunk_size);
							error = cudaGetLastError();
							checkCudaError(error,__LINE__);
						}
						for(int chunk_num=0; chunk_num < num_chunks_copied_to_device[which_one]; chunk_num++){
							error = cudaStreamSynchronize(sum_streams[chunk_num]);
							checkCudaError(error,__LINE__);
						}
						new_chunk_size =grid_dim;
						aligner<<<(new_chunk_size*num_chunks_copied_to_device[which_one]-1)/block_dim +1, 
	block_dim, 0, t4_stream>>>(d_data_batch_transformed[which_one], chunk_size, num_chunks_copied_to_device[which_one]);
						error = cudaGetLastError();
						checkCudaError(error,__LINE__);
						error = cudaStreamSynchronize(t4_stream);
						checkCudaError(error,__LINE__);
					}
					for(int i =0; i < num_chunks_copied_to_device[which_one]; i++)
						cudaStreamDestroy(sum_streams[i]);
					free(sum_streams);
					cufftComplex* lad_results;
					error = cudaMallocHost(&lad_results, num_chunks_copied_to_device[t4_pos%3]*sizeof(cufftComplex));
					checkCudaError(error, __LINE__);
					error = cudaMemcpyAsync(lad_results, d_data_batch_transformed[t4_pos%3], 
							num_chunks_copied_to_device[t4_pos%3]*sizeof(cufftComplex),cudaMemcpyDeviceToHost, t4_stream); 
					checkCudaError(error, __LINE__);
					error = cudaStreamSynchronize(t4_stream);
					checkCudaError(error, __LINE__);
					int min_index=0;
					for(int i=0; i<num_chunks_copied_to_device[t4_pos%3]; i++){
						if(lad_results[i].x<lad_results[min_index].x)
							min_index = i;
					}
					/*if(t4_pos == 0)global_min_lad = lad_results[min_index].x;
					else if(lad_results[min_index].x < global_min_lad)
						global_min_lad = lad_results[min_index].x;
					*/	
					if(lad_results[min_index].x < 4200)
						match_found = true;
					error = cudaFreeHost(lad_results);
					checkCudaError(error, __LINE__);
					t4_pos++;
				}while(!t4_stop && !match_found);
				//printf("min_lad: %f\n", global_min_lad);
			}
		}
	}
	cudaFree(d_data_batch[0]);
	checkCudaError(error, __LINE__);
	cudaFree(d_data_batch[1]);
	checkCudaError(error, __LINE__);
	cudaFree(d_data_batch[2]);
	checkCudaError(error, __LINE__);
	cudaFree(d_data_batch_transformed[0]);
	checkCudaError(error, __LINE__);
	cudaFree(d_data_batch_transformed[1]);
	checkCudaError(error, __LINE__);
	cudaFree(d_data_batch_transformed[2]);
	checkCudaError(error, __LINE__);
	cudaStreamDestroy(t2_stream);
	cudaStreamDestroy(t3_stream);
	cudaStreamDestroy(t4_stream);
	cudaFreeHost(data);
	return match_found;
}

std::string captureCommandOutput(std::string command) {
	std::string data;
	FILE * stream;
	const int max_buffer = 256;
	char buffer[max_buffer];
	command.append(" 2>&1");
	
	stream = popen(command.c_str(), "r");
	if (stream) {
		while (!feof(stream))
			if (fgets(buffer, max_buffer, stream) != NULL) data.append(buffer);
				pclose(stream);
	}
	return data;
}

size_t getFileSize(std::string path, int downsampling_factor){
	size_t out;
	std::ifstream in;
	in.open(path, std::ifstream::ate);
	out = in.tellg()/downsampling_factor;
	in.close();
	return out;
}
void readFile(std::string path, cufftComplex** data, size_t* size, int downsampling_factor){
	//here the file is downsampled and shrinked to the greatest number less than size which is only a product of 2 and 3
	size_t original_size = getFileSize(path, downsampling_factor);
	size_t shrinked_size = (size_t)pow(2, log2(original_size));
	while(shrinked_size <= original_size){
		shrinked_size *=3;
		shrinked_size /=2;
	}
	*size = 2*shrinked_size/3;
	cudaMallocHost(data, (*size)*sizeof(cufftComplex));//pinned memory for stream

	size_t tmp_size = *size*downsampling_factor;
	char* tmp = (char*)malloc(tmp_size*sizeof(char));
	FILE* file = fopen(path.c_str(), "r"); 
	fread(tmp, 1, tmp_size, file);
	for(int i =0; i < tmp_size; i += downsampling_factor){
		(*data)[i/downsampling_factor].x = tmp[i];
		(*data)[i/downsampling_factor].y = 0;
	}
	free(tmp);
	fclose(file);
}

unsigned int getNumFiles(std::string list){
	int c=0;
	for(int i=0; i < list.length() ; i ++)
		list.at(i) == '\n' && c++;
	return c;
}

unsigned int calcBatchSize(size_t sample_size){
	size_t work_area_size;
	size_t free_mem;
	cufftEstimate1d(sample_size, CUFFT_R2C, 1, &work_area_size);
	cudaMemGetInfo(&free_mem, NULL);
	//min_batch_size*3*sample_size*sizeof(cufftComplex) +min_batch_size*(1 batch work_area) = free_mem 
	//the multiplication by 6 is because of the 3 size array in findMatches
	unsigned int batch_size = free_mem/(work_area_size + 6*sample_size*sizeof(cufftComplex));
	return batch_size*7/8;
}


void checkCudaError(cudaError_t error, int line){
	if(error != cudaSuccess){
		printf("%s line: %d\n", cudaGetErrorString(error), line);
		exit(EXIT_FAILURE);
	}
}

void checkCufftResult(cufftResult_t result, int line){
	if(result != CUFFT_SUCCESS){
		printf("CUFFT error number %d at line: %d\n", result, line);
		exit(EXIT_FAILURE);
	}
}
	
