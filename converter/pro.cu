#include<fstream>
#include<stdlib.h>
#include<string>
#include<stdio.h>
#include<omp.h>
#include<cuda_runtime.h>
#include<cufft.h>
#include<cufftXt.h>
#include<math.h>

void copySamples(char* sample_songs_directory);

//provide the path to complete songs' data and function below will fetch the data and some other information about them
void prepareSamples(char* samples_directory, cufftComplex*** samples, 
		char*** sample_names, size_t** sample_sizes, unsigned int* num_samples);

//runs command and returns its output as a string
std::string captureCommandOutput(std::string command);

//gets a list of files(from a command output) and returns how many file there are in that list
unsigned int getNumFiles(std::string files_list);

/*reads file with path 'path' and fill the information. Assuming the sampling rate for our data is 44kHz and that input
data are songs, furthermore assuming that frequency of musical sound doesn't have frequencies greater than 4.4kHz (which
is a good assumption), we can downsample our input file with up to a factor of 5 without loss of accuracy.
*/
void readFile(std::string path, cufftComplex** data, size_t* size, int downsampling_factor);

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

	cufftComplex** samples;
	char** sample_names;
	size_t* sample_sizes;
	unsigned int num_samples;
	prepareSamples(argv[2], &samples, &sample_names, &sample_sizes, &num_samples);
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

void prepareSamples(char* data_files_directory, cufftComplex*** samples, 
		char*** sample_names, size_t** sample_sizes, unsigned int* num_samples){
	//first get a list of all files in that directory
	std::string command = "ls ";
	command.append(data_files_directory);
	std::string data_files_list = captureCommandOutput(command);

	//find how many data files there are
	*num_samples = getNumFiles(data_files_list);

	//do memory initializations 
	*samples = (cufftComplex**)malloc(*num_samples*sizeof(cufftComplex*));
	*sample_names = (char**)malloc(*num_samples*sizeof(char*));
	*sample_sizes = (size_t*)malloc(*num_samples*sizeof(size_t));

	//fetch all data and fill the arrays
	std::string delim = "\n";
	size_t pos = 0;
	std::string data_file_name;
	unsigned int data_file_no = 0;
	while((pos = data_files_list.find(delim)) != std::string::npos){
		data_file_name = data_files_list.substr(0,pos);
		data_files_list.erase(0, pos+ delim.length());
		
		(*sample_names)[data_file_no] = (char*)malloc(data_file_name.length()*sizeof(char));
		strcpy((*sample_names)[data_file_no], data_file_name.c_str());

		std::string full_path = data_files_directory;
		full_path.append("/");
		full_path.append(data_file_name);
		readFile(full_path, &(*samples)[data_file_no], &(*sample_sizes)[data_file_no], 10);
		printf("%s: data read\n", data_file_name.c_str());
	}	
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

void readFile(std::string path, cufftComplex** data, size_t* size, int downsampling_factor){
	std::ifstream in;
	in.open(path, std::ifstream::ate);
	*size = in.tellg();
	(*data) = (cufftComplex*)malloc((*size)*sizeof(cufftComplex));
	char c;
	size_t pos =0;
	in.seekg(pos);
	while(in.get(c)){
		(*data)[pos/downsampling_factor].x = c;
		(*data)[pos/downsampling_factor].y = 0;
		pos += downsampling_factor;
		in.seekg(pos);
	}
	in.close();
}

unsigned int getNumFiles(std::string list){
	int c=0;
	for(int i=0; i < list.length() ; i ++)
		list.at(i) == '\n' && c++;
	return c;
}
	
