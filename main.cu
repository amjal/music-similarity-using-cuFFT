#include<cuda_runtime.h>
#include<iostream>
#include<stdio.h>
#include<string>
#include<fstream>
#include<stdlib.h>
#include<omp.h>

enum data {wav, txt};

typedef struct  WAV_HEADER{
    char                RIFF[4];        // RIFF Header      Magic header
    unsigned int        chunkSize;      // RIFF Chunk Size  
    char                WAVE[4];        // WAVE Header      
    char                fmt[4];         // FMT header       
    unsigned int        subchunk1Size;  // Size of the fmt chunk                                
    unsigned short      audioFormat;    // Audio format 1=PCM,6=mulaw,7=alaw, 257=IBM Mu-Law, 258=IBM A-Law, 259=ADPCM 
    unsigned short      numOfChan;      // Number of channels 1=Mono 2=Sterio                   
    unsigned int        samplesPerSec;  // Sampling Frequency in Hz                             
    unsigned int        bytesPerSec;    // bytes per second 
    unsigned short      blockAlign;     // 2=16-bit mono, 4=16-bit stereo 
    unsigned short      bitsPerSample;  // Number of bits per sample      
    char                subchunk2ID[4]; // "data"  string   
    unsigned int        subchunk2Size;  // Sampled data length    

}wav_hdr; 
const unsigned int HEADER_SIZE = sizeof(wav_hdr);

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
//TODO should I open in binary mode?
char* get_wav_data(std::string wav_path){
	wav_hdr wav_header;
	FILE* wav_file;
	wav_file = fopen(wav_path.c_str(), "r");
	if(wav_file == NULL){
		printf("Error: Couldn't open file %s\n", wav_path.c_str());
		exit(EXIT_FAILURE);
	}
	fread(&wav_header, HEADER_SIZE, 1, wav_file);
	char* data = (char*)malloc(wav_header.subchunk2Size*sizeof(char));
	fread(data, wav_header.subchunk2Size, 1, wav_file);
	fclose(wav_file);
	return data;
}

char* get_txt_data(std::string txt_path){
	std::ifstream in(txt_path.c_str(), std::ifstream::ate | std::ifstream::binary);
	size_t size = in.tellg();
	in.close();
	FILE* txt_file;
	txt_file = fopen(txt_path.c_str(), "r");
	if(txt_file == NULL){
		printf("Error: Couldn't open file %s\n", txt_path.c_str());
		exit(EXIT_FAILURE);
	}
	char* data = (char*)malloc(size*sizeof(char));
	fread(data, size,1,txt_file);
	fclose(txt_file);
	return data;
}

	
char** extract_path_data(std::string input_path, std::string files, data type){
	int c =0;
	for(int i =0; i < files.length() ; i ++)
		files.at(i) == '\n' && c++;
	char** all_data = (char**)malloc(c*sizeof(char*));
	std::string delimiter = "\n";
	size_t pos = 0;
	std::string file;
	float start = omp_get_wtime();		
	while ((pos = files.find(delimiter)) != std::string::npos){
		file = files.substr(0, pos);
		files.erase(0, pos + delimiter.length());
		std::string s = input_path;
		s.append(file);
		c--;
		switch(type){
			case wav:
				all_data[c] = get_wav_data(s);
				printf("%s: data extracted\n", file.c_str());
				break;
			case txt:
				all_data[c] = get_txt_data(s);
				printf("%s: data read\n", file.c_str());
				break;
		}
	}
	float end = omp_get_wtime();
	printf("time elapsed: %f", end - start);
	return all_data;
}

int main(int argc, char* argv[]){
	if(argc != 3){
		printf("incorrect arguments\n");
		exit(EXIT_FAILURE);
	}
	std::string command = "ls ";
	command.append(argv[1]);
	char** all_input_data = extract_path_data(argv[1], run_command(command), data::txt);
	command = "ls ";
	command.append(argv[2]);
	char** all_music_data = extract_path_data(argv[2], run_command(command), data::wav);
	return 0;
}

