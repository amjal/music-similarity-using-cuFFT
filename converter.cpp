#include<fstream>

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



int main(int argc, char* argv[]){
	wav_hdr wav_header;
	FILE* wav_file;
	wav_file = fopen(argv[1], "r");
	if(wav_file == NULL){
		printf("Error: Couldn't open file %s\n", argv[1]);
		exit(EXIT_FAILURE);
	}
	fread(&wav_header, HEADER_SIZE, 1, wav_file);
	char* data = (char*)malloc(wav_header.subchunk2Size*sizeof(char));
	fread(data, wav_header.subchunk2Size, 1, wav_file);
	printf("%d\n", wav_header.subchunk2Size);
	fclose(wav_file);
	FILE* txt_file;
	txt_file = fopen(argv[2], "w");
	fwrite(data, wav_header.subchunk2Size, 1, txt_file);
	fclose(txt_file);
	return 0;
}


