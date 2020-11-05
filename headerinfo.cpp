#include "soundfile-2.2/include/soundfile.h"
#include <stdlib.h>

#include <iostream>
using namespace std;

int main(int argc, char** argv) {
   Options options;
   options.process(argc, argv);
   SoundFileRead  insound;

   int i;
   for (i=1; i<=options.getArgCount(); i++) {
      insound.setFile(options.getArg(i));
      cout << "Filename:        " << insound.getFilename() << "\n";
      cout << insound << endl;

   }

   return 0;
}
