import platform
import sys
import os
from os import listdir
from os.path import isfile, join
from scipy.io import wavfile
print("Starting...")
if (sys.platform == "win32"):
    if platform.architecture()[0]=="64bit":
        onlyfiles = [f for f in listdir(".\\musics") if isfile(join(".\\musics", f))]
        for M in onlyfiles:
            RX="ffmpeg.exe -i \".\\musics\\"+M+"\" \".\\Converted\\"+M[:len(M[0])-4]+"wav\""
            os.system(RX)
        converted = [f for f in listdir(".\\Converted") if isfile(join(".\\Converted", f))]
        for L in converted:
            fs, data = wavfile.read('.\\Converted\\'+L)
            R=data.ravel()
            RS = [str(i) for i in R]
            path=".\\Data\\"+L[:len(L[0]) - 4]+"txt"
            print(path)
            outf=open(path,'w')
            outf.write(" ".join(RS))

    else:
        onlyfiles = [f for f in listdir(".\\musics") if isfile(join(".\\musics", f))]
        for M in onlyfiles:
            RX = "ffmpeg32.exe -i \".\\musics\\" + M + " \" \".\\Converted\\" + M[:len(M[0]) - 4] + "wav\""
            os.system(RX)
        converted = [f for f in listdir(".\\Converted") if isfile(join(".\\Converted", f))]
        for L in converted:
            fs, data = wavfile.read('.\\Converted\\'+L)
            R=data.ravel()
            RS = [str(i) for i in R]
            path=".\\Data\\"+L[:len(L[0]) - 4]+"txt"
            print(path)
            outf=open(path,'w')
            outf.write(" ".join(RS))
elif sys.platform=="darwin":
    onlyfiles = [f for f in listdir("./musics") if isfile(join("./musics", f))]
    for M in onlyfiles:
        RX = "ffmpeg -i ./musics/" + M + " ./Converted/" + M[:len(M[0]) - 4] + "wav"
        os.system(RX)
    converted = [f for f in listdir("./Converted") if isfile(join("./Converted", f))]
    for L in converted:
        fs, data = wavfile.read('./Converted/' + L)
        R = data.ravel()
        RS = [str(i) for i in R]
        path = "./Data/" + L[:len(L[0]) - 4] + "txt"
        print(path)
        outf = open(path, 'w')
        outf.write(" ".join(RS))
elif sys.platform=="linux":
    onlyfiles = [f for f in listdir("./musics") if isfile(join("./musics", f))]
    for M in onlyfiles:
        RX = "ffmpeg -i ./musics/" + "\'"+M+"\'" + " ./Converted/" + "\'"+M[:len(M[0]) - 5]+"\'" + ".wav"
        os.system(RX)
    converted = [f for f in listdir("./Converted") if isfile(join("./Converted", f))]
    for L in converted:
        print(L)
        fs, data = wavfile.read('./Converted/' + L)
        R = data.ravel()
        RS = [str(i) for i in R]
        path = "./Data/" + L[:len(L[0]) - 4] + "txt"
        print(path)
        outf = open(path, 'w')
        outf.write(" ".join(RS))

else:
    print("undefined")
