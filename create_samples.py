import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import math
import os

# Creates collection of 2-second music samples

fileOrigin = "Recordings"
fileDestination = "samples"
sr_resample = 16000 # resample to 16 kHz
sampleTime = 1 # length of samples in seconds
trimThreshold = 30 # threshold for silence in decibels

# For each file
for filename in os.listdir(fileOrigin):    
    # Convert from stereo to mono and resample to 16 kHz
    y, sr = librosa.load(fileOrigin + '/' + filename, sr=sr_resample, mono=True)
    
    # Remove silence at beginning and end
    yt, index = librosa.effects.trim(y, top_db=trimThreshold)
    
#    # Plot original sample and trimmed sample
#    plt.figure()
#    plt.plot(y)
#    plt.show()
#    
#    plt.figure()
#    plt.plot(yt)
#    plt.show()
    
    # Cut into 2-second samples
    numSamples = math.floor((len(yt)/sr)/sampleTime)
    ysplit = np.array_split(yt[:numSamples*sampleTime*sr],numSamples)
    
    # Save short samples
    for i in range(len(ysplit)):
        sf.write(fileDestination + '/' + filename.split('.')[0] + '_' + str(i) + '.wav', ysplit[i], sr)
