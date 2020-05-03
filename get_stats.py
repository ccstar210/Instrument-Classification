# -*- coding: utf-8 -*-
"""
Created on Sat May  2 17:49:42 2020

@author: Ava
"""

#%%
# Get statistics for the data distribution

import os
import librosa


def detLabel(filename):
    #determine the label of the recording
    if "cello" in filename: 
        label = 0
    elif "church" in filename:
        label = 1
    elif "clarinet" in filename:
        label = 2
    elif "flute" in filename:
        label = 3
    elif "guitar" in filename:
        label = 4
    elif "harp" in filename:
        label = 5
    elif "marimba" in filename:
        label = 6
    elif "perldrop" in filename:
        label = 7
    elif "piano" in filename:
        label = 8
    elif "synlead3" in filename:
        label = 9
    else: #violin
        label = 10
    return label

def detSong(filename):
    #determine the label of the recording
    if "babillarde" in filename: 
        song = 0
    elif "let" in filename:
        song = 1
    elif "sea" in filename:
        song = 2
    elif "styrienne" in filename:
        song = 3
    else: # tarantella
        song = 4
    return song

samplesOrigin = "samples"
songsOrigin = "Recordings"
numLabels = 11
numSongs = 5
numFiles = {}
lengthInstr = {}
lengthSong = {}
hitsSong = {}

# Initialize dicts
for i in range(numLabels):
    numFiles[i] = 0
    lengthInstr[i] = 0
for j in range(numSongs):
    lengthSong[j] = 0
    hitsSong[j] = 0

# Get stats
for filename in os.listdir(samplesOrigin):
    label = detLabel(filename)
    y, sr = librosa.load(samplesOrigin + '/' + filename)    
    numFiles[label] += 1
    lengthInstr[label] += len(y)/sr

    
for filename in os.listdir(songsOrigin):
    song = detSong(filename)
    y, sr = librosa.load(songsOrigin + '/' + filename)    
    lengthSong[song] += len(y)/sr
    hitsSong[song] += 1
    
for x in range(numLabels):
    print('Label ' + str(x))
    print('Number of files ' + str(numFiles[x]))
    print('Total length ' + str(lengthInstr[x]) + ' seconds')
    print('')
    
for y in range(numSongs):
    print('Song ' + str(y))
    print('Average length ' + str(lengthSong[y]/hitsSong[y]) + ' seconds')
    print('')
    
print('Total data length ' + str(sum(lengthSong.values())/60) + ' minutes')
