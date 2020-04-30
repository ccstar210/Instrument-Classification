#%%
# Creates a list of each filename and instrument and saves it as a csv file

import os
import csv

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

fileOrigin = "samples"
dataFile = "data.csv"

with open(dataFile, 'w', newline='') as csvfile:
    fileWriter = csv.writer(csvfile, delimiter=',')
    fileWriter.writerow(['filename', 'instrument'])
    for filename in os.listdir(fileOrigin):
        fileWriter.writerow([filename, detLabel(filename)])
