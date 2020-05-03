import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Plot the spectrum for two instruments
numInstr = 2
nameInstr = ['Piano', 'Trumpet']
filenames = ['example_piano.wav', 'example_trumpet.wav']
matplotlib.rcParams.update({'font.size' : 20}) # Increase font size
plt.figure(figsize=(18, 8))

for i in range(numInstr):
    plt.subplot(1, 2, i+1)
    y, sr = librosa.load(filenames[i])
    CQT = librosa.amplitude_to_db(np.abs(librosa.cqt(y, sr=sr, n_bins=100)), ref=np.max)
    librosa.display.specshow(CQT, x_axis='time', y_axis='cqt_hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(nameInstr[i] + ' spectrogram for A4 (440 Hz)')    
    plt.tight_layout()

plt.show()
