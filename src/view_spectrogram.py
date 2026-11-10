import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('../data/test_segment.pkl', 'rb') as file:
    segment = pickle.load(file)

S = segment['spectrogram_db']
time = segment['time_edges']
freq = segment['frequency_edges']

dynamic_range = segment['metadata']['dynamic_range_db'] // 2

vmax = np.max(S)
vmin = vmax - dynamic_range

plt.figure(figsize=(14, 6))

plt.pcolormesh(
    time, freq, S,
    shading='auto',
    cmap='magma',
    vmin=vmin,
    vmax=vmax
)

plt.colorbar(label='Amplitude (dB)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram')

plt.ylim(0, 600)
plt.tight_layout()
plt.show()