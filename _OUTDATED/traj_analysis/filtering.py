
from scipy.signal import butter, filtfilt

# Low-pass filter to reduce noise in raw data
def lowpass(signal, fs, cutoff=0.6, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, signal, axis=0)