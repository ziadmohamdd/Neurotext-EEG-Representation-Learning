from fractions import Fraction
import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, resample_poly

def filter_resample(x, sfreq, target_sfreq=200, low=0.5, high=45., notch=50., already_filtered=False):
    x = np.asarray(x, dtype=np.float32)
    if not already_filtered:
        if not 0 < low < high < sfreq / 2:
            raise ValueError("Bandpass must lie strictly below input Nyquist")
        x = sosfiltfilt(butter(4, [low, high], fs=sfreq, btype="bandpass", output="sos"), x, axis=-1)
        if notch and notch < sfreq / 2:
            b, a = iirnotch(notch, 30, fs=sfreq)
            x = filtfilt(b, a, x, axis=-1)
    # Polyphase resampling includes an anti-alias lowpass even for previously filtered signals.
    if sfreq != target_sfreq:
        ratio = Fraction(str(target_sfreq)) / Fraction(str(sfreq))
        x = resample_poly(x, ratio.numerator, ratio.denominator, axis=-1)
    return np.asarray(x, dtype=np.float32)
