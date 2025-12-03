"""Simple optical fiber channel simulator."""

import numpy as np
from scipy.fft import fft, ifft, fftfreq

def chromatic_dispersion(signal, beta2, distance, sample_rate):
    """
    Apply chromatic dispersion using frequency domain.

    Args:
        signal: Input signal
        beta2: Dispersion parameter (ps^2/km), typically ~16-17 for SMF
        distance: Fiber length (km)
        sample_rate: Sampling rate (Hz)
    """
    N = len(signal)
    freq = fftfreq(N, d=1/sample_rate)
    omega = 2 * np.pi * freq

    # Dispersion transfer function: H(ω) = exp(-j*beta2*ω^2*L/2)
    H = np.exp(-1j * beta2 * 1e-24 * omega**2 * distance * 1e3 / 2)

    # Apply in frequency domain
    signal_freq = fft(signal)
    signal_dispersed = ifft(signal_freq * H)

    return signal_dispersed

def add_awgn(signal, snr_db):
    """Add AWGN noise for given SNR."""
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))
    return signal + noise

def fiber_channel(signal, distance_km=100, beta2=17, snr_db=20, sample_rate=50e9):
    """Complete fiber channel with dispersion and noise."""
    # Apply chromatic dispersion
    signal_dispersed = chromatic_dispersion(signal, beta2, distance_km, sample_rate)

    # Add noise
    signal_noisy = add_awgn(signal_dispersed, snr_db)

    return signal_noisy
