"""Modulation schemes for optical communications."""

import numpy as np
from scipy.signal import upfirdn

def rrc_filter(num_taps=101, rolloff=0.5, sps=4):
    """Root-raised cosine pulse shaping filter."""
    t = np.arange(-num_taps//2, num_taps//2+1) / sps

    h = np.zeros(len(t))
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1 - rolloff + (4*rolloff/np.pi)
        elif abs(ti) == 1/(4*rolloff):
            h[i] = (rolloff/np.sqrt(2)) * (((1+2/np.pi)*np.sin(np.pi/(4*rolloff))) + ((1-2/np.pi)*np.cos(np.pi/(4*rolloff))))
        else:
            h[i] = (np.sin(np.pi*ti*(1-rolloff)) + 4*rolloff*ti*np.cos(np.pi*ti*(1+rolloff))) / (np.pi*ti*(1-(4*rolloff*ti)**2))

    h = h / np.sqrt(np.sum(h**2))
    return h

class BinaryNRZ:
    """Binary NRZ modulation."""

    def __init__(self, sps=4):
        self.sps = sps  # Samples per symbol
        self.levels = 2
        self.rrc = rrc_filter(sps=sps)

    def modulate(self, bits):
        """Modulate bits to signal."""
        symbols = 2*bits - 1  # Map {0,1} to {-1,+1}
        signal = upfirdn(self.rrc, symbols, up=self.sps)
        return signal

    def demodulate(self, signal):
        """Simple threshold demodulation."""
        # Matched filter
        matched = np.convolve(signal.real, self.rrc[::-1], mode='same')
        # Sample at symbol centers
        samples = matched[len(self.rrc)//2::self.sps][:len(signal)//self.sps]
        # Threshold
        bits = (samples > 0).astype(int)
        return bits

class PAM4:
    """4-PAM modulation."""

    def __init__(self, sps=4):
        self.sps = sps
        self.levels = 4
        self.symbols = np.array([-3, -1, 1, 3])
        self.rrc = rrc_filter(sps=sps)

    def modulate(self, bits):
        """Modulate bits to 4-PAM signal."""
        # Convert bits to symbols (2 bits per symbol)
        bits_reshaped = bits.reshape(-1, 2)
        symbol_indices = bits_reshaped[:, 0] * 2 + bits_reshaped[:, 1]
        symbols = self.symbols[symbol_indices]

        # Pulse shape
        signal = upfirdn(self.rrc, symbols, up=self.sps)
        return signal

    def demodulate(self, signal):
        """ML demodulation for 4-PAM."""
        # Matched filter
        matched = np.convolve(signal.real, self.rrc[::-1], mode='same')
        # Sample at symbol centers
        samples = matched[len(self.rrc)//2::self.sps][:len(signal)//self.sps]

        # ML decision
        symbol_indices = np.argmin(np.abs(samples[:, None] - self.symbols[None, :]), axis=1)

        # Convert to bits
        bits = np.zeros(len(symbol_indices) * 2, dtype=int)
        bits[0::2] = symbol_indices // 2
        bits[1::2] = symbol_indices % 2

        return bits
