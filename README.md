# Optical Fiber ML Equalization - Baseline Version

Deep Learning for Optical Fiber Channel Equalization using CNN and Transformer architectures.

## Features

- **Interactive Streamlit Demo** with real-time visualizations
- **Multiple Modulation Schemes**: Binary NRZ, 4-PAM, QPSK, 16-QAM, 64-QAM
- **CNN Baseline Model**: Convolutional neural network equalizer
- **Transformer Model**: Attention-based equalizer with superior performance
- **Live Performance Metrics**: SNR, EVM, BER comparison
- **3-Way Comparison**: Without ML vs CNN vs Transformer

## Performance Results

- **SNR Improvement**: Up to 9.1 dB gain with Transformer
- **EVM Reduction**: 87% improvement over unequalized signals
- **BER**: 3-4 orders of magnitude improvement

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch demo
LAUNCH_DEMO.bat
```

## Project Structure

```
├── demo_app.py              # Main Streamlit application
├── models.py                # CNN and Transformer architectures
├── channel.py               # Optical fiber channel simulation
├── modulation.py            # Modulation schemes
├── visualizations.py        # Eye diagram, constellation plots
├── data/                    # Training datasets
├── trained_models/          # Pre-trained model weights
└── LAUNCH_DEMO.bat         # Demo launcher
```

## Customizable Parameters

- Fiber distance (0-250 km)
- SNR (5-30 dB)
- Dispersion coefficient (5-30 ps²/km)
- Noise figure (2-12 dB)
- Fiber loss (0.1-0.5 dB/km)
- Nonlinearity factor (0-5)

## Models

### CNN Baseline
- Convolutional layers for feature extraction
- ~150K parameters
- Moderate performance improvement

### Transformer (Novel)
- Multi-head self-attention mechanism
- ~100K parameters (fewer than CNN!)
- Superior long-range ISI compensation
- Best performance across all metrics

## Version

**Baseline v1.0** - Initial working version with 3-way comparison and performance metrics

## Future Improvements

- Additional architectures (LSTM, Hybrid models)
- Enhanced training techniques
- Real-time model inference
- Extended modulation support

## License

MIT License
