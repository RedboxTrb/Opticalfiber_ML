"""
Professional visualization tools for optical communications.
Publication-quality plots: eye diagrams, constellation diagrams, BER curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec


def plot_eye_diagram(signal, samples_per_symbol=4, num_traces=100, title="Eye Diagram", ax=None):
    """
    Create professional eye diagram.

    Args:
        signal: Received signal
        samples_per_symbol: Samples per symbol
        num_traces: Number of symbol traces to overlay
        title: Plot title
        ax: Optional matplotlib axis to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Extract traces
    trace_length = 2 * samples_per_symbol
    for i in range(min(num_traces, len(signal) // trace_length)):
        start = i * samples_per_symbol
        trace = signal[start:start + trace_length].real
        if len(trace) == trace_length:
            time = np.arange(len(trace)) / samples_per_symbol
            ax.plot(time, trace, 'b-', alpha=0.1, linewidth=1)

    # Eye opening markers
    ax.axvline(1.0, color='r', linestyle='--', alpha=0.5, label='Sampling point')

    ax.set_xlabel('Time (Symbol Periods)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    if ax is None:
        plt.tight_layout()
    return fig


def plot_constellation(signal_points, symbol_points=None, title="Constellation Diagram"):
    """
    Create professional constellation diagram.

    Args:
        signal_points: Received signal samples (complex)
        symbol_points: Ideal symbol locations (complex, optional)
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot received points
    ax.scatter(signal_points.real, signal_points.imag,
               alpha=0.3, s=10, c='blue', label='Received symbols')

    # Plot ideal constellation
    if symbol_points is not None:
        ax.scatter(symbol_points.real, symbol_points.imag,
                   s=200, c='red', marker='x', linewidths=3,
                   label='Ideal symbols', zorder=10)

        # Draw decision boundaries
        for point in symbol_points:
            circle = Circle((point.real, point.imag), radius=0.3,
                            fill=False, edgecolor='red', linestyle='--',
                            alpha=0.3)
            ax.add_patch(circle)

    ax.set_xlabel('In-Phase', fontsize=12)
    ax.set_ylabel('Quadrature', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend()

    plt.tight_layout()
    return fig


def plot_ber_curves(snr_range, ber_results, title="BER vs SNR Performance"):
    """
    Create professional BER curves with multiple models.

    Args:
        snr_range: List of SNR values
        ber_results: Dict of {model_name: [ber_values]}
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    markers = ['o', 's', '^', 'd', 'v', '<', '>']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    for idx, (model_name, ber_values) in enumerate(ber_results.items()):
        marker = markers[idx % len(markers)]
        color = colors[idx % len(colors)]

        ax.semilogy(snr_range, ber_values,
                    marker=marker, linestyle='-', linewidth=2.5,
                    markersize=10, label=model_name, color=color)

    # Reference lines
    ax.axhline(1e-3, color='gray', linestyle=':', alpha=0.5, label='10⁻³ threshold')
    ax.axhline(1e-6, color='gray', linestyle=':', alpha=0.5, label='10⁻⁶ threshold')

    ax.set_xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim([1e-7, 1])

    plt.tight_layout()
    return fig


def plot_training_history(history_dict, title="Training History"):
    """
    Plot training and validation curves for multiple models.

    Args:
        history_dict: Dict of {model_name: {'train_loss': [...], 'val_acc': [...]}}
        title: Plot title
    """
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (model_name, history) in enumerate(history_dict.items()):
        color = colors[idx % len(colors)]

        epochs = range(1, len(history['train_loss']) + 1)

        # Loss plot
        ax1.plot(epochs, history['train_loss'], linestyle='-',
                 color=color, linewidth=2, label=f'{model_name}')

        # Accuracy plot
        ax2.plot(epochs, history['val_acc'], linestyle='-',
                 color=color, linewidth=2, label=f'{model_name}')

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Accuracy', fontsize=12)
    ax2.set_title('Validation Accuracy', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=15, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_model_comparison(results_df, title="Model Performance Comparison"):
    """
    Create comprehensive model comparison visualization.

    Args:
        results_df: DataFrame with columns ['Model', 'BER', 'Accuracy', 'Inference_Time']
        title: Plot title
    """
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    models = results_df['Model'].values
    x_pos = np.arange(len(models))

    # BER comparison
    ax1.bar(x_pos, results_df['BER'].values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_ylabel('Bit Error Rate', fontsize=11, fontweight='bold')
    ax1.set_title('BER (Lower is Better)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')

    # Accuracy comparison
    ax2.bar(x_pos, results_df['Accuracy'].values * 100, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Accuracy (Higher is Better)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([90, 100])

    # Inference time comparison
    ax3.bar(x_pos, results_df['Inference_Time'].values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax3.set_ylabel('Inference Time (ms)', fontsize=11, fontweight='bold')
    ax3.set_title('Computational Cost (Lower is Better)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')

    fig.suptitle(title, fontsize=15, fontweight='bold')
    plt.tight_layout()
    return fig


def save_all_figures(figures_dict, save_dir='figures'):
    """
    Save all figures to directory.

    Args:
        figures_dict: Dict of {filename: figure_object}
        save_dir: Directory to save figures
    """
    from pathlib import Path
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for filename, fig in figures_dict.items():
        filepath = save_path / f"{filename}.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {filepath}")

    print(f"\n📁 All figures saved to: {save_path}")
