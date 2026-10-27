from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def robustness_plot(rows, path):
    fig, ax = plt.subplots(figsize=(7,4))
    for name in sorted({r["corruption"] for r in rows}):
        points = sorted([r for r in rows if r["corruption"] == name], key=lambda r: r["severity"])
        ax.plot([r["severity"] for r in points], [r["top1"] for r in points], marker="o", label=name)
    ax.set(xlabel="Signal degradation severity", ylabel="Top-1 sentence retrieval")
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)

def signal_plot(x, sfreq, path):
    fig, axes = plt.subplots(2,1,figsize=(8,5))
    import numpy as np
    axes[0].plot(np.arange(x.shape[-1])/sfreq, x[:min(3,len(x))].T)
    axes[0].set(xlabel="Time (s)", ylabel="EEG (microvolts)")
    axes[1].psd(x[0], Fs=sfreq, NFFT=min(256,x.shape[-1]))
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path,dpi=150)
    plt.close(fig)
