import matplotlib.pyplot as plt
import torch
from IPython.display import Audio

def plot_waveform(waveform):
    plt.figure(figsize=(10, 4))
    plt.title("Original Waveform")
    plt.plot(waveform[0].numpy())
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()

def plot_mel_spectogram(mel_spec_db):
    plt.figure(figsize=(10, 4))
    plt.title("Mel Spectrogram")
    plt.imshow(mel_spec_db.squeeze(0).numpy(), origin="lower", aspect="auto")
    plt.colorbar(label="dB")
    plt.xlabel("Time frames")
    plt.ylabel("Frequency Bins (Mel)")
    plt.show()

def listen(waveform, sample_rate=16000):
    """
    Utility to play audio in a Jupyter notebook.
    waveform: torch.Tensor or np.ndarray, shape [1, time] or [time].
    """
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.squeeze().cpu().numpy()
    else:
        waveform = waveform.squeeze()
    return Audio(waveform, rate=sample_rate)