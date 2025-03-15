import matplotlib.pyplot as plt

def plot_waveform(waveform):
    plt.figure(figsize=(10, 4))
    plt.title("Original Waveform")
    plt.plot(waveform[0].numpy())
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()

def plot_spectogram(mel_spec_db):
    # 6) Plot del espectrograma
    plt.figure(figsize=(10, 4))
    plt.title("Mel Spectrogram")
    # mel_spec_db.shape = [1, 64, time_frames], erase channel dimension with .squeeze(0)
    plt.imshow(mel_spec_db.squeeze(0).numpy(), origin="lower", aspect="auto")
    plt.colorbar(label="dB")
    plt.xlabel("Time frames")
    plt.ylabel("Frequency Bins (Mel)")
    plt.show()