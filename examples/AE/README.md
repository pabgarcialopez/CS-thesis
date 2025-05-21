# Autoencoder Examples

This directory contains a set of trained autoencoder experiments.  
Each numbered folder corresponds to a row in **Table 1: Autoencoder Training Parameters with Accuracy** of the main report.

Inside each `NN/` folder:

1. **`examples/`**  
   Contains pairs of original and reconstructed audio files.  
   - `X.wav` is the input.  
   - `Xr.wav` is the autoencoder’s reconstruction of `X.wav`.

2. **`autoencoder.pth`**  
   The saved PyTorch model of the trained autoencoder.

3. **`losses.png`**  
   A plot showing training (and validation) loss curves across epochs.

4. **`configs.txt`**  
   JSON-formatted configuration parameters used for that experiment.
