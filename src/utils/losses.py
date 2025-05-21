import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self,
                 orig_n_fft: int,
                 orig_hop_length: int,
                 orig_win_length: int,
                 fft_sizes=(512,1024,2048),
                 hop_sizes=(120,240,480),
                 win_lengths=(512,1024,2048),
                 weight=1.0):
        super().__init__()
        # store your “original” STFT parameters for the first ISTFT
        self.orig_n_fft       = orig_n_fft
        self.orig_hop_length  = orig_hop_length
        self.orig_win_length  = orig_win_length
        self.orig_window      = torch.hann_window(orig_win_length)

        self.weight = weight
        self.configs = list(zip(fft_sizes, hop_sizes, win_lengths))
        # precompute all windows for the new STFTs
        self.windows = {
            n_fft: torch.hann_window(n_fft)
            for n_fft,_,_ in self.configs
        }

    def forward(self, x, x_hat):
        """
        x, x_hat: [B,2,F,T] (mag, phase)
        """
        # unpack mag/phase
        mag_x, ph_x     = x   [:,0], x   [:,1]
        mag_xh, ph_xh   = x_hat[:,0], x_hat[:,1]

        # rebuild complex STFTs at your *training* resolution
        S_x  = mag_x  * torch.exp(1j * ph_x)
        S_xh = mag_xh * torch.exp(1j * ph_xh)

        # 1) invert to waveform *once* at your original STFT settings
        wav   = torch.istft(
            S_x,
            n_fft=self.orig_n_fft,
            hop_length=self.orig_hop_length,
            win_length=self.orig_win_length,
            window=self.orig_window.to(x.device),
            center=False,
            onesided=False,
        )
        wav_h = torch.istft(
            S_xh,
            n_fft=self.orig_n_fft,
            hop_length=self.orig_hop_length,
            win_length=self.orig_win_length,
            window=self.orig_window.to(x.device),
            center=False,
            onesided=False,
        )

        sc_loss  = 0.0
        mag_loss = 0.0

        # 2) for each multires config, re‐STFT the *waveforms*
        for n_fft, hop_length, win_length in self.configs:
            window = self.windows[n_fft].to(x.device)

            X   = torch.stft(
                wav,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                return_complex=True,
                center=False,
            )
            Xh  = torch.stft(
                wav_h,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                return_complex=True,
                center=False,
            )

            mag_X  = X .abs()
            mag_Xh = Xh.abs()

            # spectral convergence
            sc_loss  += F.l1_loss(mag_X, mag_Xh) / (mag_X.mean() + 1e-8)
            # log magnitude difference
            mag_loss += F.l1_loss(torch.log(mag_X + 1e-7),
                                   torch.log(mag_Xh + 1e-7))

        return self.weight * (sc_loss + mag_loss) / len(self.configs)
