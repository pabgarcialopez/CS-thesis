import torch.nn as nn
from torch.distributions import Normal, ContinuousBernoulli
from src.utils.models import *


class Encoder(nn.Module):
    def __init__(self, input_size, latent_dim, channels, variational=False):
        super().__init__()

        # Is the encoder serving an autoencoder or a variational autoencoder?
        self.variational = variational

        conv_kernel_size = (3, 3)
        conv_stride = (2, 2)  # Stride 2 for downsampling
        conv_padding = (1, 1)

        self.sizes = [input_size]
        current_size = input_size

        blocks = []
        for i in range(1, len(channels)):
            conv = nn.Conv2d(channels[i - 1], channels[i], kernel_size=conv_kernel_size, stride=conv_stride,padding=conv_padding)
            current_size = compute_conv2D_output_size(current_size, conv_kernel_size, conv_stride, conv_padding)
            self.sizes.append(current_size)

            if variational: # BatchNorm can hurt VAE
                block = nn.Sequential(conv, nn.ReLU())
            else:
                block = nn.Sequential(conv, nn.ReLU(), nn.BatchNorm2d(channels[i]))

            blocks.append(block)
        self.encoder = nn.Sequential(*blocks)

        self.flatten = nn.Flatten()

        # If varational=True, fc1 represents the mean layer
        self.fc1 = nn.Linear(channels[-1] * current_size[0] * current_size[1], latent_dim)

        # fc2 represents the log_var layer
        self.fc2 = nn.Linear(channels[-1] * current_size[0] * current_size[1], latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)

        if self.variational:
            mu = self.fc1(x)
            log_var = self.fc2(x)
            return x, mu, log_var
        
        x = self.fc1(x)
        return x

    def get_sizes(self):
        return self.sizes


class Decoder(nn.Module):
    def __init__(self, sizes, latent_dim, channels, variational=False):
        super().__init__()

        kernel_size = (3, 3)
        stride = (2, 2)
        padding = (1, 1)
        output_padding = 0

        rev_channels = list(reversed(channels))
        rev_sizes = list(reversed(sizes))

        expected_size = rev_sizes[0]
        self.fc = nn.Linear(latent_dim, rev_channels[0] * expected_size[0] * expected_size[1])
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(rev_channels[0], expected_size[0], expected_size[1]))

        deconv_blocks = []
        for i in range(1, len(rev_sizes)):
            deconv = nn.ConvTranspose2d(
                rev_channels[i - 1],
                rev_channels[i],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding
            )

            # Don't add RELU in last layer.

            if i < len(rev_sizes) - 1: # or not variational:
                block = nn.Sequential(deconv, nn.ReLU())
            else:
                block = nn.Sequential(deconv)

            deconv_blocks.append(block)
        self.decoder = nn.Sequential(*deconv_blocks)

    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.decoder(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, input_size, latent_dim):
        super().__init__()

        self.input_size = input_size
        channels = [2, 16, 32, 64]

        self.encoder = Encoder(input_size, latent_dim, channels)
        sizes = self.encoder.get_sizes()
        self.decoder = Decoder(sizes, latent_dim, channels)

        print("Encoder: ", self.encoder)
        print("Decoder: ", self.decoder)

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        target_height, target_width = x.shape[2], x.shape[3]
        reconstructed = adjust_shape(reconstructed, (target_height, target_width))
        return reconstructed

class VAE(nn.Module):
    def __init__(self, input_size, latent_dim, conditional=False, num_classes=None):
        super().__init__()

        self.input_size = input_size
        self.latent_dim = latent_dim
        self.conditional = conditional
        self.num_classes = num_classes

        self.normal = Normal(0, 1)
        self.normal.loc = self.normal.loc.cuda()
        self.normal.scale = self.normal.scale.cuda()

        self.channels = [2, 16, 32, 64] 
        self.encoder = Encoder(input_size, latent_dim, self.channels, variational=True)
        sizes = self.encoder.get_sizes()
        self.decoder = Decoder(sizes, latent_dim, self.channels, variational=True)  

        if conditional:
            H, W = input_size
            C = self.channels[0]
            self.label_projector_encoder = nn.Sequential(
                nn.Linear(num_classes, C * H * W),
                nn.ReLU()
            )
            self.label_projector_decoder = nn.Sequential(
                nn.Linear(num_classes, latent_dim),
                nn.ReLU()
            )

        print("Encoder:", self.encoder)
        print("Decoder:", self.decoder)

    def reparameterization(self, mean, log_var):
        
        std = torch.exp(0.5 * log_var)
        eps = self.normal.sample(mean.shape)
        Z = mean + eps * std
        return Z.to(device=std.device)
    
    def compute_kld(self, mu, logvar):
        kld = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=0)
        return kld

    def calculate_elbo(self, x, sigma=1.0):
        """
        Compute per-sample ELBO = log p(x|z) - KL[q(z|x) || p(z)]
        where p(x|z) = Normal(decoder(z), sigma^2 I)
        Returns tensor of shape (B,)
        """
        B, C, H, W = x.shape

        # Encode to get posterior parameters
        _, mean, log_var = self.encoder(x)
        log_var = torch.clamp(log_var, min=-20, max=20)

        # Sample z ~ q(z|x)
        z = self.reparameterization(mean, log_var)

        # Decode to reconstruction
        x_hat = self.decoder(z)
        x_hat = adjust_shape(x_hat, (H, W))  # shape [B,2,H,W]

        # Monte Carlo estimate of E_q[log p(x|z)] under Gaussian
        sq_err   = ((x_hat - x) ** 2).mean(dim=(1,2,3))  # per-sample MSE
        log_px_z = - sq_err / (2 * sigma**2)           # (B,)

        # The KL term remains
        kld = self.compute_kld(mean, log_var, free_bits=0.0)          # (B,)

        return log_px_z, kld

    # y is the labels tensor
    def forward(self, x):
        return self.calculate_elbo(x)

    def loss_function(self, log_px_z, kld, beta=1.0):
        # elbos is a tensor of shape (B)
        return -(log_px_z - beta * kld).mean(0) # Loss is -elbo
    
class CVAE(VAE):
    def __init__(self, input_size, latent_dim, num_classes):
        super().__init__(input_size, latent_dim, conditional=True, num_classes=num_classes)
