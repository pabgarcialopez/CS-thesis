import torch.nn as nn

from src.utils.models import *

class Encoder(nn.Module):
    def __init__(self, input_size, latent_dim, channels):
        super().__init__()

        conv_kernel_size = (3, 3)
        conv_stride = (2, 2)  # Stride 2 for downsampling
        conv_padding = (1, 1)

        # Initialize sizes and compute them iteratively.
        # Here, input_size is assumed to be a tuple (freq, time).
        self.sizes = [input_size]
        current_size = input_size  # e.g., (512, 497)

        blocks = []
        for i in range(1, len(channels)):
            # Convolution layer with stride 2 (downsampling)
            conv = nn.Conv2d(channels[i - 1], channels[i],
                             kernel_size=conv_kernel_size,
                             stride=conv_stride,
                             padding=conv_padding)
            # Compute size after convolution (should reduce both dimensions)
            conv_size = compute_conv2D_output_size(current_size, conv_kernel_size, conv_stride, conv_padding)
            current_size = conv_size  # After convolution
            self.sizes.append(current_size)

            block = nn.Sequential(conv, nn.ReLU())
            blocks.append(block)
        self.encoder = nn.Sequential(*blocks)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(channels[-1] * current_size[0] * current_size[1], latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

    def get_sizes(self):
        return self.sizes


class Decoder(nn.Module):
    def __init__(self, sizes, latent_dim, channels):
        super().__init__()

        kernel_size = (3, 3)
        stride = (2, 2)
        padding = (1, 1)
        output_padding = 0

        # Reverse channels and sizes (each size is a tuple (freq, time))
        rev_channels = list(reversed(channels))
        rev_sizes = list(reversed(sizes))

        expected_size = rev_sizes[0]
        self.fc2 = nn.Linear(latent_dim, rev_channels[0] * expected_size[0] * expected_size[1])
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
            block = nn.Sequential(deconv, nn.ReLU())
            deconv_blocks.append(block)
        self.decoder = nn.Sequential(*deconv_blocks)

    def forward(self, x):
        x = self.fc2(x)
        x = self.unflatten(x)
        x = self.decoder(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, input_size, latent_dim):
        super().__init__()

        # Adjusted filter configuration
        channels = [2, 32, 64, 128]

        self.encoder = Encoder(input_size, latent_dim, channels)
        sizes = self.encoder.get_sizes()
        self.decoder = Decoder(sizes, latent_dim, channels)

        print("Encoder: ", self.encoder)
        print("Decoder: ", self.decoder)

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        target_height, target_width = x.shape[2], x.shape[3]
        reconstructed = adjust_shape(reconstructed, target_height, target_width)
        assert reconstructed.shape == x.shape, f"Expected {x.shape}, got {reconstructed.shape}"
        return reconstructed

