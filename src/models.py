import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.models import *

class Encoder(nn.Module):
    def __init__(self, input_size, latent_dim, channels):
        super().__init__()

        conv_kernel_size = 3
        conv_stride = 1
        conv_padding = 1

        pool_kernel_size = 2
        pool_stride = 2
        pool_padding = 0

        # Initialize sizes and compute them iteratively
        self.sizes = [input_size]
        current_size = input_size

        blocks = []
        for i in range(1, len(channels)):
            conv = nn.Conv2d(channels[i - 1], channels[i], kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)
            current_size = compute_conv2D_output_size(current_size, conv_kernel_size, conv_stride, conv_padding)
            pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
            current_size = compute_conv2D_output_size(current_size, pool_kernel_size, pool_stride,  pool_padding)
            self.sizes.append(current_size)

            block = nn.Sequential(conv, nn.ReLU(), pool)
            blocks.append(block)
        self.encoder_blocks = nn.Sequential(*blocks)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(channels[-1] * current_size[0] * current_size[1], latent_dim)

    def forward(self, x):
        # print("Initial x.shape = ", x.shape)
        x = self.encoder_blocks(x)
        # print("x.shape after conv and pool blocks = ", x.shape)
        x = self.flatten(x)
        # print("Flattened x.shape = ", x.shape)
        x = self.fc1(x)
        # print("Linearized x.shape = ", x.shape)
        return x

    def get_sizes(self):
        # print("Sizes: ", self.sizes)
        return self.sizes


class Decoder(nn.Module):
    def __init__(self, sizes, latent_dim, channels):
        super().__init__()

        kernel_size = 3
        stride = 2
        padding = 1

        # Reverse channels and sizes without modifying the originals
        rev_channels = list(reversed(channels))
        rev_sizes = list(reversed(sizes))

        # Linear layer to expand the latent vector
        expected_size = rev_sizes[0]
        current_size = rev_sizes[0]
        self.fc2 = nn.Linear(latent_dim, rev_channels[0] * expected_size[0] * expected_size[1])
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(rev_channels[0], expected_size[0], expected_size[1]))

        deconv_blocks = []
        # Build deconvolutional layers iteratively
        for i in range(1, len(rev_sizes)):
            expected_size = rev_sizes[i]
            current_size = compute_convTranspose2D_output_size(current_size, kernel_size, stride, padding)
            output_padding = compute_output_padding(expected_size, current_size, kernel_size, padding, stride)
            # print("\nExpected size: ", expected_size)
            # print("Current size: ", current_size)
            # print("Output padding size: ", output_padding)
            current_size = tuple(map(lambda a, b: a + b, current_size, output_padding))
            # print("Final current size: ", current_size)
            deconv = nn.ConvTranspose2d(rev_channels[i - 1], rev_channels[i], kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
            block = nn.Sequential(deconv, nn.ReLU())
            deconv_blocks.append(block)
        self.deconv_blocks = nn.Sequential(*deconv_blocks)

    def forward(self, x):
        x = self.fc2(x)
        x = self.unflatten(x)
        x = self.deconv_blocks(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, input_size, latent_dim):
        super().__init__()

        channels = [2, 16, 32, 64]

        self.encoder = Encoder(input_size, latent_dim, channels)
        sizes = self.encoder.get_sizes()
        self.decoder = Decoder(sizes, latent_dim, channels)

        print("Encoder: ", self.encoder)
        print("Decoder: ", self.decoder)

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)

        assert reconstructed.shape == x.shape, f"Expected output shape {x.shape}, but got {reconstructed.shape}"

        return reconstructed
