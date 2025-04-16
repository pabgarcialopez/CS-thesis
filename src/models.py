import torch.nn as nn
from torch.distributions import Normal
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
            # Convolution layer with stride 2 (downsampling)
            conv = nn.Conv2d(channels[i - 1], channels[i],
                             kernel_size=conv_kernel_size,
                             stride=conv_stride,
                             padding=conv_padding)
            # Compute size after convolution
            current_size = compute_conv2D_output_size(current_size, conv_kernel_size, conv_stride, conv_padding)
            self.sizes.append(current_size)

            block = nn.Sequential(conv, nn.ReLU())
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
            return mu, log_var
        
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

        self.input_size = input_size

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
        reconstructed = adjust_shape(reconstructed, (target_height, target_width))
        assert reconstructed.shape == x.shape, f"Expected {x.shape}, got {reconstructed.shape}"
        return reconstructed
    

# class VAE(nn.Module):
#     def __init__(self, input_size, latent_dim):
#         super().__init__()

#         self.input_size = input_size
#         self.latent_dim = latent_dim

#         self.normal = Normal(0, 1)
#         self.normal.loc = self.normal.loc.cuda()
#         self.normal.scale = self.normal.scale.cuda()

#         channels = [2, 16, 32, 64]

#         self.encoder = Encoder(input_size, latent_dim, channels, variational=True)
#         sizes = self.encoder.get_sizes()
#         self.decoder = Decoder(sizes, latent_dim, channels)

#         print("Encoder: ", self.encoder)
#         print("Decoder: ", self.decoder)

#     def reparameterization(self, mean, var):
#         eps = self.normal.sample(mean.shape)
#         return mean + var * eps

#     def forward(self, x):
#         mean, log_var = self.encoder(x)
#         x = self.reparameterization(mean, log_var)
#         x_hat = self.decoder(x)
#         x_hat = adjust_shape(x_hat, self.input_size)
#         return x_hat, mean, log_var

#     def loss_function(self, x, x_hat, mean, log_var, batch_size):
#         # Mean squared error loss
#         BCE = F.mse_loss(x, x_hat, reduction='mean')
#         # KL divergence between N(mu, var) and N(0, 1) is
#         KL = -0.5 * torch.sum(1 + log_var - torch.exp(log_var) - torch.pow(mean, 2)) / batch_size
#         return 0.6 * BCE + 0.4 * KL

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
        self.decoder = Decoder(sizes, latent_dim, self.channels)  

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
        eps = self.normal.sample(mean.shape)
        return mean + torch.exp(0.5 * log_var) * eps

    def forward(self, x, y=None):
        B, C, H, W = x.shape

        if self.conditional:
            y_enc = self.label_projector_encoder(y.float()).view(B, C, H, W)
            x = x + y_enc

        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, log_var)

        if self.conditional:
            z = z + self.label_projector_decoder(y.float())

        x_hat = self.decoder(z)
        x_hat = adjust_shape(x_hat, self.input_size)
        return x_hat, mean, log_var

    def loss_function(self, x, x_hat, mean, log_var, batch_size):
        BCE = F.mse_loss(x, x_hat, reduction='mean')
        KL = -0.5 * torch.sum(1 + log_var - torch.exp(log_var) - mean.pow(2)) / batch_size
        return 0.6 * BCE + 0.4 * KL
    
class CVAE(VAE):
    def __init__(self, input_size, latent_dim, num_classes):
        super().__init__(input_size, latent_dim, conditional=True, num_classes=num_classes)

