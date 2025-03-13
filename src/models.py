import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import *
from src.utils.models import compute_output_size, compute_flattened_size

class Encoder(nn.Module):
    def __init__(self, input_height, input_width, latent_dim, in_channels=1, filters=[32, 64, 128]):
        super().__init__()

        current_height = input_height
        current_width = input_width

        layers = []
        c_in = in_channels
        for c_out in filters:
            layers.append(nn.Conv2d(
                in_channels=c_in, 
                out_channels=c_out, 
                kernel_size=CONV_KERNEL_SIZE, 
                stride=CONV_STRIDE, 
                padding=CONV_PADDING))
            layers.append(nn.ReLU())
            
            # Compute output size
            current_height = compute_output_size(current_height, CONV_KERNEL_SIZE, CONV_STRIDE, CONV_PADDING)
            current_width = compute_output_size(current_width, CONV_KERNEL_SIZE, CONV_STRIDE, CONV_PADDING)
            
            c_in = c_out
        
        # Flatten + linear to latent_dim
        flattened_size = compute_flattened_size(filters[-1], current_height, current_width)
        layers.append(nn.Flatten())
        layers.append(nn.Linear(flattened_size, latent_dim))

        self.encoder = nn.Sequential(*layers)

        # Store final shape so the decoder knows how to unflatten
        self.output_channels = filters[-1]
        self.output_height = current_height
        self.output_width = current_width

    def forward(self, x):
        return self.encoder(x)

    def get_output_shape(self):
        """
        Returns (channels, height, width) that the encoder produces
        before flattening.
        """
        return (self.output_channels, self.output_height, self.output_width)

class Decoder(nn.Module):
    def __init__(self, latent_dim, in_channels, filters, out_shape):
        """
        out_shape: (channels, height, width) from the encoder
        """
        super().__init__()
        c_out, h_out, w_out = out_shape  # e.g. (128, 8, 16)

        dec_layers = []
        
        # Convert latent vector back to (c_out, h_out, w_out)
        flattened_size = c_out * h_out * w_out
        dec_layers.append(nn.Linear(latent_dim, flattened_size))
        dec_layers.append(nn.Unflatten(dim=1, unflattened_size=(c_out, h_out, w_out)))

        # Reverse the convolution blocks
        c_in = c_out
        for c_out in reversed(filters):
            dec_layers.append(nn.ConvTranspose2d(
                in_channels=c_in, 
                out_channels=c_out, 
                kernel_size=CONV_KERNEL_SIZE, 
                stride=CONV_STRIDE, 
                padding=CONV_PADDING,
                output_padding=1))  # doubles the spatial dims
            dec_layers.append(nn.ReLU())
            c_in = c_out

        # Final layer to get back to in_channels
        dec_layers.append(nn.ConvTranspose2d(
            in_channels=c_in,
            out_channels=in_channels,
            kernel_size=CONV_KERNEL_SIZE,
            stride=1,
            padding=CONV_PADDING
        ))
        dec_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.decoder(x)

class AutoEncoder(nn.Module):
    def __init__(self, input_height, input_width, latent_dim, in_channels=1, filters=[32, 64, 128]):
        super().__init__()

        # 1) Build the encoder
        self.encoder = Encoder(input_height, input_width, latent_dim, in_channels, filters)

        # 2) Get the final shape from the encoder
        out_shape = self.encoder.get_output_shape()  # e.g. (128, 8, 16)

        # 3) Build the decoder with that shape
        self.decoder = Decoder(latent_dim, in_channels, filters, out_shape)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# VAE class inherits from AutoEncoder and overrides necessary parts
class VAE(AutoEncoder):
    def __init__(self, input_height, input_width, latent_dim, in_channels=1, filters=[32, 64, 128]):
        super(VAE, self).__init__(input_height, input_width, latent_dim, in_channels, filters)

        # Modify the encoder to output both mean and log variance
        self.fc_mu = nn.Linear(self.encoder.encoder[-1].in_features, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder.encoder[-1].in_features, latent_dim)

    def encode(self, x):
        latent_rep = self.encoder(x)
        mu = self.fc_mu(latent_rep)
        logvar = self.fc_logvar(latent_rep)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        # Reconstruction loss: MSE
        BCE = F.mse_loss(recon_x, x, reduction='sum')

        # KL divergence: between learned latent distribution and standard normal
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss is reconstruction loss + KL divergence
        return BCE + KLD
