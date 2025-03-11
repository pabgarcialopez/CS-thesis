import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import *
from src.utils.models import *

class Encoder(nn.Module):
    def __init__(self, input_height, input_width, latent_dim, in_channels=1, filters=[32, 64, 128]):
        super(Encoder, self).__init__()

        current_height = input_height
        current_width = input_width

        # Define the encoder architecture
        layers = []
        input_channels = in_channels
        for output_channels in filters:
            layers.append(nn.Conv2d(
                input_channels, 
                output_channels, 
                kernel_size=CONV_KERNEL_SIZE, 
                stride=CONV_STRIDE, 
                padding=CONV_PADDING))
            layers.append(nn.ReLU())
            
            # Compute output size for convolution
            current_height = compute_output_size(current_height, CONV_KERNEL_SIZE, CONV_STRIDE, CONV_PADDING)
            current_width = compute_output_size(current_width, CONV_KERNEL_SIZE, CONV_STRIDE, CONV_PADDING)
            
            input_channels = output_channels
        
        # After the convolutions, flatten the tensor to pass through the fully connected layer
        flattened_size = compute_flattened_size(filters[-1], current_height, current_width)
        layers.append(nn.Flatten())
        layers.append(nn.Linear(flattened_size, latent_dim))

        # Define the encoder as a sequential container
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, input_height, input_width, latent_dim, in_channels=1, filters=[32, 64, 128]):
        super(Decoder, self).__init__()

        current_height = input_height
        current_width = input_width

        dec_layers = []
        
        # Convert flattened tensor back to 4d tensor the decoder can work with
        dec_layers.append(nn.Linear(latent_dim, current_height * current_width * filters[-1]))
        dec_layers.append(nn.Unflatten(dim=1, unflattened_size=(filters[-1], current_height, current_width)))

        # Apply "backward" convolutions
        input_channels = filters[-1]
        for output_channels in reversed(filters):
            dec_layers.append(nn.ConvTranspose2d(
                input_channels, 
                output_channels, 
                kernel_size=CONV_KERNEL_SIZE, 
                stride=CONV_STRIDE, 
                padding=CONV_PADDING,
                output_padding=1))  # Padding 1 to get back correct output dimensions
            dec_layers.append(nn.ReLU())
            input_channels = output_channels

        dec_layers.append(nn.ConvTranspose2d(
            in_channels=input_channels,  # (= filters[0])
            out_channels=in_channels,     
            kernel_size=CONV_KERNEL_SIZE,
            stride=1,  # No spatial change here
            padding=CONV_PADDING))
        dec_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.decoder(x)

class AutoEncoder(nn.Module):
    def __init__(self, input_height, input_width, latent_dim, in_channels=1, filters=[32, 64, 128]):
        super(AutoEncoder, self).__init__()

        # Initialize the encoder and decoder separately
        self.encoder = Encoder(input_height, input_width, latent_dim, in_channels, filters)
        self.decoder = Decoder(input_height, input_width, latent_dim, in_channels, filters)

    def forward(self, x):
        return self.decoder(self.encoder(x))

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
