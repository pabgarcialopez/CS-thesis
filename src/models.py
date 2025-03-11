import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import *
from utils.models import *

class AutoEncoder(nn.Module):
    def __init__(self, input_height, input_width, latent_dim, in_channels=1, filters=[32, 64, 128]):
        super().__init__()

        # ----------------
        # Encoder setup
        # ----------------

        current_height = input_height
        current_width = input_width

        # Define encoder's architecture: (convolution -> activation -> pooling) -> linearity        
        enc_layers = []
        input_channels = in_channels
        for output_channels in filters:
            enc_layers.append(nn.Conv2d(
                input_channels, 
                output_channels, 
                kernel_size=CONV_KERNEL_SIZE, 
                stride=CONV_STRIDE, 
                padding=CONV_PADDING))
            enc_layers.append(nn.ReLU())
         
            # Compute output size for convolution
            current_height = compute_output_size(current_height, CONV_KERNEL_SIZE, CONV_STRIDE, CONV_PADDING)
            current_width = compute_output_size(current_width, CONV_KERNEL_SIZE, CONV_STRIDE, CONV_PADDING)
            
            # Prepare input_channels of next layer
            input_channels = output_channels

        # Flatten the data to be able to apply the linear layer
        flattened_size = compute_flattened_size(filters[-1], current_height, current_width)
        enc_layers.append(nn.Flatten())
        enc_layers.append(nn.Linear(flattened_size, latent_dim))

        # Finally define encoder
        self.encoder = nn.Sequential(*enc_layers)

        # ----------------
        # Decoder setup
        # ----------------

        dec_layers = []
        
        # Convert flattened tensor back to a 4d tensor the decoder can work with
        # [batch_size, latent_dim] --> [batch_size, flattened_size]
        dec_layers.append(nn.Linear(latent_dim, flattened_size))
        # [batch_size, flattened_size] --> [batch_size, filters[-1], height, width]
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
                output_padding=1)) # Padding 1 to get back correct output dimensions
            dec_layers.append(nn.ReLU())
            input_channels = output_channels

        dec_layers.append(nn.ConvTranspose2d(
            in_channels=input_channels, # (= filters[0])
            out_channels=in_channels,     
            kernel_size=CONV_KERNEL_SIZE,
            stride=1, # No spatial change here
            padding=CONV_PADDING))
        dec_layers.append(nn.Sigmoid())

        # Finally define decoder
        self.decoder = nn.Sequential(*dec_layers)
        
    def forward(self, x):
        return self.decoder(self.encoder(x))

class VAE(AutoEncoder):
    def __init__(self, input_channels, latent_dim):
        super().__init__(input_channels, latent_dim)
        # Additional layers for VAE:
        self.fc_mu = nn.Linear(some_flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(some_flattened_size, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoder part: extract features, then get mu and logvar
        features = self.encoder(x)
        features_flat = features.view(features.size(0), -1)
        mu = self.fc_mu(features_flat)
        logvar = self.fc_logvar(features_flat)
        z = self.reparameterize(mu, logvar)
        # Use z as latent for decoder
        dec_input = self.fc_dec(z)
        dec_input = dec_input.view(...)  # reshape appropriately
        reconstruction = self.decoder(dec_input)
        return reconstruction, mu, logvar

class CVAE(VAE):
    def __init__(self, input_channels, latent_dim, condition_dim):
        super().__init__(input_channels, latent_dim)
        # Modify the architecture to accept conditioning information
        # For example, add an embedding layer for the condition if needed:
        self.condition_embed = nn.Linear(condition_dim, condition_dim)
        # Adjust the encoder and/or decoder to incorporate the condition.
        # This might be done by concatenating the condition vector with the input or latent vector.
        
    def forward(self, x, condition):
        # Process the condition (e.g., embedding or one-hot encoding)
        cond_emb = self.condition_embed(condition)
        # Option 1: Concatenate condition with input:
        # x = torch.cat([x, cond_emb], dim=?) 
        # Option 2: Concatenate condition with latent vector:
        features = self.encoder(x)
        features_flat = features.view(features.size(0), -1)
        mu = self.fc_mu(features_flat)
        logvar = self.fc_logvar(features_flat)
        z = self.reparameterize(mu, logvar)
        # Concatenate the latent vector with condition before decoding
        z_cond = torch.cat([z, cond_emb], dim=1)
        dec_input = self.fc_dec(z_cond)
        dec_input = dec_input.view(...)  # reshape appropriately
        reconstruction = self.decoder(dec_input)
        return reconstruction, mu, logvar
