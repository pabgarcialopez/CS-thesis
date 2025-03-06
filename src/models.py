import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super().__init__()
        # Define the encoder with Conv2d layers and linear layers
        self.encoder = nn.Sequential(
            # Example:
            # nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(),
            # ... more conv layers ...
        )
        # After conv layers, flatten and reduce to latent_dim
        self.fc_enc = nn.Linear(some_flattened_size, latent_dim)
        
        # Define the decoder with a linear layer and ConvTranspose2d layers
        self.fc_dec = nn.Linear(latent_dim, some_flattened_size)
        self.decoder = nn.Sequential(
            # Example:
            # nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.Sigmoid(),
        )
        
    def forward(self, x):
        # Encode and decode x
        features = self.encoder(x)
        features_flat = features.view(features.size(0), -1)
        latent = self.fc_enc(features_flat)
        # Decoder: reshape after linear layer
        dec_input = self.fc_dec(latent)
        # Reshape dec_input back into feature maps
        dec_input = dec_input.view(...)  # fill in appropriate shape
        reconstruction = self.decoder(dec_input)
        return reconstruction

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
