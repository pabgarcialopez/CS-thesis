import torch.nn as nn
import torch.nn.functional as F

from src.utils.models import compute_conv2D_output_size, compute_convTranspose2D_output_size, compute_flattened_size

class Encoder(nn.Module):
    def __init__(self, input_height, input_width, latent_dim, in_channels=1, filters=[32, 64, 128], use_pooling=False, conv_kernel_size=3, conv_stride=2, conv_padding=1, pool_kernel_size=2, pool_stride=2):
        super().__init__()

        self.use_pooling = use_pooling
        self.output_shapes = []
        current_height = input_height
        current_width = input_width

        self.output_shapes.append((current_height, current_width))

        layers = []
        c_in = in_channels

        for c_out in filters:

            layers.append(nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding))
            layers.append(nn.ReLU())

            # Compute output size after conv
            current_height = compute_conv2D_output_size(current_height, conv_kernel_size, conv_stride, conv_padding)
            current_width = compute_conv2D_output_size(current_width, conv_kernel_size, conv_stride, conv_padding)

            if use_pooling:
                layers.append(nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride))
                current_height = compute_conv2D_output_size(current_height, pool_kernel_size, pool_stride, 0)
                current_width = compute_conv2D_output_size(current_width, pool_kernel_size, pool_stride, 0)

            self.output_shapes.append((current_height, current_width))

            print(self.output_shapes)

            c_in = c_out

        flattened_size = compute_flattened_size(filters[-1], current_height, current_width)
        layers.append(nn.Flatten())
        layers.append(nn.Linear(flattened_size, latent_dim))
        
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

    def get_output_shapes(self):
        return self.output_shapes


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        in_channels,  
        filters,
        output_shapes,   
        use_pooling=False,
        conv_kernel_size=3,
        conv_stride=2,
        conv_padding=1,
        pool_kernel_size=2,
        pool_stride=2
    ):
        super().__init__()
        
        c_out = filters[-1]
        h_out, w_out = output_shapes.pop()
        dec_layers = []

        # 1) Linear -> Unflatten
        flattened_size = c_out * h_out * w_out
        dec_layers.append(nn.Linear(latent_dim, flattened_size))
        dec_layers.append(nn.Unflatten(dim=1, unflattened_size=(c_out, h_out, w_out)))

        # 2) Reverse the filters
        c_in = c_out
        current_height = h_out
        current_width = w_out
        for c_rev in reversed(filters):
            if use_pooling:
                dec_layers.append(nn.Upsample(scale_factor=pool_stride, mode='nearest'))
                dec_layers.append(nn.ConvTranspose2d(in_channels=c_in, out_channels=c_rev, kernel_size=conv_kernel_size, stride=1, padding=conv_padding))
            else:
                input_size = (current_height, current_width)
                output_padding = self.compute_output_padding(input_size, conv_kernel_size, conv_stride, conv_padding, output_shapes.pop())
                dec_layers.append(nn.ConvTranspose2d(
                    in_channels=c_in,
                    out_channels=c_rev,
                    kernel_size=conv_kernel_size,
                    stride=conv_stride,
                    padding=conv_padding,
                    output_padding=output_padding,
                ))

                current_height = compute_convTranspose2D_output_size(current_height, conv_kernel_size, conv_stride, conv_padding, output_padding[0])
                current_width = compute_convTranspose2D_output_size(current_width, conv_kernel_size, conv_stride, conv_padding, output_padding[1])

            dec_layers.append(nn.ReLU())
            c_in = c_rev

        # 3) Final layer to get back to in_channels
        dec_layers.append(nn.ConvTranspose2d(
            in_channels=c_in, 
            out_channels=in_channels, 
            kernel_size=conv_kernel_size, 
            stride=1, 
            padding=conv_padding
        ))

        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.decoder(x)
    
    def compute_output_padding(self, input_size, kernel_size, stride, padding, expected_output_size):

        expected_height_output_size, expected_width_output_size = expected_output_size
        computed_height_output_size = (input_size[0] - 1) * stride - 2 * padding + kernel_size
        computed_width_output_size = (input_size[1] - 1) * stride - 2 * padding + kernel_size

        if computed_height_output_size <= expected_height_output_size and computed_width_output_size <= expected_width_output_size:
            return (expected_height_output_size - computed_height_output_size, expected_width_output_size - computed_width_output_size)
        
        raise Exception(f"expected_height_output_size = {expected_height_output_size} > {computed_height_output_size} = computed_height_output_size"
                        f"orexpected_width_output_size = {expected_width_output_size} > {computed_width_output_size} = computed_width_output_size")

class AutoEncoder(nn.Module):
    def __init__(self, input_height, input_width, latent_dim, in_channels=1, filters=[32, 64, 128], use_pooling=False, conv_kernel_size=3, conv_stride=2, conv_padding=1, pool_kernel_size=2, pool_stride=2):
        super().__init__()

        # Build the encoder
        self.encoder = Encoder(input_height, input_width, latent_dim, in_channels, filters, use_pooling, conv_kernel_size, conv_stride, conv_padding, pool_kernel_size, pool_stride)

        # Get final shape from encoder
        output_shapes = self.encoder.get_output_shapes()

        # Build the decoder
        self.decoder = Decoder(latent_dim, in_channels, filters, output_shapes, use_pooling, conv_kernel_size, conv_stride, conv_padding, pool_kernel_size, pool_stride)

        print("Encoder: ", self.encoder)
        print("Decoder: ", self.decoder)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

