# ----------------------------------------------------
# Helper functions src/models.py
# ----------------------------------------------------

from math import floor
import torch
import torch.functional as F

def compute_magnitude_and_phase(stft_spec):
    magnitude = stft_spec.abs()  
    phase = torch.angle(stft_spec) 
    return magnitude, phase

def compute_conv2D_output_size(input_size, kernel_size, stride, padding):
    """
        Returns the output (2D) size of a convolutional layer based on the formula:
        `output size = 1 + floor((input_size + 2 * padding - kernel_size) / stride)`
    """
    def _compute_conv2D_output_size(input_size, kernel_size, stride, padding):
        return 1 + floor((input_size + 2 * padding - kernel_size) / stride)
    
    if stride == 0: raise ZeroDivisionError("Cannot compute output size with stride = 0")
    output_height = _compute_conv2D_output_size(input_size[0], kernel_size[0], stride[0], padding[0])
    output_width = _compute_conv2D_output_size(input_size[1], kernel_size[1], stride[1], padding[1])
    return (output_height, output_width)

def compute_convTranspose2D_output_size(input_size, kernel_size, stride, padding):
    """
        Returns the output (2D) size of a transposed convolutional layer based on the formula:
        `output size = (input_size - 1) * stride - 2 * padding + kernel_size`
    """
    def _compute_convTransposed2D_output_size(input_size, kernel_size, stride, padding):
        return (input_size - 1) * stride - 2 * padding + kernel_size
    
    output_height = _compute_convTransposed2D_output_size(input_size[0], kernel_size[0], stride[0], padding[0])
    output_width = _compute_convTransposed2D_output_size(input_size[1], kernel_size[1], stride[1], padding[1])
    return (output_height, output_width)

def compute_output_padding(expected_output_size, computed_output_size):
    return (expected_output_size[0] - computed_output_size[0], expected_output_size[1] - computed_output_size[1])

def adjust_shape(x, target_height, target_width):
    # Crop or pad height
    H, W = x.shape[2], x.shape[3]
    if H > target_height:
        x = x[:, :, :target_height, :]
    elif H < target_height:
        pad_h = target_height - H
        x = F.pad(x, (0, 0, 0, pad_h))
    
    # Crop or pad width
    if W > target_width:
        x = x[:, :, :, :target_width]
    elif W < target_width:
        pad_w = target_width - W
        x = F.pad(x, (0, pad_w, 0, 0))
    
    return x