# ----------------------------------------------------
# Helper functions src/models.py
# ----------------------------------------------------

from math import floor
import torch

def compute_magnitude_and_phase(stft_spec):
    magnitude = stft_spec.abs()  
    phase = torch.angle(stft_spec) 
    return magnitude, phase

def compute_conv2D_output_size(input_size, kernel_size, stride, padding):
    """
        Returns the output (2D) size of a convolutional layer based on the formula:
        `output size = 1 + floor((input_size + 2 * padding - kernel_size) / stride)`
    """
    def _compute_conv2D_output_size(input_size):
        return 1 + floor((input_size + 2 * padding - kernel_size) / stride)
    
    if stride == 0: raise ZeroDivisionError("Cannot compute output size with stride = 0")
    output_height = _compute_conv2D_output_size(input_size[0])
    output_width = _compute_conv2D_output_size(input_size[1])
    return (output_height, output_width)

def compute_convTranspose2D_output_size(input_size, kernel_size, stride, padding):
    """
        Returns the output (2D) size of a transposed convolutional layer based on the formula:
        `output size = (input_size - 1) * stride - 2 * padding + kernel_size`
    """
    def _compute_convTransposed2D_output_size(input_size):
        return (input_size - 1) * stride - 2 * padding + kernel_size
    
    output_height = _compute_convTransposed2D_output_size(input_size[0])
    output_width = _compute_convTransposed2D_output_size(input_size[1])
    return (output_height, output_width)

def compute_output_padding(expected_output_size, computed_output_size, kernel_size, padding, stride):
    return (expected_output_size[0] - computed_output_size[0], expected_output_size[1] - computed_output_size[1])