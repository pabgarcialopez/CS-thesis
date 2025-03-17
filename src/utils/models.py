# ----------------------------------------------------
# Helper functions src/models.py
# ----------------------------------------------------

from math import floor

def compute_conv2D_output_size(input_size, kernel_size, stride, padding):
    """
        Returns the output (1D) size of a convolutional layer based on the formula:
        
        `output size = 1 + floor((input_size + 2 * padding - kernel_size) / stride)`
    """

    if stride == 0:
        raise ZeroDivisionError("Cannot compute output size with stride = 0")
    return 1 + floor((input_size + 2 * padding - kernel_size) / stride)
def compute_convTranspose2D_output_size(input_size, kernel_size, stride, padding, output_padding):
    """
        Returns the output (1D) size of a transposed convolutional layer based on the formula:
        
        `output size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding`
    """
    return (input_size - 1) * stride - 2 * padding + kernel_size + output_padding



def compute_flattened_size(num_channels, height, width):
    """
        Returns the flattened size of a tensor with shape `[num_channels, height, width]`
    """
    return num_channels * height * width