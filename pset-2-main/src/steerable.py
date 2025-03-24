import torch
from jaxtyping import Float, Complex
from torch import Tensor

from .gauss import _gaussian_filter_1d
import torch.nn.functional as F



def oriented_filter(theta: float, sigma: float, **kwargs) -> Float[Tensor, "N N"]:
    """
    Return an oriented first-order Gaussian filter
    given an angle (in radians) and standard deviation.

    Hint:
    - Use `.gauss._gaussian_filter_1d`!

    Implementation details:
    - **kwargs are passed to `_gaussian_filter_1d`
    """
    # raise NotImplementedError("Homework!")
    h0 = _gaussian_filter_1d(sigma, order=0, **kwargs)  # 1D Gaussian (smoothing)
    h1 = _gaussian_filter_1d(sigma, order=1, **kwargs)  # 1D first derivative
    
    # Form the separable filters:
    # Horizontal derivative: differentiate along x, smooth along y.
    G0 = h0[:, None] * h1[None, :]
    
    # Vertical derivative: differentiate along y, smooth along x.
    G90 = h1[:, None] * h0[None, :]
    
    # Steer the filter to the desired angle.
    oriented = torch.cos(torch.tensor(theta)) * G0 + torch.sin(torch.tensor(theta)) * G90
    return oriented



def conv(
    img: Float[Tensor, "B 1 H W"],  # Input image
    kernel: Float[Tensor, "N N"] | Complex[Tensor, "N N"],  # Convolutional kernel
    mode: str = "reflect",  # Padding mode
) -> Float[Tensor, "B 1 H W"]:
    """
    Convolve an image with a 2D kernel (assume N < H and N < W).
    """
    # raise NotImplementedError("Homework!")
    

    B, C, H, W = img.shape
    N = kernel.shape[0]  # assume kernel is square, so kernel.shape[1] == N

    # Compute padding: if N is odd, p = N//2 on each side works.
    # (For even N, more care is needed, but here we assume N is odd.)
    p = N // 2

    # Pad the image on the spatial dimensions (F.pad expects (pad_left, pad_right, pad_top, pad_bottom))
    img_padded = F.pad(img, (p, p, p, p), mode=mode)

    # Reshape kernel to (out_channels, in_channels, kernel_height, kernel_width)

    if torch.is_complex(kernel):
    # Explicitly convert the real and imaginary parts to the image's dtype.
        kernel_real = kernel.real.to(img.dtype).unsqueeze(0).unsqueeze(0)
        kernel_imag = kernel.imag.to(img.dtype).unsqueeze(0).unsqueeze(0)
        conv_real = F.conv2d(img_padded, kernel_real)
        conv_imag = F.conv2d(img_padded, kernel_imag)
        output = conv_real + 1j * conv_imag
    else:
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        output = F.conv2d(img_padded, kernel)

    return output




def steer_the_filter(
    img: Float[Tensor, "B 1 H W"], theta: float, sigma: float, **kwargs
) -> Float[Tensor, "B 1 H W"]:
    """
    Return the image convolved with a steered filter.
    """
    # raise NotImplementedError("Homework!")

    kernel = oriented_filter(theta, sigma, **kwargs)
    
    # Convolve the image with the kernel.
    return conv(img, kernel)


def steer_the_images(
    img: Float[Tensor, "B 1 H W"], theta: float, sigma: float, **kwargs
) -> Float[Tensor, "B 1 H W"]:
    """
    Return the steered image convolved with a filter.
    """
    # raise NotImplementedError("Homework!")

    h0 = _gaussian_filter_1d(sigma, order=0, **kwargs)  # 1D Gaussian (smoothing)
    h1 = _gaussian_filter_1d(sigma, order=1, **kwargs)  # 1D first derivative
    
    # Form the separable filters:
    # Horizontal derivative: differentiate along x, smooth along y.
    G0 = h0[:, None] * h1[None, :]
    
    # Vertical derivative: differentiate along y, smooth along x.
    G90 = h1[:, None] * h0[None, :]

    # apply the filter to the image 

    img0 = conv(img, G0)
    img90 = conv(img, G90)

    # steer the image 

    img_steered = torch.cos(torch.tensor(theta)) * img0 + torch.sin(torch.tensor(theta)) * img90

    return img_steered



def measure_orientation(
    img: Float[Tensor, "B 1 H W"], sigma: float, **kwargs
) -> Float[Tensor, "B 1 H W"]:
    """
    Design a filter to measure the orientation of edges in an image.

    Hint:
    - Consider the complex filter from the README
    - You will need to design a method for noise suppression
    """
    # raise NotImplementedError("Homework!")

    h0 = _gaussian_filter_1d(sigma, order=0, **kwargs)  # 1D Gaussian (smoothing)
    h1 = _gaussian_filter_1d(sigma, order=1, **kwargs)  # 1D first derivative   

    G0 = h0[:, None] * h1[None, :]
    G90 = h1[:, None] * h0[None, :]

    G_complex = G0 + 1j * G90

    img0 = conv(img, G_complex)

    # compute magnitude and phase

    mag = torch.abs(img0)
    phase = torch.angle(img0)

    # set some thrshold 

    phase = torch.where(mag > torch.max(mag) * 0.1, phase, torch.zeros_like(phase))

    return phase 





