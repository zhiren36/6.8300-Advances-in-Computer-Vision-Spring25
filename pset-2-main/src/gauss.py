from math import ceil

import torch
from jaxtyping import Float
from torch import Tensor
import math 

def _gaussian_filter_1d(
    sigma: float,  # Standard deviation of the Gaussian
    order: int,  # Order of the derivative
    truncate: float = 4.0,  # Truncate the filter at this many standard deviations
    dtype: torch.dtype = torch.float32,  # Data type to run the computation in
    device: torch.device = torch.device("cpu"),  # Device to run the computation on
) -> Float[Tensor, " filter_size"]:
    """
    Return a 1D Gaussian filter of a specified order.

    Implementation details:
    - filter_size = 2r + 1, where r = ceil(truncate * sigma)
    """
    ## raise NotImplementedError("Homework!")

    r = ceil(truncate * sigma)
    x = torch.arange(-r, r + 1, dtype=dtype, device=device)
    
    # Precompute the normalization factor as a tensor.
    norm_factor = torch.sqrt(torch.tensor(2 * math.pi, dtype=dtype, device=device)) * sigma

    if order == 0:
        # Standard Gaussian kernel
        kernel = torch.exp(-x**2 / (2 * sigma**2)) / norm_factor
        return kernel

    elif order == 1:
        dtype_high = torch.float64
        sigma_high = float(sigma)
        # Ensure a minimum support for stability.
        xs = torch.arange(-r, r + 1, dtype=dtype_high, device=device)
        
        # Normalization constant for the Gaussian.
        norm_const = torch.sqrt(torch.tensor(2 * math.pi, dtype=dtype_high, device=device)) * sigma_high

        # Define the continuous first derivative function.
        def continuous_first_deriv(t):
            # G^(1)(t) = -t/sigma^2 * G(t)
            return -t / (sigma_high**2) * torch.exp(-t**2 / (2 * sigma_high**2)) / norm_const

        # Simpson's rule integration over an interval [a, b].
        def simpson_integration(f, a, b, n=101):
            t = torch.linspace(a, b, n, dtype=dtype_high, device=device)
            y = f(t)
            h = (b - a) / (n - 1)
            return h / 3 * (y[0] + y[-1] + 4 * torch.sum(y[1:-1:2]) + 2 * torch.sum(y[2:-2:2]))

        # Compute the discrete kernel by integrating over each pixel interval [xi-0.5, xi+0.5].
        kernel = torch.empty_like(xs, dtype=dtype_high)
        for i, xi in enumerate(xs):
            a = xi - 0.5
            b = xi + 0.5
            kernel[i] = simpson_integration(continuous_first_deriv, a, b, n=101)

        # Enforce zero-sum (DC cancellation). Since the continuous derivative is odd,
        # any nonzero sum is due to discretization error.
        center_idx = xs.numel() // 2
        kernel[center_idx] -= kernel.sum()

        # Compute the discrete first moment: ideally, ∑_x x * h[x] should equal -1.
        moment1 = torch.sum(xs * kernel)
        if torch.abs(moment1) > 1e-12:
            kernel = kernel * ((-1.0) / moment1)
        else:
            raise ValueError("Discrete moment is too close to zero")

        # Cast back to the requested dtype.
        return kernel.to(dtype)

    elif order == 2:
        dtype_high = torch.float64
        sigma_high = float(sigma)
        norm_const = torch.sqrt(torch.tensor(2 * math.pi, dtype=dtype_high, device=device)) * sigma_high

        def continuous_second_deriv(t):
            # t can be a tensor.
            # Note: For t=0, (t^2 - sigma^2) is negative.
            return ((t**2 - sigma_high**2) / (sigma_high**4)) * torch.exp(-t**2 / (2 * sigma_high**2)) / norm_const

        # Simpson's rule integration over an interval [a, b] with n points (n must be odd)
        def simpson_integration(f, a, b, n=101):
            t = torch.linspace(a, b, n, dtype=dtype_high, device=device)
            y = f(t)
            h = (b - a) / (n - 1)
            # Simpson's rule: (h/3)[f(t0) + f(tn) + 4 * sum(f(odd)) + 2 * sum(f(even))]
            return h / 3 * (y[0] + y[-1] + 4 * torch.sum(y[1:-1:2]) + 2 * torch.sum(y[2:-2:2]))
        
        # Use a fine grid integration for each pixel interval.
        xs = torch.arange(-r, r + 1, dtype=dtype_high, device=device)
        kernel = torch.empty_like(xs, dtype=dtype_high)
        for i, xi in enumerate(xs):
            a = xi - 0.5
            b = xi + 0.5
            kernel[i] = simpson_integration(continuous_second_deriv, a, b, n=101)
        
        # At this point, kernel approximates the integral of the continuous second derivative over each pixel.
        # In theory, the continuous second derivative integrates to zero, so we enforce zero sum by adjusting the center.
        center_idx = xs.numel() // 2
        kernel[center_idx] -= kernel.sum()
        
        # Next, compute the discrete second moment:
        moment2 = torch.sum(xs**2 * kernel)
        if torch.abs(moment2) > 1e-12:
            kernel = kernel * (2.0 / moment2)  # Scale so that the moment equals 2.
        else:
            raise ValueError("Discrete moment is too close to zero")
        
        return kernel.to(dtype)

    else:
        # Use Hermite polynomials for higher orders
        const = torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device)) * sigma
        kernel = (
            (-1)**order *
            torch.exp(-x**2 / (2 * sigma**2)) *
            torch.pow(1 / const, order) *
            torch.exp(-x**2 / (2 * sigma**2)) *
            torch.special.hermite_polynomial_h(x, order)
        )
        return kernel






