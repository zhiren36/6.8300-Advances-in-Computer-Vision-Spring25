from typing import Tuple, TypeVar

import torch
from jaxtyping import Float
from torch import Tensor

N = TypeVar("N")


def shift_operator(img_shape: Tuple[int, int], shift_x: int, shift_y: int) -> Tensor:
    """
    Constructs a 2D shift operator for an image with circular boundaries.

    Args:
        img_shape: Tuple[int, int]
            The (height, width) dimensions of the image.
        shift_x: int
            The number of pixels to shift horizontally.
        shift_y: int
            The number of pixels to shift vertically.

    Returns:
        Tensor of shape (h*w, h*w)
            A matrix that, when applied to a flattened image, shifts it by the specified amounts.
    """

    # first create 1d shift matrix

     # Create 1D shift matrices for each axis
    V_shift = torch.eye(img_shape[1])
    H_shift = torch.eye(img_shape[0])
    
    # Correct the axes: vertical shift (rows) should use shift_y and horizontal shift (columns) should use shift_x
    V_shift = torch.roll(V_shift, shift_x, dims=1)
    H_shift = torch.roll(H_shift, shift_y, dims=1)
    
    # Combine to create the 2D shift matrix
    shift_matrix = torch.kron(H_shift, V_shift)
    
    return shift_matrix
    



def matrix_from_convolution_kernel(
    kernel: Float[Tensor, "*"], n: int
) -> Float[Tensor, "n n"]:
    """
    Constructs a circulant matrix of size n x n from a 1D convolution kernel with periodic alignment.

    Args:
        kernel: Tensor
            A 1D convolution kernel.
        n: int
            The desired size of the circulant matrix.

    Returns:
        Tensor of shape (n, n)
            The circulant matrix representing the convolution with periodic boundary conditions.
    """
    ## raise NotImplementedError("Homework!")

    L = kernel.numel()
    assert L % 2 == 1, "Kernel length must be odd for this construction."
    center = L // 2  # index of the central element

    # Create a generating vector v of length n, initialized to zeros.
    v = torch.zeros(n, dtype=kernel.dtype, device=kernel.device)
    # Place the center element at index 0
    v[0] = kernel[center]
    # Place the kernel elements that come after the center into the beginning of v
    # For indices 1,2,... up to (L-center-1)
    for i in range(1, L - center):
        if i < n:  # ensure we stay in bounds
            v[i] = kernel[center + i]
    # Place the kernel elements before the center at the end of v in reversed order
    # so that v[n-i] gets kernel[center-i] for i=1,...,center.
    for i in range(1, center + 1):
        if i < n:
            v[n - i] = kernel[center - i]


    rows = []
    for i in range(n):
        # Roll the kernel by i positions to create each row
        row = torch.roll(v, i)
        rows.append(row)
    
    circulant_matrix = torch.stack(rows, dim=0)
    return circulant_matrix



def image_operator_from_sep_kernels(
    img_shape: Tuple[int, int],
    kernel_x: Float[Tensor, "*"],
    kernel_y: Float[Tensor, "*"],
) -> Float[Tensor, "N N"]:
    """
    Constructs a 2D convolution operator for an image by combining separable 1D kernels.

    Args:
        img_shape: Tuple[int, int]
            The (height, width) dimensions of the image.
        kernel_x: Tensor
            The 1D convolution kernel to be applied horizontally.
        kernel_y: Tensor
            The 1D convolution kernel to be applied vertically.

    Returns:
        Tensor of shape (h*w, h*w)
            The 2D convolution operator acting on a flattened image.
    """
    ## raise NotImplementedError("Homework!")

    Matrix_Kernel_x = matrix_from_convolution_kernel(kernel_x, img_shape[1])
    Matrix_Kernel_y = matrix_from_convolution_kernel(kernel_y, img_shape[0])

    return torch.kron(Matrix_Kernel_x, Matrix_Kernel_y)







def eigendecomposition(
    operator: Float[Tensor, "N N"], descending: bool = True
) -> Tuple[Float[Tensor, "N"], Float[Tensor, "N N"]]:
    """
    Computes the eigenvalues and eigenvectors of a self-adjoint (Hermitian) linear operator.

    Args:
        operator: Tensor of shape (N, N)
            A self-adjoint linear operator.
        descending: bool
            If True, sort the eigenvalues and eigenvectors in descending order.

    Returns:
        A tuple (eigenvalues, eigenvectors) where:
            eigenvalues: Tensor of shape (N,)
            eigenvectors: Tensor of shape (N, N)
    """
    ## raise NotImplementedError("Homework!")


    eigenvalues, eigenvectors =  torch.linalg.eigh(operator)
    if descending:  # sort in descending order
        eigenvalues, indices = torch.sort(eigenvalues, descending=True)
        eigenvectors = eigenvectors[:, indices]
    return eigenvalues, eigenvectors 


def fourier_transform_operator(
    operator: Float[Tensor, "N N"], basis: Float[Tensor, "N N"]
) -> Float[Tensor, "N N"]:
    """
    Computes the representation of a linear operator in the Fourier (eigen) basis.

    Args:
        operator: Tensor of shape (N, N)
            The original linear operator in pixel space.
        basis: Tensor of shape (N, N)
            The Fourier eigenbasis.

    Returns:
        Tensor of shape (N, N)
            The operator represented in the Fourier basis.
    """
    ## raise NotImplementedError("Homework!")

    return torch.matmul(torch.matmul(basis.T, operator), basis)





def fourier_transform(
    img: Float[Tensor, "N"], basis: Float[Tensor, "N N"]
) -> Float[Tensor, "N"]:
    """
    Projects a flattened image onto the Fourier (eigen) basis.

    Args:
        img: Tensor of shape (N,)
            A flattened image.
        basis: Tensor of shape (N, N)
            The Fourier eigenbasis.

    Returns:
        Tensor of shape (N,)
            The image represented in the Fourier domain.
    """
    ## raise NotImplementedError("Homework!")

    return torch.matmul(basis.T, img)







def inv_fourier_transform(
    fourier_img: Float[Tensor, "N"], basis: Float[Tensor, "N N"]
) -> Float[Tensor, "N"]:
    """
    Reconstructs an image in pixel space from its Fourier coefficients using the provided eigenbasis.

    Args:
        fourier_img: Tensor of shape (N,)
            The image in the Fourier domain.
        basis: Tensor of shape (N, N)
            The Fourier eigenbasis used in the forward transform.

    Returns:
        Tensor of shape (N,)
            The reconstructed image in pixel space.
    """
    ## raise NotImplementedError("Homework!")

    return torch.linalg.inv(basis).T @ fourier_img
