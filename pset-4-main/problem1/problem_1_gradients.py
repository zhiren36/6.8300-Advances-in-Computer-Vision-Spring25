"""
For each function below
- y is the output of the model
- x is the input of the model

Use automatic differentiation to compute all functions of the gradient of y with respect to x.

Hint: implement the functions in order from top to bottom.
"""

import torch


def gradient(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor `y` that represents a function of `x` (`y` could be multi-dimensional),
    return the gradient of `y` with respect to `x`. It is important to note for the assignment
    that the PURPOSE of this function is to be used not only to compute gradients, but
    also to compute them in such a way that we can use them as part of the computational
    graph for backpropagation THROUGH the gradient function. If this is unclear, you
    may note that if you were to run the forward pass of a neural network, then
    compute the output of the backwards pass up to an input `x`, the concatenated
    [forwards pass, backwards pass] is a function of `x` and so defines a longer, 
    concatenated "forward pass" of `x`. We want to use this to train a SIREN on the gradients
    of an image, such that the SIREN's actual output corresponds to the image.
    
    Hint: You may find the `torch.autograd.grad` function useful. To achieve the stated purpose
    above, you will need to set a specific boolean parameter to one of two options. Can you
    read the documentation and figure out which one and why?
    """
    # raise NotImplementedError("Not implemented!")

    grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad






def divergence(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor `y` that represents a function of `x` (`y` could be multi-dimensional),
    return the divergence of `y` with respect to `x`. By convention we will compute
    the divergence along the LAST axis of `y`. When we use this function it will usually
    be flattened in practice.

    Hint: You may find the ` torch.autograd.grad` function useful. Like in `gradient`,
    you need to set a specific boolean parameter to the correct out of two options.
    """
    # raise NotImplementedError("Not implemented!")

    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div





def laplace(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor `y` that represents a function of `x` (`y` could be multi-dimensional),
    return the laplacian of `y` with respect to `x`.
    
    Hint: You may find some of our previous functions useful and the identity in the
    `Gradient` section of Wikipedia: https://en.wikipedia.org/wiki/Laplace_operator#Generalization.
    """
    # raise NotImplementedError("Not implemented!")

    grad = gradient(y, x)
    return divergence(grad, x)
