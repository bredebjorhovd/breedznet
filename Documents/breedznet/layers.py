"""
The neural net is made up of layers.
Each layer needs to pass its input forward
and propagate gradients backwards, for example
a neural net might look like

inputs -> Linear -> Tahn -> Linear -> output
"""
from typing import Dict, Callable
import numpy as np
from breedznet.tensor import Tensor

class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Produce the outputs corresponding to these inputs
        """
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backpropagate this gradient through the layer
        """
        raise NotImplementedError

class Linear(Layer):
    """
    Computes output = inputs @ w + b
    """
    def __init__(self, input_size: int, output_size:int) -> None:
        # Inputs will be (batch_size, input_size)
        # Outputs will be (batch_size, input_size)
        super().__init__()
        self.params["w"] = np.random.rand(input_size, output_size)
        self.params["b"] = np.random.rand(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        outputs = inputs @ w + b
        """
        self.inputs = inputs
        return input @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        """
        If y = f(x) and x = a * b + c
        then dy/da = f'(x) * b
        and dy/db = f'(x) * a
        and dy/dc = f'(x)

        if y = f(x) and x = a @ b + c
        then dy/da = f'(x) @ b.T
        and dy/db = a.T @ f'(x)
        and dy/dc = f'(x)
        """
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T


F = Callable[[Tensor], Tensor]


class Activation(Layer):
    """
    Applies a function elementwise to ints inputs
    """
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        reutnr self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        """
        return self.f_prime(self.inputs) * grad

def tahn(x: Tensor) -> Tensor:
    return np.tahn(x)

def tahn_prime(x: Tensor) -> Tensor:
    y = tahn(x)
    return 1 - y ** 2

class Tahn(Activation):
    def __init__(self):
        super().__init__(tahn, tahn_prime)
        



