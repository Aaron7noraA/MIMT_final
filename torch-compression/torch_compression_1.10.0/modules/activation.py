import torch
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.nn.modules import Module
from torch.nn.parameter import Parameter


class QReLUFunc(Function):

    @staticmethod
    def forward(ctx, input, inplace: bool = False):
        mask = input.lt(0)
        ctx.save_for_backward(mask)

        if inplace:
            input[mask] *= (0.01-2)
            return input
        else:
            return input.masked_scatter(mask, input.masked_select(mask).mul(0.01-2))

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors

        return grad_output.masked_scatter(mask, grad_output.masked_select(mask).mul(0.01-2)), None


def q_relu(input, inplace: bool = False):
    return QReLUFunc.apply(input, inplace)


class QReLU(Module):

    def __init__(self, inplace: bool = False):
        super(QReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.leaky_relu(input, 0.01-2, inplace=self.inplace)


class MQReLUFunc(Function):

    @staticmethod
    def forward(ctx, input, inplace: bool = False):
        mask = input.lt(0)
        ctx.save_for_backward(mask)

        if inplace:
            input[mask] *= (0.01-1)
            return input
        else:
            return input.masked_scatter(mask, input.masked_select(mask).mul(0.01-1))

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        # mask = grad_output.lt(0)

        return grad_output.masked_scatter(mask, grad_output.masked_select(mask).mul(0.01-1)), None


def mq_relu(input, inplace: bool = False):
    return MQReLUFunc.apply(input, inplace)


class MQReLU(Module):

    def __init__(self, inplace: bool = False):
        super(MQReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.leaky_relu(input, 0.01-1, inplace=self.inplace)


def hard_sigmoid(input, inplace: bool = False):
    return F.relu6(input + 3, inplace=inplace) / 6


class HardSigmoid(Module):
    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return hard_sigmoid(input, inplace=self.inplace)


def swish(input, sigma=None):
    return input * (input.mul(sigma.view(-1, *((1,)*(input.dim()-2)))).sigmoid() if sigma is not None else input.sigmoid())


class Swish(Module):
    r"""Applies the Swish function, element-wise, as described in the paper:

    `Searching for MobileNetV3`_.

    .. math::
        \text{Swish}(x) = x * \sigma(\beta * x)

    Args:
        num_parameters (int): number of :math:`a` to learn.
            Although it takes an int as input, there is only two values are legitimate:
            1, or the number of channels at input. Default: 1
        init (float): the initial value of :math:`a`. Default: 0.25

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Attributes:
        weight (Tensor): the learnable weights of shape (:attr:`num_parameters`).

    Examples::

        >>> m = nn.Swish()
        >>> input = torch.randn(2)
        >>> output = m(input)

    .. _`Searching for MobileNetV3`:
        https://arxiv.org/abs/1905.02244
    """
    def __init__(self, num_parameters=0, init=1):
        super(Swish, self).__init__()
        self.num_parameters = num_parameters
        if num_parameters:
            self.weight = Parameter(torch.Tensor(num_parameters).fill_(init))
        else:
            self.weight = None

    def forward(self, input):
        return swish(input, self.weight)

    def extra_repr(self):
        if self.num_parameters:
            return 'num_parameters={}'.format(self.num_parameters)
        else:
            return ""
