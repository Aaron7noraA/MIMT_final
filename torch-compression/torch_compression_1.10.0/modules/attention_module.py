import torch
from torch import nn
from torch.nn.modules.utils import _pair

from .activation import HardSigmoid


class SqueezeExcitationNet(nn.Sequential):

    def __init__(self, num_features, reduction):
        super(SqueezeExcitationNet, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features//reduction, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_features//reduction, num_features, 1),
            HardSigmoid(inplace=True)
        )

    def forward(self, input):
        return input * super().forward(input)


class ZPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.cat([input.max(1, keepdim=True)[0], input.mean(1, keepdim=True)], dim=1)


class TripleAttentionModule(nn.Module):
    """TripleAttention"""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size-1)//2
        self.channel_wise = nn.Sequential(
            ZPool(), nn.Conv2d(2, 1, kernel_size, padding=padding), nn.Sigmoid())
        self.height_wise = nn.Sequential(
            ZPool(), nn.Conv2d(2, 1, kernel_size, padding=padding), nn.Sigmoid())
        self.width_wise = nn.Sequential(
            ZPool(), nn.Conv2d(2, 1, kernel_size, padding=padding), nn.Sigmoid())

    def RAC(self, input, model, dim=1):
        data = input.transpose(1, dim) if dim > 1 else input
        mask = model(data)
        output = mask * data
        output = output.transpose(1, dim) if dim > 1 else output
        return output

    def forward(self, input):
        channel_wise = self.RAC(input, self.channel_wise)
        height_wise = self.RAC(input, self.height_wise, dim=2)
        width_wise = self.RAC(input, self.width_wise, dim=3)

        return (channel_wise + height_wise + width_wise) / 3


class NonLocalBlock(nn.Module):
    """NonLocalBlock"""

    def __init__(self, num_features, block_size=1, reduce_ratio=2):
        super().__init__()
        self.transform = nn.Conv2d(
            num_features, (num_features//reduce_ratio)*3, 1)
        self.latent = (num_features//reduce_ratio)
        self.rdk = (num_features//reduce_ratio) ** 0.5

        if block_size > 1:
            self.downsample = nn.AvgPool2d(block_size)
        else:
            self.downsample = None

        if reduce_ratio > 1:
            self.decode = nn.Conv2d(
                num_features//reduce_ratio, num_features, 1)
        else:
            self.decode = None

    def forward(self, input):
        QV, K = self.transform(input).split(self.latent*2, dim=1)
        if self.downsample is not None:
            QV = self.downsample(QV)

        Q, V = QV.chunk(2, dim=1)
        score = Q.flatten(2).transpose(1, 2) @ K.flatten(2)
        index = score.div(self.rdk).softmax(1)
        # print(V.flatten(2).shape, index.shape)
        output = (V.flatten(2) @ index).view_as(K)
        # print(output.shape)
        if self.decode is not None:
            output = self.decode(output)

        return output + input


class Involution2d(nn.Module):
    """
    This class implements the 2d involution proposed in:
    https://arxiv.org/pdf/2103.06255.pdf
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 reduce_ratio: int = 1,
                 sigma_mapping: nn.Module = None,
                 **kwargs) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param kernel_size: (Union[int, Tuple[int, int]]) Kernel size to be used
        :param stride: (Union[int, Tuple[int, int]]) Stride factor to be utilized
        :param groups: (int) Number of groups to be employed
        :param padding: (Union[int, Tuple[int, int]]) Padding to be used in unfold operation
        :param dilation: (Union[int, Tuple[int, int]]) Dilation in unfold to be employed
        :param reduce_ratio: (int) Reduce ration of involution channels
        :param sigma_mapping: (nn.Module) Non-linear mapping as introduced in the paper. If none BN + ReLU is utilized
        :param **kwargs: Unused additional key word arguments
        """
        super(Involution2d, self).__init__()
        assert isinstance(sigma_mapping, nn.Module) or sigma_mapping is None, \
            "Sigma mapping must be an nn.Module or None to utilize the default mapping (BN + ReLU)."
        assert isinstance(
            reduce_ratio, int) and reduce_ratio > 0, "reduce ratio must be a positive integer."

        # Save parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.reduce_ratio = reduce_ratio
        self.kernel_prod = self.kernel_size[0] * self.kernel_size[1]

        # Init modules
        self.input_mapping = nn.Sequential()
        if self.in_channels != self.out_channels:
            self.input_mapping.add_module(
                "transform", nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1),
                                       stride=(1, 1), padding=(0, 0), bias=False))
        self.input_mapping.add_module("unfold", nn.Unfold(
            kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride))

        if reduce_ratio > 1:
            self.kernel_mapping = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride),
                nn.Conv2d(in_channels, out_channels // reduce_ratio, kernel_size=(1, 1),
                          stride=(1, 1), padding=(0, 0), bias=False),
                sigma_mapping if sigma_mapping is not None else nn.Sequential(
                    nn.BatchNorm2d(out_channels // reduce_ratio), nn.ReLU()),
                nn.Conv2d(out_channels // reduce_ratio, self.kernel_prod * self.groups,
                          kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
            )
        elif reduce_ratio == 1:
            self.kernel_mapping = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride),
                nn.Conv2d(out_channels, self.kernel_prod * self.groups,
                          kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
            )

    def __repr__(self) -> str:
        """
        Method returns information about the module
        :return: (str) Info string
        """
        s = self.__class__.__name__ + \
            "({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.reduce_ratio > 1:
            s += ", reduce_ratio={reduce_ratio}"
            s += ", sigma_mapping={sigma_mapping}".format(
                sigma_mapping=str(self.kernel_mapping[2]))
        return (s+")").format(**self.__dict__)

    def _get_output_shape(self, shape):
        out_shape = (shape[0], self.groups, -1, self.kernel_prod)
        for i in range(len(self.kernel_size)):
            out_shape += (int((shape[2+i] + self.padding[i]*2 - self.dilation[i]
                               * (self.kernel_size[i] - 1) - 1)//self.stride[i] + 1),)
        return out_shape

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size, in channels, height, width]
        :return: (torch.Tensor) Output tensor of the shape [batch size, out channels, height, width] (w/ same padding)
        """
        # Check input dimension of input tensor
        assert input.dim() == 4, \
            "Input tensor to involution must be 4d but {}d tensor is given".format(
                input.dim())
        out_shape = self._get_output_shape(input.size())
        # Unfold and reshape input tensor
        input_unfolded = self.input_mapping(input).reshape(*out_shape)
        # print(input_unfolded.shape)
        # Generate kernel
        kernel = self.kernel_mapping(input).reshape(*out_shape)
        # print(kernel.shape)
        # Apply kernel to produce output "bgckhw,bgikhw->bgchw->bchw"
        output = input_unfolded.mul(kernel).sum(3).flatten(1, 2)
        return output
