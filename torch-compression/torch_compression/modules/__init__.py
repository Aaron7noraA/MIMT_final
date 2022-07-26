from .activation import (HardSigmoid, MQReLU, QReLU, Swish, hard_sigmoid,
                         mq_relu, q_relu, swish)
from .attention_module import (Involution2d, NonLocalBlock,
                               SqueezeExcitationNet, TripleAttentionModule)
from .conditional_module import ConditionalLayer
from .context_model import ContextModel, MaskedConv2d
from .entropy_models import *
from .generalizedivisivenorm import (GeneralizedDivisiveNorm,
                                     generalized_divisive_norm)
from .signalconv import SignalConv2d, SignalConvTranspose2d

__CONV_TYPES__ = {'Signal': SignalConv2d,
                  'Standard': nn.Conv2d}
__conv_type__ = __CONV_TYPES__['Signal']
__ICNR__ = False


class SubPixelConv2d(__conv_type__):
    """SubPixelConv2d"""

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, ICNR=__ICNR__, *args, **kwargs):
        if __conv_type__ != SignalConv2d:
            if 'parameterizer' in kwargs:
                kwargs.pop('parameterizer')
            else:
                kwargs['parameterizer'] = None
        if 'padding' in kwargs:
            kwargs.pop('padding')
        if 'output_padding' in kwargs:
            kwargs.pop('output_padding')
        self.upscale_factor = stride
        self.ICNR = ICNR and (self.upscale_factor > 1)
        super(SubPixelConv2d, self).__init__(in_channels, out_channels*(stride**2), kernel_size, stride=1,
                                             padding=(kernel_size - 1) // 2, *args, **kwargs)

    def reset_parameters(self):
        super().reset_parameters()
        if self.ICNR:
            C = self.out_channels // self.upscale_factor ** 2

            def tile(input, dim: int, n_tile: int):
                expanse = list(input.size())
                expanse.insert(dim, n_tile)
                return input.unsqueeze(dim).expand(expanse).transpose(dim, dim+1).flatten(dim, dim+1)

            @torch.no_grad()
            def ICNR_(tensor):
                tensor.data.copy_(tile(tensor[:C], 0, self.upscale_factor**2))

            ICNR_(self.weight)
            if self.bias is not None:
                ICNR_(self.bias.unsqueeze(1))

    def extra_repr(self):
        return super().extra_repr() + ", upscale_factor={upscale_factor}, ICNR={ICNR}".format(**self.__dict__)

    def forward(self, input):
        return F.pixel_shuffle(super().forward(input), self.upscale_factor)


class RSubPixelConv2d(SubPixelConv2d):
    """SubPixelConv2d with ICNR

        args as SubPixelConv2d  
        ICNR (bool, optional): Nearest Neighbor init (ICNR) in https://arxiv.org/pdf/1707.02937.pdf. Default: True
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, ICNR=True, *args, **kwargs):
        super().__init__(in_channels, out_channels,
                         kernel_size, stride, ICNR, *args, **kwargs)


__DECONV_TYPES__ = {'Signal': SignalConvTranspose2d,
                    'Transpose': nn.ConvTranspose2d,
                    'SubPixel': SubPixelConv2d,
                    'RSubPixel': RSubPixelConv2d}
__deconv_type__ = __DECONV_TYPES__['Signal']


def set_default_conv(conv_type=SignalConv2d, deconv_type=SignalConvTranspose2d):
    """setConv_type"""
    global __conv_type__, __deconv_type__
    if isinstance(conv_type, str) and conv_type in __CONV_TYPES__.keys():
        __conv_type__ = __CONV_TYPES__[conv_type]
    elif isinstance(conv_type, type) and conv_type in __CONV_TYPES__.values():
        __conv_type__ = conv_type
    else:
        raise ValueError("Expect conv_type in {}, got {}".format(
            __CONV_TYPES__, type(conv_type)))

    if isinstance(deconv_type, str) and deconv_type in __DECONV_TYPES__.keys():
        __deconv_type__ = __DECONV_TYPES__[deconv_type]
    elif isinstance(deconv_type, type) and deconv_type in __DECONV_TYPES__.values():
        __deconv_type__ = deconv_type
    else:
        raise ValueError("Expect deconv_type in {}, got {}".format(
            __DECONV_TYPES__, type(deconv_type)))


def Conv2d(in_channels, out_channels, kernel_size=5, stride=1, *args, **kwargs):
    """Conv2d"""
    if __conv_type__ != SignalConv2d and 'parameterizer' in kwargs:
        kwargs.pop('parameterizer')
    if 'padding' in kwargs:
        kwargs.pop('padding')
    return __conv_type__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=(kernel_size - 1) // 2, *args, **kwargs)


def ConvTranspose2d(in_channels, out_channels, kernel_size=5, stride=1, *args, **kwargs):
    """ConvTranspose2d"""
    if __deconv_type__ not in [SignalConv2d, SignalConvTranspose2d] and 'parameterizer' in kwargs:
        kwargs.pop('parameterizer')
    if 'padding' in kwargs:
        kwargs.pop('padding')
    if 'output_padding' in kwargs:
        kwargs.pop('output_padding')
    return __deconv_type__(in_channels, out_channels, kernel_size, stride=stride,
                           padding=(kernel_size - 1) // 2, output_padding=stride-1, *args, **kwargs)


 
