import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, init
from torch.nn.modules.utils import _pair, _single, _triple


def init_hidden(module: nn.Module, hx=None):
    if not isinstance(module, nn.Sequential):
        module = [module]
    for m in module:
        if isinstance(m, ConvRNNCell):
            # print('set')
            m.set_hidden(hx)
        else:
            for child in m.children():
                init_hidden(child, hx)


class ConvRNNCell(nn.Module):
    """ConvRNNCell"""

    def __init__(self, mode, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, batch_first, bidirectional):
        super(ConvRNNCell, self).__init__()
        self.mode = mode
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        if mode == 'LSTM':
            gate_size = 4 * out_channels
        elif mode == 'GRU':
            gate_size = 3 * out_channels
        elif mode == 'RNN_TANH':
            gate_size = out_channels
        elif mode == 'RNN_RELU':
            gate_size = out_channels
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        self.gate_size = gate_size

        gate_size *= (2 if bidirectional else 1)
        in_channels += out_channels

        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, gate_size // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                gate_size, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(gate_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.hx = None

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        if not self.batch_first:
            s += ", batch_first={batch_first}"
        if self.bidirectional:
            s += ", bidirectional={bidirectional}"
        return s.format(**self.__dict__)

    def _init_hidden(self, input):
        size = list(input.size()[-4:])
        size[1] = self.out_channels
        zeros = input.new_zeros(*size)
        hx = zeros
        return hx

    def _forward_parameters(self):
        yield self.weight[:, :self.gate_size] if self.transposed else self.weight[:self.gate_size, :]
        if self.bias is not None:
            yield self.bias[:self.gate_size]

    def _reverse_parameters(self):
        yield self.weight[:, self.gate_size:] if self.transposed else self.weight[self.gate_size:, :]
        if self.bias is not None:
            yield self.bias[self.gate_size:]

    def _conv_forward(self, input, weight, bias):
        raise NotImplementedError()

    def set_hidden(self, hx):
        self.hx = hx

    def _singal_forward(self, input, hx, weight, bias):
        raise NotImplementedError()

    def forward(self, input, hx=None):  # B, T, C, H, W
        if input.dim() == 5 and self.batch_first:
            input = input.transpose(0, 1)  # T, B, C, H, W
        if self.hx is not None:
            hx = self.hx
        if hx is None:
            # print('init')
            hx = self._init_hidden(input)

        if input.dim() == 4:  # B, C, H, W
            hx = self._singal_forward(
                input, hx, *self._forward_parameters())
            self.set_hidden(hx)
            # print('update')
            return hx[0]

        outputs = []
        for T in range(input.size(0)):
            # print(input[T].shape, hx[0].shape, hx[1].shape)
            hx = self._singal_forward(
                input[T], hx, *self._forward_parameters())
            outputs.append(hx[0])

        output = torch.stack(outputs)

        if self.bidirectional:
            r_outputs = []
            if hx is None:
                hx = self._init_hidden(input)
            for T in range(-1, -1-input.size(0), -1):
                # print(input[T].shape, hx[0].shape, hx[1].shape)
                hx = self._singal_forward(
                    input[T], hx, *self._reverse_parameters())
                r_outputs.append(hx[0])

            r_output = torch.stack(r_outputs[::-1])
            output = torch.cat([output, r_output], dim=2)

        if input.dim() == 5 and self.batch_first:
            output = output.transpose(0, 1)
        return output


class _ConvLSTMNd(ConvRNNCell):
    """ConvLSTM"""

    def __init__(self, *args, **kwargs):
        super(_ConvLSTMNd, self).__init__("LSTM", *args, **kwargs)

    def _init_hidden(self, input):
        zeros = super()._init_hidden(input)
        hx = (zeros, zeros)
        return hx

    def _singal_forward(self, input, hx, weight, bias):
        h_cur, c_cur = hx
        # print(torch.cat([input, h_cur], dim=1).shape)

        corr_input = torch.cat([input, h_cur], dim=1)
        combined_conv = self._conv_forward(corr_input, weight, bias)
        It, Ft, Ot, Ct = combined_conv.chunk(4, dim=1)
        # print("SF", It.shape, c_cur.shape)

        c_next = Ft.sigmoid() * c_cur + It.sigmoid() * Ct.tanh()
        h_next = Ot.sigmoid() * c_next.tanh()

        return h_next, c_next


class ConvLSTM2d(_ConvLSTMNd):
    """ConvLSTM"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', batch_first=True, bidirectional=False):
        padding = (kernel_size-1)//2 if padding is None else padding
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(ConvLSTM2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                         dilation, False, _pair(0), groups, bias, padding_mode, batch_first=batch_first, bidirectional=bidirectional)

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)


class TemporalSequential(nn.Module):
    """TemporalSequential"""

    def __init__(self, module: nn.Module, batch_first=True, parallel=True):
        super(TemporalSequential, self).__init__()
        self.m = module

        assert batch_first
        self.batch_first = batch_first
        self.parallel = parallel

    def sequential_forward(self, input):  # B, T, C, H, W
        if self.batch_first:
            input = input.transpose(0, 1)  # T, B, C, H, W

        outputs = []
        for T in range(input.size(0)):
            outputs.append(self.m(input[T]))

        output = torch.stack(outputs)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output

    def parallel_forward(self, input):  # B, T, C, H, W
        output = self.m(input.flatten(0, 1))
        return output.reshape(*input.size()[:2], *output.size()[1:])

    def forward(self, input):
        return self.parallel_forward(input) if self.parallel else self.sequential_forward(input)


if __name__ == "__main__":
    from util.functional import torchseed
    torchseed()
    # t = torch.rand(2, 3, 3)
    # m = nn.LSTM(3, 4, bidirectional=True)
    # print(m)
    # t2 = m(t)
    # print(t2[0].shape)

    m = ConvLSTM2d(3, 4, 3, bidirectional=False)
    print(m)
    m = nn.Sequential(
        nn.Conv2d(3, 3, 3, padding=1),
        nn.Conv2d(3, 3, 3, padding=1),
        ConvLSTM2d(3, 4, 3, bidirectional=False),
        nn.Conv2d(4, 3, 3, padding=1),
        nn.Conv2d(3, 6, 3, padding=1),
    )
    t = torch.rand(2, 3, 4, 4)
    t2 = m(t)
    print(t2.shape)
    t2 = m(t)
    print(t2.shape)
    m[2].set_hidden(None)
    t2 = m(t)
    print(t2.shape)

    # m = TemporalSequential(nn.Conv2d(3, 4, 3, stride=2, padding=1))
    # print(m)
    # t2 = m(t)
    # print(t2.shape)
