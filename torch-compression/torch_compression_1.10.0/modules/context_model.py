import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

import torch_compression as trc
import torch_compression.torchac.torchac as ac
from torch_compression.modules.ConvRNN import ConvLSTM2d
from torch_compression.modules.functional import space_to_depth, depth_to_space

__version__ = '0.9.6'


class MaskedConv2d(nn.Conv2d):
    """Custom Conv2d Layer with mask for context model

    Args:
        as nn.Conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size, mode='A', **kwargs):
        kwargs["padding"] = 0
        super(MaskedConv2d, self).__init__(
            in_channels, out_channels, kernel_size, **kwargs)
        self.mode = mode.upper()
        self._set_mask()
        # print(self._mask)

    def extra_repr(self):
        return super().extra_repr()+", mode={mode}".format(**self.__dict__)

    @property
    def center(self):
        return tuple([(kernel_size - 1) // 2 for kernel_size in self.kernel_size])

    def _set_mask(self):
        self.register_buffer("_mask", torch.zeros(*self.kernel_size))
        center_h, center_w = self.center
        self._mask[:center_h, :] = 1
        self._mask[:center_h+1, :center_w] = 1
        if self.mode == 'B':
            self._mask[center_h, center_w] = 1

    def pad(self, input):
        padding = ()
        for center in reversed(self.center):
            padding += (center,) * 2
        return F.pad(input, pad=padding, mode={'zeros': 'constant', 'border': 'repilcation'}[self.padding_mode])

    def crop(self, input, left_up=None, windows=None):
        """mask conv crop"""
        if left_up is None:
            left_up = self.center
        if windows is None:
            windows = self.kernel_size
        elif isinstance(windows, int):
            windows = (windows, windows)
        return input[:, :, left_up[0]:left_up[0]+windows[0], left_up[1]:left_up[1]+windows[1]]

    def forward(self, input, padding=True):
        if padding:
            input = self.pad(input)
        # for torch>=1.7
        return self._conv_forward(input, self.weight*self._mask, self.bias)
        # for torch==1.4
        #return self.conv2d_forward(input, self.weight*self._mask)


class MaskedConv2d2(nn.Conv2d):
    """Custom Conv2d Layer with mask for context model

    Args:
        as nn.Conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size, mode='A', **kwargs):
        kwargs["padding"] = 0
        super(MaskedConv2d2, self).__init__(
            in_channels, out_channels, kernel_size, **kwargs)
        self._weight = torch.zeros_like(self.weight.flatten(2))
        mask_size = np.prod(self.kernel_size) // 2
        self.mode = mode.upper()
        if self.mode == 'B':
            mask_size += 1
        self.weight_shape = self.weight.size()
        self.weight = nn.Parameter(self.weight.flatten(2)[..., :mask_size])
        self.mask = torch.zeros_like(self._weight).bool()
        self.mask[..., :mask_size] = True
        # print(self._mask)

    def extra_repr(self):
        return super().extra_repr()+", mode={mode}".format(**self.__dict__)

    @property
    def center(self):
        return (self.kernel_size[0]-1) // 2, (self.kernel_size[1]-1) // 2

    def pad(self, input):
        center_h, center_w = self.center
        return F.pad(input, pad=(center_w, center_w, center_h, center_h), mode={'zeros': 'constant', 'border': 'repilcation'}[self.padding_mode])

    def crop(self, input, left_up=None, windows=None):
        """mask conv crop"""
        if left_up is None:
            left_up = self.center
        if windows is None:
            windows = self.kernel_size
        elif isinstance(windows, int):
            windows = (windows, windows)
        return input[:, :, left_up[0]:left_up[0]+windows[0], left_up[1]:left_up[1]+windows[1]]

    def forward(self, input, padding=True):
        if padding:
            input = self.pad(input)
        if self._weight.device != self.weight.device:
            self._weight.to(self.weight.device)
        weight = self._weight.masked_scatter(self.mask, self.weight)
        return self.conv2d_forward(input, weight.reshape(self.weight_shape))


class ContextModel(nn.Module):
    """ContextModel"""

    def __init__(self, num_features, num_phi_features, entropy_model, kernel_size=5):
        super(ContextModel, self).__init__()
        self.num_features = num_features
        assert isinstance(
            entropy_model, trc.SymmetricConditional), type(entropy_model)
        self.entropy_model = entropy_model
        self.mask = MaskedConv2d(num_features, num_phi_features, kernel_size)
        self.padding = (kernel_size-1)//2

        self.reparam = nn.Sequential(
            nn.Conv2d(num_phi_features*2, 640, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(640, 640, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(640, num_features*self.entropy_model.condition_size, 1)
        )

    def _set_condition(self, output, phi, padding=True):
        masked = self.mask(output, padding)
        # assert masked.size() == phi.size(), (masked.size(), phi.size())

        condition = self.reparam(torch.cat([masked, phi], dim=1))
        self.entropy_model._set_condition(condition)

    def get_cdf(self):
        pass

    @torch.no_grad()
    def compress(self, input, condition, return_sym=False):
        """Compress input and store their binary representations into strings.

        Arguments:
            input: `Tensor` with values to be compressed.

        Returns:
            compressed: String `Tensor` vector containing the compressed
                representation of each batch element of `input`.

        Raises:
            ValueError: if `input` has an integral or inconsistent `DType`, or
                inconsistent number of channels.
        """
        symbols = self.entropy_model.quantize(input, "symbols")

        self._set_condition(symbols.float(), condition)

        cdf, cdf_length, offset, idx = self.entropy_model.get_cdf()  # CxL
        assert symbols.dtype == cdf_length.dtype == offset.dtype == torch.int16

        strings = ac.range_index_encode(symbols - offset, cdf, cdf_length, idx)

        if return_sym:
            return strings, self.dequantize(symbols)
        else:
            return strings

    @torch.no_grad()
    def decompress(self, strings, shape, condition):
        """Decompress values from their compressed string representations.

        Arguments:
            strings: A string `Tensor` vector containing the compressed data.

        Returns:
            The decompressed `Tensor`.
        """
        B, C, H, W = [int(s) for s in shape]
        assert B == 1

        # strings, outbound_strings = strings.split(b'\x46\xE2\x84\x91')

        input = self.mask.pad(torch.zeros(size=shape, device=condition.device))

        for h_idx in range(H):
            for w_idx in range(W):
                patch = self.mask.crop(input, (h_idx, w_idx))
                patch_phi = self.mask.crop(condition, (h_idx, w_idx), 1)

                self._set_condition(patch, patch_phi, padding=False)

                cdf, cdf_length, offset, idx = self.entropy_model.get_cdf()

                rec = ac.any_backend.decode_cdf_index(
                    cdf, cdf_length, idx, *string).to(offset.device)

                rec = self.entropy_model.dequantize(rec + offset)
                self.mask.crop(patch, windows=1).copy_(rec)

        return self.mask.crop(input, windows=(H, W))

    def forward(self, input, condition):
        output = self.entropy_model.quantize(
            input, self.entropy_model.quant_mode if self.training else "round")

        self._set_condition(output, condition)

        likelihood = self.entropy_model._likelihood(output)

        return output, likelihood


class ContextModel2(ContextModel):
    def __init__(self, num_features, num_phi_features, entropy_model, kernel_size):
        super().__init__(num_features, num_phi_features,
                         entropy_model, kernel_size=kernel_size)

    def forward(self, input, condition, scale_factor):
        input = input / scale_factor
        output = self.entropy_model.quantize(
            input, self.entropy_model.quant_mode if self.training else "round")

        dequant = output * scale_factor
        self._set_condition(dequant, condition)
        self.entropy_model.mean = self.entropy_model.mean / \
            scale_factor.unsqueeze(1)

        likelihood = self.entropy_model._likelihood(output)

        return dequant, likelihood


class GroupContextModel(nn.Module):
    def __init__(self, num_features, num_phi_features, entropy_model, kernel_size=3):
        super(GroupContextModel, self).__init__()
        self.num_features = num_features
        assert isinstance(
            entropy_model, trc.SymmetricConditional), type(entropy_model)
        self.entropy_model = entropy_model
        padding = (kernel_size-1)//2

        self.reparam = nn.Sequential(
            nn.Conv2d(num_phi_features+num_features,
                      num_features*2, kernel_size, padding=padding),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_features*2, num_features*2,
                      kernel_size, padding=padding),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_features*2, num_features *
                      self.entropy_model.condition_size, kernel_size, padding=padding)
        )
        # self.reparams = nn.ModuleList(
        #     [nn.Sequential(
        #         nn.Conv2d(num_phi_features*4+g * num_features,
        #                   num_features*3//2, kernel_size, padding=padding),
        #         nn.LeakyReLU(inplace=True),
        #         nn.Conv2d(num_features*3//2, num_features*3 //
        #                   2, kernel_size, padding=padding),
        #         nn.LeakyReLU(inplace=True),
        #         nn.Conv2d(num_features*3//2, num_features *
        #                   self.entropy_model.condition_size, kernel_size, padding=padding)
        #     ) for g in range(4)
        #     ])

    def forward(self, input, condition):
        output = self.entropy_model.quantize(
            input, self.entropy_model.quant_mode if self.training else "round")
        outputs = space_to_depth(output, block_size=2).chunk(4, dim=1)
        if condition.size()[-2:] == input.size()[-2:]:
            condition = space_to_depth(condition, block_size=2)

        contexts, likelihoods = [], []
        for g in range(4):
            # print("G", g)
            cond = self.reparam(torch.cat([condition, outputs[g]], dim=1))
            # cond = self.reparams[g](condition)
            # print(cond.shape)
            self.entropy_model._set_condition(cond)
            ll = self.entropy_model._likelihood(outputs[g])

            # condition = torch.cat([condition, outputs[g]], dim=1)
            contexts.append(outputs[g])
            likelihoods.append(ll)

        # output = depth_to_space(torch.cat(contexts, dim=1), 2)
        # likelihood = depth_to_space(torch.cat(lls, dim=1), 2)
        return output, likelihoods


class GroupContextModel2(nn.Module):
    def __init__(self, num_features, num_phi_features, entropy_model, kernel_size=3):
        super(GroupContextModel2, self).__init__()
        self.num_features = num_features
        assert isinstance(
            entropy_model, trc.SymmetricConditional), type(entropy_model)
        self.entropy_model = entropy_model
        padding = (kernel_size-1)//2

        self.aggregation = ConvLSTM2d(
            num_features, num_phi_features, 3, padding=1)

        self.reparam = nn.Sequential(
            nn.Conv2d(num_phi_features+num_features,
                      num_features*2, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_features*2, num_features*2,
                      kernel_size, padding=padding),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_features*2, num_features *
                      self.entropy_model.condition_size, 1)
        )

    def forward(self, input, condition):
        output = self.entropy_model.quantize(
            input, self.entropy_model.quant_mode if self.training else "round")
        outputs = space_to_depth(output, block_size=2).chunk(4, dim=1)
        if condition.size()[-2:] == input.size()[-2:]:
            condition = space_to_depth(condition, block_size=2)

        contexts, lls = [], []
        self.aggregation.set_hidden((condition, condition))
        for g in range(4):
            # print("G", g)
            yi = outputs[g-1] if g else torch.zeros_like(outputs[0])
            hidden = self.aggregation(yi)
            cond = self.reparam(torch.cat([hidden, yi], dim=1))
            # print(cond.shape)
            self.entropy_model._set_condition(cond)
            ll = self.entropy_model._likelihood(outputs[g])

            # condition = torch.cat([condition, outputs[g]], dim=1)
            contexts.append(outputs[g])
            lls.append(ll)

        # output = depth_to_space(torch.cat(contexts, dim=1), 2)
        # likelihood = depth_to_space(torch.cat(lls, dim=1), 2)
        return output, likelihood
