import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from torch_compression.models import CompressesModel
from torch_compression.modules.entropy_models import (__CONDITIONS__,
                                                      ConditionalFactorizer,
                                                      EntropyBottleneck,
                                                      EntropyModel)
from torch_compression.util.math import lower_bound
from torch_compression.util.quantization import (noise_quant, quantize,
                                                 random_quant, scale_quant, StochasticGumbelAnnealing)


iprint = print
debug = True


def print(*args, **kwargs):
    if debug:
        iprint(*args, **kwargs)


def check_range(input, name=""):
    if debug:
        iprint(name, "%.4f, %.4f, %.4f, %.4f" %
               (input.min(), input.median(), input.mean(), input.max()))


def space_to_depth(x):
    xs = x.size()
    # Pick off every second element
    x = x.view(xs[0], xs[1], xs[2] // 2, 2, xs[3] // 2, 2)
    # Transpose picked elements next to channels.
    x = x.permute(0, 1, 3, 5, 2, 4)
    # Combine with channels.
    x = x.reshape(xs[0], xs[1] * 4, xs[2] // 2, xs[3] // 2)
    return x


def space_to_depth2(x):
    xs = x.size()
    # Pick off every second element
    x = x.view(xs[0], xs[1], xs[2] // 2, 2, xs[3] // 2, 2)
    # Transpose picked elements next to channels.
    x = x.permute(0, 3, 5, 1, 2, 4)
    # Combine with channels.
    x = x.reshape(xs[0], xs[1] * 4, xs[2] // 2, xs[3] // 2)
    return x


def depth_to_space(x):
    xs = x.size()
    # Pick off elements from channels
    x = x.view(xs[0], xs[1] // 4, 2, 2, xs[2], xs[3])
    # Transpose picked elements next to HW dimensions.
    x = x.permute(0, 1, 4, 2, 5, 3)
    # Combine with HW dimensions.
    x = x.reshape(xs[0], xs[1] // 4, xs[2] * 2, xs[3] * 2)
    return x


def depth_to_space2(x):
    xs = x.size()
    # Pick off elements from channels
    x = x.view(xs[0], 2, 2, xs[1] // 4, xs[2], xs[3])
    # Transpose picked elements next to HW dimensions.
    x = x.permute(0, 3, 4, 1, 5, 2)
    # Combine with HW dimensions.
    x = x.reshape(xs[0], xs[1] // 4, xs[2] * 2, xs[3] * 2)
    return x


class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, rev=False):
        if not rev:
            return space_to_depth(input)
        else:
            return depth_to_space(input)


class ChannelShuffle(nn.Module):
    """Channel Shuffle Layer

    Args:
        num_features(int): Number of channels of input
        groups(int): Number of shuffle groups, if groups is `0`, 
            output will shuffled by initial random indexs, if groups
            grate than `1`, output will shuffled by groups, if groups
            is `-1`, output will be reverse indexs
    """

    def __init__(self, num_features: int, groups: int = 0):
        super().__init__()
        if groups == 0:
            indexs = torch.randperm(num_features)
            self.mode = 'random'
        elif groups > 1:
            indexs = torch.arange(num_features)
            indexs = indexs.reshape(-1, groups).t().flatten()
            self.mode = 'groups'+str(groups)
        elif groups == -1:
            indexs = torch.arange(num_features).flip(0)
            self.mode = 'reverse'

        self.register_buffer('indexs', indexs)
        # TODO save only random seed instead of whole index
        inv_indexs = torch.empty_like(indexs)
        inv_indexs[indexs] = torch.arange(num_features)
        self.register_buffer('inv_indexs', inv_indexs)

    def extra_repr(self):
        return "mode={mode}".format(**self.__dict__)

    def forward(self, input, rev=False):
        if not rev:
            return input[:, self.indexs]
        else:
            return input[:, self.inv_indexs]


def Q(input, scale=1):
    if scale != 1:
        return scale_quant(input, scale)
    else:
        return quantize(input)


class CouplingLayer(nn.Module):
    """CouplingLayer"""

    def __init__(self, in_channels, factor, archi, **kwargs):
        super().__init__()
        self.split_channels = int(in_channels * factor)
        self.NN = archi(in_channels - self.split_channels,
                        self.split_channels, **kwargs)

    def forward(self, input, rev=False):
        z, y = input[:, :self.split_channels], input[:, self.split_channels:]

        loc = self.NN(y)

        loc = Q(loc)

        if not rev:
            z = z + loc
        else:
            z = z - loc

        return torch.cat([z, y], dim=1)


class FactorPrior(nn.Module):
    """FactorPrior"""

    def __init__(self, in_channels, factor, archi, condition, use_mean=True, **kwargs):
        super().__init__()
        self.split_channels = int(in_channels * factor)
        self.NN = archi(in_channels - self.split_channels,
                        self.split_channels * (2 if use_mean else 1), **kwargs)
        self.conditional_bottleneck = __CONDITIONS__[
            condition](use_mean=use_mean, quant_mode="pass")

    def forward(self, input, counting=False, figname=''):
        y, z = input[:, :self.split_channels], input[:, self.split_channels:]
        # z means paper y, y means paper z1

        condition = self.NN(z)

        self.conditional_bottleneck._set_condition(condition)
        self.conditional_bottleneck.mean = Q(self.conditional_bottleneck.mean)
        likelihood = self.conditional_bottleneck._likelihood(y)

        if counting:
            fig, ax = plt.subplots(3)
            ax[0].hist(y.flatten().cpu(), label='data', bins=np.arange(
                y.min().item()-1, y.max().item()+1)+0.5)
            mean = self.conditional_bottleneck.mean.round()
            ax[1].hist(mean.flatten().cpu(), label='mean', bins=np.arange(
                mean.min().item()-1, mean.max().item()+1)+0.5)
            res = y-mean
            ax[2].hist(res.flatten().cpu(), label='res', bins=np.arange(
                res.min().item()-1, res.max().item()+1)+0.5)
            bpd = likelihood.clamp_min(1e-9).log2().neg().mean()
            ax[0].set_title(
                "IDF factor out L{}, bpd={:.4f}".format(counting, bpd.cpu()))
            [ax[i].legend() for i in range(3)]
            plt.savefig(figname+'_factor_out_'+str(counting)+".png")
            plt.close(fig)

            fig, ax = plt.subplots(1)
            ll_sum = likelihood.flatten(2).sum(2)
            max_idx = ll_sum.max(1)[1].item()

            H, W = y.size()[-2:]
            fig, ax = plt.subplots(figsize=(6.4*(W/H*.75), 4.8))
            sns.heatmap(likelihood[0, max_idx].cpu(), ax=ax, vmin=0,
                        linewidths=0, xticklabels=False, yticklabels=False)
            ax.set_title("IDF factor out L{}, bpd={:.4f}".format(counting, bpd.cpu()))
            plt.savefig(figname+'_factor_out_'+str(counting)+"_ll_max.png")
            plt.close('all')

        return z, y, likelihood

    def inverse(self, z, y=None):
        if y is None:
            pass
        return torch.cat([y, z], dim=1)


nn_types = ['shallow', 'resnet', 'densenet']


def BackBone(num_filters, kernel_size=3, nn_type="shallow", densenet_depth=12, **kwargs):
    """BackBone"""
    assert nn_type in nn_types
    padding = (kernel_size-1)//2
    if nn_type == "shallow":
        class Shallow(nn.Sequential):
            def __init__(self, in_channels, out_channels):
                super().__init__(
                    nn.Conv2d(in_channels, num_filters, kernel_size=kernel_size,
                              padding=padding),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_filters, num_filters,
                              kernel_size=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_filters, out_channels, kernel_size=kernel_size,
                              padding=padding)
                )
        return Shallow

    elif nn_type == 'densenet':
        class DenseLayer(nn.Module):
            def __init__(self, in_channels, growth):
                super().__init__()

                self.nn = torch.nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels, growth, kernel_size,
                              padding=padding),
                    nn.ReLU(inplace=True)
                )

            def forward(self, x):
                return torch.cat([x, self.nn(x)], dim=1)

        class DenseBlock(nn.Sequential):
            def __init__(self, in_channels, out_channels):
                depth = densenet_depth

                future_growth = out_channels - in_channels

                layers = []

                for d in range(depth):
                    growth = future_growth // (depth - d)

                    layers.append(DenseLayer(in_channels, growth))
                    in_channels += growth
                    future_growth -= growth

                super().__init__(*layers)

        class DenseNet(nn.Sequential):
            def __init__(self, in_channels, out_channels):
                super().__init__(
                    DenseBlock(in_channels, in_channels + num_filters),
                    nn.Conv2d(num_filters + in_channels, out_channels,
                              kernel_size, padding=padding)
                )

        return DenseNet

    else:
        raise NotImplementedError()


class IntegerDiscreteFlow(CompressesModel):
    """IntegerDiscreteFlow"""

    def __init__(self, in_channels, num_layers=2, num_flows=4, factor=0.5, factor_out=0.5,
                 archi=BackBone(128), use_mean=True, condition='Logistic', factorizer="LogisticMixtureModel",
                 quant_mode='round', num_mixtures=5, **kwargs):
        super().__init__()
        self.quant_mode = quant_mode
        if quant_mode == 'SGA':
            self.SGA = StochasticGumbelAnnealing()
        layers = []

        for L in range(num_layers):
            layers.append(Squeeze())
            in_channels *= 4

            for _ in range(num_flows):
                layers.append(ChannelShuffle(in_channels))
                layers.append(CouplingLayer(
                    in_channels, factor, archi, **kwargs))

            if L < num_layers - 1:
                layers.append(FactorPrior(
                    in_channels, factor_out, archi, condition=condition, use_mean=use_mean, **kwargs))
                in_channels -= layers[-1].split_channels
            elif factorizer == 'factorizer':
                layers.append(EntropyBottleneck(
                    in_channels, quant_mode="pass"))
            else:
                layers.append(ConditionalFactorizer(
                    in_channels, condition=factorizer, use_mean=use_mean, K=num_mixtures, quant_mode="pass"))

        for idx, layer in enumerate(layers):
            self.add_module(str(idx), layer)

    def quantize(self, input, mode, mean=None):
        """Perturb or quantize a `Tensor` and optionally dequantize.

        Arguments:
            input: `Tensor`. The input values.
            mode: String. Can take on one of three values: `'noise'` (adds uniform
                noise), `'dequantize'` (quantizes and dequantizes), and `'symbols'`
                (quantizes and produces integer symbols for range coder).

        Returns:
            The quantized/perturbed `input`. The returned `Tensor` should have type
            `self.dtype` if mode is `'noise'`, `'dequantize'`; `ac.dtype` if mode is
            `'symbols'`.
        """
        if mode == "pass":
            return input
        if mode == 'RSGA':
            return random_quant(input, self.SGA, mean)
        if mode == 'RUN':
            return random_quant(input, noise_quant, mean)
        if mode == 'SGA':
            return self.SGA(input)

        outputs = quantize(input, mode, mean)

        return outputs

    def forward(self, input, break_layer=0, counting=False, figname=''):
        output = Q(input)
        z, likelihoods, zs = output, (), []
        # print(z)

        # print(output.flatten()[:10])
        layer_count = 0
        for layer in self.children():
            # print(layer.__class__.__name__, z.shape)
            # check_range(z, layer.__class__.__name__)
            if isinstance(layer, FactorPrior):
                z, splited, likelihood = layer(
                    z, layer_count+1 if counting else False, figname)
                likelihoods += (likelihood,)
                zs.append(splited)
                layer_count += 1
            elif isinstance(layer, (ConditionalFactorizer, EntropyBottleneck)):
                splited, likelihood = layer(z)
                likelihoods += (likelihood,)
                zs.append(splited)
            else:
                z = layer(z)

            # print(z.shape)
            # print(z)
            if break_layer and layer_count >= break_layer:
                break

        if self.quant_mode != 'round':
            output = self.quantize(
                input, self.quant_mode if self.training else 'round')

        # print(output.flatten()[:10])
        return output, likelihoods, zs

    def inverse(self, zs):
        z = zs.pop()

        for layer in reversed(list(self.children())):
            if isinstance(layer, FactorPrior):
                z = layer.inverse(z, zs.pop())
            elif isinstance(layer, (ConditionalFactorizer, EntropyBottleneck)):
                pass
            else:
                z = layer(z, rev=True)

        return z


class IDFPriorCoder(CompressesModel):
    """IDFPriorCoder"""

    def __init__(self, num_features, **kwargs):
        super(IDFPriorCoder, self).__init__()
        self.analysis = nn.Sequential()
        self.synthesis = nn.Sequential()

        self.IDFPrior = IntegerDiscreteFlow(num_features, **kwargs)

    def forward(self, input, reload_AE=False, break_layer=0, counting=False, figname=''):
        if reload_AE:
            with torch.no_grad():
                features = self.analysis(input)
        else:
            features = self.analysis(input)

        y_tilde, likelihoods, _ = self.IDFPrior(
            features, break_layer, counting, figname)

        if reload_AE:
            reconstructed = input
        else:
            reconstructed = self.synthesis(y_tilde)

        return reconstructed, likelihoods
