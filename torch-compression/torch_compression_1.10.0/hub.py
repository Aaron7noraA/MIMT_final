from math import tau
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_compression.AugmentedNormalizedFlows import AugmentedNormalizedFlow, SFTCondAugmentedNormalizedFlow
from torch_compression.IntegerDescreteFlows import IDFPriorCoder
from torch_compression.models import (CompressesModel, ContextCoder,
                                      FactorizedCoder, HyperPriorCoder)
from torch_compression.modules import (Conv2d, ConvTranspose2d,
                                       GeneralizedDivisiveNorm, SignalConv2d,
                                       SignalConvTranspose2d,
                                       conditional_module)
from torch_compression.modules.activation import *
from torch_compression.modules.attention_module import *
from torch_compression.util.math import lower_bound
from torch_compression.util.toolbox import *
from torch_compression.util.vision import channel_analysis, fft_visual
from torch_compression.util.quantization import quantize
from torch_compression.modules.convlstm import ConvLSTMCell

class ResidualBlock(nn.Sequential):
    """Builds the residual block"""

    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__(
            Conv2d(num_filters, num_filters//2, 1, stride=1),
            nn.ReLU(inplace=True),
            Conv2d(num_filters//2, num_filters//2, 3, stride=1),
            nn.ReLU(inplace=True),
            Conv2d(num_filters//2, num_filters, 1, stride=1)
        )

    def forward(self, input):
        return input + super().forward(input)


class AttentionBlock(nn.Module):
    """Builds the non-local attention block"""

    def __init__(self, num_filters, non_local=False, block_size=2):
        super(AttentionBlock, self).__init__()
        self.trunk_branch = nn.Sequential(
            ResidualBlock(num_filters),
            ResidualBlock(num_filters),
            ResidualBlock(num_filters)
        )

        nl = [NonLocalBlock(num_filters, block_size)] if non_local else []
        nl += [
            ResidualBlock(num_filters),
            ResidualBlock(num_filters),
            ResidualBlock(num_filters),
            Conv2d(num_filters, num_filters, 1, stride=1),
            nn.Sigmoid()
        ]
        self.attention_branch = nn.Sequential(*nl)

    def forward(self, input):
        return input + self.attention_branch(input) * self.trunk_branch(input)


class GoogleAnalysisTransform(nn.Sequential):
    def __init__(self, in_channels, num_features, num_filters, kernel_size, simplify_gdn):
        super(GoogleAnalysisTransform, self).__init__(
            Conv2d(in_channels, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, simplify=simplify_gdn),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, simplify=simplify_gdn),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, simplify=simplify_gdn),
            Conv2d(num_filters, num_features, kernel_size, stride=2)
        )


class GoogleSynthesisTransform(nn.Sequential):
    def __init__(self, out_channels, num_features, num_filters, kernel_size, simplify_gdn):
        super(GoogleSynthesisTransform, self).__init__(
            ConvTranspose2d(num_features, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(
                num_filters, inverse=True, simplify=simplify_gdn),
            ConvTranspose2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(
                num_filters, inverse=True, simplify=simplify_gdn),
            ConvTranspose2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(
                num_filters, inverse=True, simplify=simplify_gdn),
            ConvTranspose2d(num_filters, out_channels, kernel_size, stride=2)
        )


class GoogleHyperAnalysisTransform(nn.Sequential):
    def __init__(self, num_features, num_filters, num_hyperpriors):
        super(GoogleHyperAnalysisTransform, self).__init__(
            Conv2d(num_features, num_filters, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            Conv2d(num_filters, num_filters, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            Conv2d(num_filters, num_hyperpriors, kernel_size=5, stride=2)
        )


class GoogleHyperScaleSynthesisTransform(nn.Sequential):
    def __init__(self, num_features, num_filters, num_hyperpriors):
        super(GoogleHyperScaleSynthesisTransform, self).__init__(
            ConvTranspose2d(num_hyperpriors, num_filters,
                            kernel_size=5, stride=2, parameterizer=None),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size=5, stride=2, parameterizer=None),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters, num_features,
                            kernel_size=3, stride=1, parameterizer=None)
        )


class GoogleHyperSynthesisTransform(nn.Sequential):
    def __init__(self, num_features, num_filters, num_hyperpriors):
        super(GoogleHyperSynthesisTransform, self).__init__(
            ConvTranspose2d(num_hyperpriors, num_filters,
                            kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters, num_filters * 3 // 2,
                            kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters * 3 // 2, num_features,
                            kernel_size=3, stride=1)
        )


class GoogleFactorizedCoder(FactorizedCoder):
    """GoogleFactorizedCoder"""

    def __init__(self, num_filters, num_features, simplify_gdn=False,
                 in_channels=3, out_channels=3, kernel_size=5, quant_mode='noise'):
        super(GoogleFactorizedCoder, self).__init__(
            num_features, quant_mode=quant_mode)

        self.analysis = GoogleAnalysisTransform(
            in_channels, num_features, num_filters, kernel_size, simplify_gdn)

        self.synthesis = GoogleSynthesisTransform(
            out_channels, num_features, num_filters, kernel_size, simplify_gdn)


class GoogleIDFPriorCoder(IDFPriorCoder):
    """GoogleIDFPriorCoder"""

    def __init__(self, num_filters, num_features, simplify_gdn=False,
                 in_channels=3, out_channels=3, kernel_size=5, quant_mode='noise', **kwargs):
        super(GoogleIDFPriorCoder, self).__init__(
            num_features, quant_mode=quant_mode, **kwargs)

        self.analysis = GoogleAnalysisTransform(
            in_channels, num_features, num_filters, kernel_size, simplify_gdn)

        self.synthesis = GoogleSynthesisTransform(
            out_channels, num_features, num_filters, kernel_size, simplify_gdn)


class GoogleHyperPriorCoder(HyperPriorCoder):
    """GoogleHyperPriorCoder"""

    def __init__(self, num_filters, num_features, num_hyperpriors, simplify_gdn=False,
                 in_channels=3, out_channels=3, kernel_size=5, use_mean=False, use_abs=False, use_context=False,
                 condition='Gaussian', quant_mode='noise'):
        super(GoogleHyperPriorCoder, self).__init__(
            num_features, num_hyperpriors, use_mean, use_abs, use_context, condition, quant_mode)
        
        self.analysis = GoogleAnalysisTransform(
            in_channels, num_features, num_filters, kernel_size, simplify_gdn)

        self.synthesis = GoogleSynthesisTransform(
            out_channels, num_features, num_filters, kernel_size, simplify_gdn)

        self.hyper_analysis = GoogleHyperAnalysisTransform(
            num_features, num_filters, num_hyperpriors)

        if self.use_mean:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features*self.conditional_bottleneck.condition_size, num_filters, num_hyperpriors)
        else:
            self.hyper_synthesis = GoogleHyperScaleSynthesisTransform(
                num_features, num_filters, num_hyperpriors)


class TriAttentionBlock(nn.Module):
    """Builds the non-local attention block"""

    def __init__(self, num_filters):
        super(TriAttentionBlock, self).__init__()
        self.trunk_branch = nn.Sequential(
            ResidualBlock(num_filters),
            ResidualBlock(num_filters),
            ResidualBlock(num_filters)
        )

        nl = [TripleAttentionModule()]
        nl += [
            ResidualBlock(num_filters),
            ResidualBlock(num_filters),
            ResidualBlock(num_filters),
            Conv2d(num_filters, num_filters, 1, stride=1),
            nn.Sigmoid()
        ]
        self.attention_branch = nn.Sequential(*nl)

    def forward(self, input):
        return input + self.attention_branch(input) * self.trunk_branch(input)


class GoogleAnalysisTransform2(nn.Sequential):
    def __init__(self, in_channels, num_features, num_filters, kernel_size, simplify_gdn):
        super(GoogleAnalysisTransform2, self).__init__(
            Conv2d(in_channels, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, simplify=simplify_gdn),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, simplify=simplify_gdn),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, simplify=simplify_gdn),
            Conv2d(num_filters, num_features, kernel_size, stride=2),
            TriAttentionBlock(num_features)
        )


class GoogleSynthesisTransform2(nn.Sequential):
    def __init__(self, out_channels, num_features, num_filters, kernel_size, simplify_gdn):
        super(GoogleSynthesisTransform2, self).__init__(
            TriAttentionBlock(num_features),
            ConvTranspose2d(num_features, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(
                num_filters, inverse=True, simplify=simplify_gdn),
            ConvTranspose2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(
                num_filters, inverse=True, simplify=simplify_gdn),
            ConvTranspose2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(
                num_filters, inverse=True, simplify=simplify_gdn),
            ConvTranspose2d(num_filters, out_channels, kernel_size, stride=2)
        )


class GoogleHyperAnalysisTransform2(nn.Sequential):
    def __init__(self, num_features, num_filters, num_hyperpriors):
        super(GoogleHyperAnalysisTransform2, self).__init__(
            Conv2d(num_features, num_filters, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            Conv2d(num_filters, num_filters, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            Conv2d(num_filters, num_hyperpriors, kernel_size=5, stride=2),
            AttentionBlock(num_hyperpriors, non_local=True, block_size=1)
        )


class GoogleHyperSynthesisTransform2(nn.Sequential):
    def __init__(self, num_features, num_filters, num_hyperpriors):
        super(GoogleHyperSynthesisTransform2, self).__init__(
            AttentionBlock(num_hyperpriors, non_local=True, block_size=1),
            ConvTranspose2d(num_hyperpriors, num_filters,
                            kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters, num_filters * 3 // 2,
                            kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters * 3 // 2, num_features,
                            kernel_size=3, stride=1)
        )


class GoogleHyperPriorCoder2(HyperPriorCoder):
    """GoogleHyperPriorCoder"""

    def __init__(self, num_filters, num_features, num_hyperpriors, simplify_gdn=False,
                 in_channels=3, out_channels=3, kernel_size=5, use_mean=False, use_abs=False, use_context=False,
                 condition='Gaussian', quant_mode='noise'):
        super(GoogleHyperPriorCoder2, self).__init__(
            num_features, num_hyperpriors, use_mean, use_abs, use_context, condition, quant_mode)

        self.analysis = GoogleAnalysisTransform(
            in_channels, num_features, num_filters, kernel_size, simplify_gdn)

        self.synthesis = GoogleSynthesisTransform(
            out_channels, num_features, num_filters, kernel_size, simplify_gdn)

        self.hyper_analysis = GoogleHyperAnalysisTransform2(
            num_features, num_filters, num_hyperpriors)

        self.hyper_synthesis = GoogleHyperSynthesisTransform2(
            num_features*self.conditional_bottleneck.condition_size, num_filters, num_hyperpriors)


def ANFNorm(num_features, mode, inverse=False):
    if mode in ["standard", "simplify"]:
        return GeneralizedDivisiveNorm(num_features, inverse, simplify=mode == "simplify")
    elif mode == "layernorm":
        return nn.InstanceNorm2d(num_features)
    elif mode == "pass":
        return nn.Sequential()


class AugmentedNormalizedAnalysisTransform(AugmentedNormalizedFlow):
    def __init__(self, in_channels, num_features, num_filters, kernel_size, use_code, distribution, gdn_mode, use_attn=False, integerlize=False):
        super(AugmentedNormalizedAnalysisTransform, self).__init__(
            Conv2d(in_channels, num_filters, kernel_size, stride=2),
            ANFNorm(num_filters, mode=gdn_mode),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            ANFNorm(num_filters, mode=gdn_mode),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            ANFNorm(num_filters, mode=gdn_mode),
            Conv2d(num_filters, num_features *
                   (2 if use_code else 1), kernel_size, stride=2),
            AttentionBlock(num_features *
                           (2 if use_code else 1), non_local=True) if use_attn else nn.Identity(),
            use_code=use_code, transpose=False, distribution=distribution, integerlize=integerlize
        )


class AugmentedNormalizedSynthesisTransform(AugmentedNormalizedFlow):
    def __init__(self, out_channels, num_features, num_filters, kernel_size, use_code, distribution, gdn_mode, use_attn=False, integerlize=False):
        super(AugmentedNormalizedSynthesisTransform, self).__init__(
            AttentionBlock(
                num_features, non_local=True) if use_attn else nn.Identity(),
            ConvTranspose2d(num_features, num_filters,
                            kernel_size, stride=2),
            ANFNorm(num_filters, inverse=True, mode=gdn_mode),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size, stride=2),
            ANFNorm(num_filters, inverse=True, mode=gdn_mode),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size, stride=2),
            ANFNorm(num_filters, inverse=True, mode=gdn_mode),
            ConvTranspose2d(num_filters, out_channels *
                            (2 if use_code else 1), kernel_size, stride=2),
            use_code=use_code, transpose=True, distribution=distribution, integerlize=integerlize
        )


class AugmentedNormalizedHyperAnalysisTransform(AugmentedNormalizedFlow):
    def __init__(self, num_features, num_filters, num_hyperpriors, use_code, distribution):
        super(AugmentedNormalizedHyperAnalysisTransform, self).__init__(
            Conv2d(num_features, num_filters, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            Conv2d(num_filters, num_filters, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            Conv2d(num_filters, num_hyperpriors *
                   (2 if use_code else 1), kernel_size=5, stride=2),
            use_code=use_code, transpose=False, distribution=distribution
        )


class AugmentedNormalizedHyperSynthesisTransform(AugmentedNormalizedFlow):
    def __init__(self, num_features, num_filters, num_hyperpriors, use_code, distribution):
        super(AugmentedNormalizedHyperSynthesisTransform, self).__init__(
            ConvTranspose2d(num_hyperpriors, num_filters,
                            kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters, num_features *
                            (2 if use_code else 1), kernel_size=3, stride=1),
            use_code=use_code, transpose=True, distribution=distribution
        )


class DQ_ResBlock(nn.Sequential):
    def __init__(self, num_filters):
        super().__init__(
            Conv2d(num_filters, num_filters, 3),
            nn.LeakyReLU(0.2, inplace=True),
            Conv2d(num_filters, num_filters, 3)
        )

    def forward(self, input):
        return super().forward(input) + input


class DeQuantizationModule(nn.Module):

    def __init__(self, in_channels, out_channels, num_filters, num_layers):
        super(DeQuantizationModule, self).__init__()
        self.conv1 = Conv2d(in_channels, num_filters, 3)
        self.resblock = nn.Sequential(
            *[DQ_ResBlock(num_filters) for _ in range(num_layers)])
        self.conv2 = Conv2d(num_filters, num_filters, 3)
        self.conv3 = Conv2d(num_filters, out_channels, 3)

    def forward(self, input):
        conv1 = self.conv1(input)
        x = self.resblock(conv1)
        conv2 = self.conv2(x) + conv1
        conv3 = self.conv3(conv2) + input

        return conv3


class CondAugmentedNormalizedFlowHyperPriorCoder(HyperPriorCoder):
    """CondAugmentedNormalizedFlowHyperPriorCoder"""

    def __init__(self, num_filters, num_features, num_hyperpriors, gdn_mode="standard",
                 in_channels=3, out_channels=3, kernel_size=5, num_layers=1, # Note: out_channels is useless
                 init_code='gaussian', use_DQ=False, share_wei=False, use_code=True, dec_add=False,
                 use_attn=False, use_QE=False,
                 hyper_filters=192, use_mean=False, use_context=False,
                 condition='Gaussian', quant_mode='noise',
                 output_nought=True, # Set False when applying ANFIC for residual coding, which will set output(x_2) as MC frame
                 cond_coupling=False, #Set True when applying ANFIC for residual coding, which will take MC frame as condition
                 num_cond_frames:int =1 # Set 1 when only MC frame is for condition ; >1 whwn multi-refertence frames as conditions
                 ):
        super(CondAugmentedNormalizedFlowHyperPriorCoder, self).__init__(
            num_features, num_hyperpriors, use_mean, False, use_context, condition, quant_mode)
        self.num_layers = num_layers
        self.share_wei = share_wei
        self.output_nought=output_nought
        self.cond_coupling = cond_coupling
        assert num_cond_frames > 0, 'number of conditioning frames must >=1'

        print('self.output_nought = ',self.output_nought)
        print('self.cond_coupling = ',self.cond_coupling)

        if not isinstance(num_filters, list):
            num_filters = [num_filters]
        if len(num_filters) != num_layers:
            num_filters = [num_filters[0]] * num_layers

        if not share_wei:
            self.__delattr__('analysis')
            self.__delattr__('synthesis')

            for i in range(num_layers):
                if not self.cond_coupling:
                    self.add_module('analysis'+str(i), AugmentedNormalizedAnalysisTransform(
                        in_channels, num_features, num_filters[i], kernel_size, 
                        use_code=use_code and init_code != 'zeros', 
                        distribution=init_code, gdn_mode=gdn_mode, 
                        use_attn=use_attn and i == num_layers-1))
                    self.add_module('synthesis'+str(i), AugmentedNormalizedSynthesisTransform(
                        in_channels, num_features, num_filters[i], kernel_size, 
                        use_code=use_code and i != num_layers-1 and not dec_add, 
                        distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn and i == num_layers-1))
                else: # For residual coding ; make transform conditional
                    self.add_module('analysis'+str(i), AugmentedNormalizedAnalysisTransform(
                        #in_channels*2, num_features, num_filters[i], kernel_size, 
                        in_channels*(1+num_cond_frames), num_features, num_filters[i], kernel_size, 
                        use_code=use_code and init_code != 'zeros', 
                        distribution=init_code, gdn_mode=gdn_mode, 
                        use_attn=use_attn and i == num_layers-1))
                    # Not applying condition on synthesis
                    self.add_module('synthesis'+str(i), AugmentedNormalizedSynthesisTransform(
                        in_channels, num_features, num_filters[i], kernel_size, 
                        use_code=use_code and i != num_layers-1 and not dec_add, 
                        distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn and i == num_layers-1))
                    # For making down-scaled MC frame feature for synthesis conditioning
                    #self.add_module('cond_encoder'+str(i), GoogleAnalysisTransform(
                    #   in_channels, num_features, num_filters[i], kernel_size))

                if use_QE:
                    self.add_module(
                        'QE'+str(i), DeQuantizationModule(in_channels, in_channels, 64, 2))
        else: 
            self.analysis = AugmentedNormalizedAnalysisTransform(
                in_channels*(1+num_cond_frames), num_features, num_filters[0], kernel_size, use_code=use_code, distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn)
            self.synthesis = AugmentedNormalizedSynthesisTransform(
                in_channels, num_features, num_filters[0], kernel_size, use_code=use_code, distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn)

            if use_QE:
                for i in range(num_layers):
                    self.add_module(
                        'QE'+str(i), DeQuantizationModule(in_channels, in_channels, 64, 2))

        self.hyper_analysis = GoogleHyperAnalysisTransform(
            num_features, hyper_filters, num_hyperpriors) if not use_attn else GoogleHyperAnalysisTransform2(num_features, hyper_filters, num_hyperpriors)

        if use_context:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features*2, hyper_filters, num_hyperpriors) if not use_attn else GoogleHyperSynthesisTransform2(num_features*2, hyper_filters, num_hyperpriors)
        elif self.use_mean or "Mixture" in condition:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features*self.conditional_bottleneck.condition_size, hyper_filters, num_hyperpriors) if not use_attn else GoogleHyperSynthesisTransform2(num_features*self.conditional_bottleneck.condition_size, hyper_filters, num_hyperpriors)
        else:
            self.hyper_synthesis = GoogleHyperScaleSynthesisTransform(
                num_features, hyper_filters, num_hyperpriors)

        self.use_QE = use_QE

        for name, m in self.named_children():
            if "ana" in name or "syn" in name:
                m.name = name

        if use_DQ:
            self.DQ = DeQuantizationModule(in_channels, in_channels, 64, 6)
        else:
            self.DQ = None

    def __getitem__(self, key):
        return self.__getattr__(key[:-1] if self.share_wei else key)

    def encode(self, input, code=None, jac=None, visual=False, figname='', cond_coupling_input=None):
        for i in range(self.num_layers):
            debug('F', i)
            # Concat input with condition (MC frame)
            if self.cond_coupling:
                cond = cond_coupling_input
                cond_input = torch.cat([input, cond], dim=1)
                _, code, jac = self['analysis'+str(i)](
                    cond_input, code, jac, layer=i, visual=visual, figname=figname+"_"+str(i))
            else:
                _, code, jac = self['analysis'+str(i)](
                    input, code, jac, layer=i, visual=visual, figname=figname+"_"+str(i))

            if i < self.num_layers-1:
                debug('S', i)
                input, _, jac = self['synthesis'+str(i)](
                    input, code, jac, layer=i, visual=visual, figname=figname+"_"+str(i))

            if visual:
                logger.write(check_range(input, "X"+str(i)))
                logger.write(check_range(code, "code"+str(i)))
                logger.write()

                channel_analysis(code, figname+f"_FCA_{i}.png")

        return input, code, jac

    def decode(self, input, code=None, jac=None, rec_code=False, visual=False, figname='', cond_coupling_input=None):
        for i in range(self.num_layers-1, -1, -1):
            debug('rF', i)
            
            input, _, jac = self['synthesis'+str(i)](
                input, code, jac, rev=True, last_layer=i == self.num_layers-1, layer=i, visual=visual, figname=figname+"_rev_"+str(i))

            if self.use_QE:
                BQE = input
                input = self['QE'+str(i)](input)
                if visual:
                    save_image(BQE, figname+f"_BQE_{i}.png")
                    save_image(input, figname+f"_AQE_{i}.png")

            if i or rec_code or jac is not None:
                debug('RF', i)
                # Concat input with condition (MC frame)
                if self.cond_coupling:
                    cond = cond_coupling_input
                    cond_input = torch.cat([input, cond], dim=1)
                    _, code, jac = self['analysis'+str(i)](
                        cond_input, code, jac, layer=i, rev=True, visual=visual, figname=figname+"_rev_"+str(i))
                else:
                    _, code, jac = self['analysis'+str(i)](
                        input, code, jac, layer=i, rev=True, visual=visual, figname=figname+"_rev_"+str(i))

                #_, code, jac = self['analysis' +
                #                    str(i)](input, code, jac, rev=True, layer=i, visual=visual, figname=figname+"_rev_"+str(i))

            if visual:
                logger.write(check_range(input, "X"+str(i)))
                logger.write(check_range(code, "code"+str(i)))
                logger.write()

        return input, code, jac

    def entropy_model(self, input, code, jac=False):
        # Enrtopy coding
        debug('E')

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition)
        # y_tilde = code

        # Encode distortion
        Y_error, _, jac = self['synthesis'+str(self.num_layers-1)](
            input, y_tilde, jac, last_layer=True, layer=self.num_layers-1)

        return Y_error, y_tilde, z_tilde, y_likelihood, z_likelihood

    def compress(self, input, cond_coupling_input=None, return_hat=False, reverse_input=None):
        if self.cond_coupling:
            assert not (cond_coupling_input is None), "cond_coupling_input should be specified"
        
        code = None
        jac = None
        input, features, jac = self.encode(
            input, code, jac, visual=visual, figname=figname, cond_coupling_input=cond_coupling_input)

        hyperpriors = self.hyper_analysis(
            features.abs() if self.use_abs else features)

        side_stream, z_hat = self.entropy_bottleneck.compress(
            hyperpriors, return_sym=True)

        condition = self.hyper_synthesis(z_hat)

        ret = self.conditional_bottleneck.compress(
            features, condition=condition, return_sym=return_hat)

        if return_hat:
            jac = None
            stream, y_hat = ret

            # Decode
            if not self.output_nought:
                assert not (reverse_input is None), "reverse_input should be specified"
                input = reverse_input
            else:
                input = torch.zeros_like(input)
                
            x_hat, code, jac = self.decode(
                input, y_hat, jac, cond_coupling_input=cond_coupling_input)
            if self.DQ is not None:
                x_hat = self.DQ(x_hat)

            return x_hat, [stream, side_stream], [hyperpriors.size()]
        else:
            stream = ret
            return [stream, side_stream], [hyperpriors.size()]

    def decompress(self, strings, shape): #TODO
        stream, side_stream = strings
        z_shape = shape

        z_hat = self.entropy_bottleneck.decompress(side_stream, z_shape)

        condition = self.hyper_synthesis(z_hat)

        y_hat = self.conditional_bottleneck.decompress(
            stream, condition.size(), condition=condition)

        reconstructed = self.synthesis(y_hat)

        return reconstructed

    def forward(self, input, code=None, jac=None, rec_code=False, rev_ng=False, 
                IDQ=False, detach_enc=False, visual=False, figname='',
                output=None, # Should assign value when self.output_nought==False
                cond_coupling_input=None # Should assign value when self.cond_coupling==True
                                         # When using ANFIC on residual coding, output & cond_coupling_input should both be MC frame
                                                                         
               ):
        if not self.output_nought:
            assert not (output is None), "output should be specified"
        if self.cond_coupling:
            assert not (cond_coupling_input is None), "cond_coupling_input should be specified"

        # Encode
        if IDQ:
            debug("IDQ")
            input = input.add(torch.rand_like(input).sub_(0.5).div_(255.))

        ori_input = input

        if visual:
            save_image(input, figname+"_input.png")
            if not (output is None):
                save_image(output, figname+"_mc_frame.png")
            visualizer.set_mean_var(input)
            visualizer.queue_visual(input, figname+"_ori.png")
            fft_visual(input, figname+"_ori.png")
            logger.open(figname+"_stati.txt")
            logger.write("min\tmedian\tmean\tvar\tmax\n")

        jac = [] if jac else None
        input, code, jac = self.encode(
            input, code, jac, visual=visual, figname=figname, cond_coupling_input=cond_coupling_input)

        # Enrtopy coding
        debug('E')

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition = self.hyper_synthesis(z_tilde)
        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition)
        # y_tilde = code # No quantize on z2

        # Encode distortion
        x_2, _, jac = self['synthesis'+str(self.num_layers-1)](
            input, y_tilde, jac, last_layer=True, layer=self.num_layers-1, visual=visual, figname=figname+"_"+str(self.num_layers-1))

        if visual:
            visualizer.plot_queue(figname+'_for.png', nrow=input.size(0))

            with torch.no_grad(): # Visualize z likelihood
                YLL_perC = y_likelihood[0].clamp_min(
                    1e-9).log2().mean(dim=(1, 2)).neg()
                fig = plt.figure()
                check_range(YLL_perC, "YLL")
                max_idx = YLL_perC.max(0)[1]
                plt.plot(YLL_perC.data.cpu(),
                         label="maxC="+str(max_idx.item()))
                plt.legend()

                plt.savefig(figname+"_YLL.png")
                plt.close(fig)
            
            # Print feature ; code: z2, no quant ; y_tilde: \hat z2
            #     nrow = 8
            #     save_image(code.div(255*2).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_feature.png", nrow=nrow)
            #     save_image(max_norm(code).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_feature_norm.png", nrow=nrow)
            #     save_image(y_tilde.div(255*2).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_quant.png", nrow=nrow)
            #     save_image(max_norm(y_tilde).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_quant_norm.png", nrow=nrow)

            #logger.write(check_range(Y_error, "X"+str(self.num_layers-1)))
            logger.write(check_range(x_2, "X"+str(self.num_layers-1)))
            logger.write(check_range(code, "code"+str(self.num_layers-1)))
            logger.write()

        if self.entropy_bottleneck.quant_mode == "SGA" and self.training:
            tau = self.entropy_bottleneck.SGA.tau.item()*0.01
            #input, code, hyper_code = Y_error*tau, y_tilde, z_tilde
            input, code, hyper_code = x_2*tau, y_tilde, z_tilde
        else:
            #input, code, hyper_code = torch.zeros_like(input), y_tilde, z_tilde
            #input, code, hyper_code = x_2, y_tilde, z_tilde # x_2 directly backward
            input, code, hyper_code = output, y_tilde, z_tilde # Correct setting ; MC frame as x_2 when decoding

        # input = Y_error
        #debug(Y_error.shape, code.shape, hyper_code.shape)
        debug(x_2.shape, code.shape, hyper_code.shape)

        if visual:
            logger.write(check_range(input, "X"+str(self.num_layers-1)))
            logger.write(check_range(code, "code"+str(self.num_layers-1)))
            logger.write()

        # Decode
        input, code, jac = self.decode(
            input, code, jac, rec_code=rec_code, visual=visual, figname=figname, cond_coupling_input=cond_coupling_input)

        if visual:
            visualizer.plot_queue(figname+'_rev.png', nrow=input.size(0))
            logger.close()

        # Quality enhancement
        if self.DQ is not None:
            debug("DQ")
            BDQ = input
            input = self.DQ(input)
            if visual:
                save_image(x_2, figname+"_x_2.png")
                save_image(BDQ, figname+"_BDQ.png")
                save_image(input, figname+"_ADQ.png")
        else:
            BDQ = None

        debug('END\n')

        #return input, (y_likelihood, z_likelihood), Y_error, jac, code, BDQ
        return input, (y_likelihood, z_likelihood), x_2, jac, code, BDQ


class CondAugmentedNormalizedFlowHyperPriorCoder2(HyperPriorCoder):
    """CondAugmentedNormalizedFlowHyperPriorCoder2"""
    """Both enc & dec have condition"""

    def __init__(self, num_filters, num_features, num_hyperpriors, gdn_mode="standard",
                 in_channels=3, out_channels=3, kernel_size=5, num_layers=1,
                 init_code='gaussian', use_DQ=False, share_wei=False, use_code=True, dec_add=False,
                 use_attn=False, use_QE=False,
                 simplify_gdn=False,
                 hyper_filters=192, use_mean=False, use_context=False,
                 condition='Gaussian', quant_mode='noise',
                 output_nought=True, # Set False when applying ANFIC for residual coding, which will set output(x_2) as MC frame
                 cond_coupling=False # Set True when applying ANFIC for residual coding, which will take MC frame as condition
                 ):
        super(CondAugmentedNormalizedFlowHyperPriorCoder2, self).__init__(
            num_features, num_hyperpriors, use_mean, False, use_context, condition, quant_mode)
        self.num_layers = num_layers
        self.share_wei = share_wei
        self.output_nought=output_nought
        self.cond_coupling = cond_coupling

        print('self.output_nought = ',self.output_nought)
        print('self.cond_coupling = ',self.cond_coupling)

        if not isinstance(num_filters, list):
            num_filters = [num_filters]
        if len(num_filters) != num_layers:
            num_filters = [num_filters[0]] * num_layers

        if not share_wei:
            self.__delattr__('analysis')
            self.__delattr__('synthesis')

            for i in range(num_layers):
                if not self.cond_coupling:# Default
                    self.add_module('analysis'+str(i), AugmentedNormalizedAnalysisTransform(
                        in_channels, num_features, num_filters[i], kernel_size, 
                        use_code=use_code and init_code != 'zeros', 
                        distribution=init_code, gdn_mode=gdn_mode, 
                        use_attn=use_attn and i == num_layers-1))
                    self.add_module('synthesis'+str(i), AugmentedNormalizedSynthesisTransform(
                        in_channels, num_features, num_filters[i], kernel_size, 
                        use_code=use_code and i != num_layers-1 and not dec_add, 
                        distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn and i == num_layers-1))
                else: # For residual coding ; make transform conditional
                    self.add_module('analysis'+str(i), AugmentedNormalizedAnalysisTransform(
                        in_channels*2, num_features, num_filters[i], kernel_size, 
                        use_code=use_code and init_code != 'zeros', 
                        distribution=init_code, gdn_mode=gdn_mode, 
                        use_attn=use_attn and i == num_layers-1))
                    # Not applying condition on synthesis
                    self.add_module('synthesis'+str(i), AugmentedNormalizedSynthesisTransform(
                        in_channels, num_features*2, num_filters[i], kernel_size, 
                        #in_channels, num_features, num_filters[i], kernel_size, 
                        use_code=use_code and i != num_layers-1 and not dec_add, 
                        distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn and i == num_layers-1))
                   #  For making down-scaled MC frame feature for synthesis conditioning
                    self.add_module('cond_encoder'+str(i), GoogleAnalysisTransform(
                       in_channels, num_features, num_filters[i], kernel_size, simplify_gdn))

                if use_QE:
                    self.add_module(
                        'QE'+str(i), DeQuantizationModule(in_channels, in_channels, 64, 2))
        else: #TODO: Double analysis & synthesis input channels when self.cond_coupling==True
            self.analysis = AugmentedNormalizedAnalysisTransform(
                in_channels, num_features, num_filters[0], kernel_size, use_code=use_code, distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn)
            self.synthesis = AugmentedNormalizedSynthesisTransform(
                in_channels, num_features, num_filters[0], kernel_size, use_code=use_code, distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn)

            if use_QE:
                for i in range(num_layers):
                    self.add_module(
                        'QE'+str(i), DeQuantizationModule(in_channels, in_channels, 64, 2))

        self.hyper_analysis = GoogleHyperAnalysisTransform(
            num_features, hyper_filters, num_hyperpriors) if not use_attn else GoogleHyperAnalysisTransform2(num_features, hyper_filters, num_hyperpriors)

        if use_context:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features*2, hyper_filters, num_hyperpriors) if not use_attn else GoogleHyperSynthesisTransform2(num_features*2, hyper_filters, num_hyperpriors)
        elif self.use_mean or "Mixture" in condition:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features*self.conditional_bottleneck.condition_size, hyper_filters, num_hyperpriors) if not use_attn else GoogleHyperSynthesisTransform2(num_features*self.conditional_bottleneck.condition_size, hyper_filters, num_hyperpriors)
        else:
            self.hyper_synthesis = GoogleHyperScaleSynthesisTransform(
                num_features, hyper_filters, num_hyperpriors)

        self.use_QE = use_QE

        for name, m in self.named_children():
            if "ana" in name or "syn" in name:
                m.name = name

        if use_DQ:
            self.DQ = DeQuantizationModule(in_channels, in_channels, 64, 6)
        else:
            self.DQ = None

    def __getitem__(self, key):
        return self.__getattr__(key[:-1] if self.share_wei else key)

    def encode(self, input, code=None, jac=None, visual=False, figname='', cond_coupling_input=None):
        for i in range(self.num_layers):
            debug('F', i)
            # Concat input with condition (MC frame)
            if self.cond_coupling:
                cond = cond_coupling_input
                cond_input = torch.cat([input, cond], dim=1)
                _, code, jac = self['analysis'+str(i)](
                    cond_input, code, jac, layer=i, visual=visual, figname=figname+"_"+str(i))
            else:
                _, code, jac = self['analysis'+str(i)](
                    input, code, jac, layer=i, visual=visual, figname=figname+"_"+str(i))

            if i < self.num_layers-1:
                debug('S', i)
                # Concat code with condition (MC frame feature)
                if self.cond_coupling:
                    cond = self['cond_encoder'+str(i)](cond_coupling_input)
                    cond_code = torch.cat([code, cond], dim=1)
                    input, _, jac = self['synthesis'+str(i)](
                        input, cond_code, jac, layer=i, visual=visual, figname=figname+"_"+str(i))
                else:
                    input, _, jac = self['synthesis'+str(i)](
                        input, code, jac, layer=i, visual=visual, figname=figname+"_"+str(i))

            if visual:
                logger.write(check_range(input, "X"+str(i)))
                logger.write(check_range(code, "code"+str(i)))
                logger.write()

                channel_analysis(code, figname+f"_FCA_{i}.png")

        return input, code, jac

    def decode(self, input, code=None, jac=None, rec_code=False, visual=False, figname='', cond_coupling_input=None):
        for i in range(self.num_layers-1, -1, -1):
            debug('rF', i)
            # Concat code with condition (MC frame feature)
            if self.cond_coupling:
                cond = self['cond_encoder'+str(i)](cond_coupling_input)
                cond_code = torch.cat([code, cond], dim=1)
                input, _, jac = self['synthesis'+str(i)](
                    input, cond_code, jac, rev=True, last_layer=i == self.num_layers-1, layer=i, visual=visual, figname=figname+"_rev_"+str(i))
            else:
                input, _, jac = self['synthesis'+str(i)](
                    input, code, jac, rev=True, last_layer=i == self.num_layers-1, layer=i, visual=visual, figname=figname+"_rev_"+str(i))

            if self.use_QE:
                BQE = input
                input = self['QE'+str(i)](input)
                if visual:
                    save_image(BQE, figname+f"_BQE_{i}.png")
                    save_image(input, figname+f"_AQE_{i}.png")

            if i or rec_code or jac is not None:
                debug('RF', i)
                # Concat input with condition (MC frame)
                if self.cond_coupling:
                    cond = cond_coupling_input
                    cond_input = torch.cat([input, cond], dim=1)
                    _, code, jac = self['analysis'+str(i)](
                        cond_input, code, jac, layer=i, rev=True, visual=visual, figname=figname+"_rev_"+str(i))
                else:
                    _, code, jac = self['analysis'+str(i)](
                        input, code, jac, layer=i, rev=True, visual=visual, figname=figname+"_rev_"+str(i))

                #_, code, jac = self['analysis' +
                #                    str(i)](input, code, jac, rev=True, layer=i, visual=visual, figname=figname+"_rev_"+str(i))

            if visual:
                logger.write(check_range(input, "X"+str(i)))
                logger.write(check_range(code, "code"+str(i)))
                logger.write()

        return input, code, jac

    def entropy_model(self, input, code, jac=False):
        # Enrtopy coding
        debug('E')

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition)
        # y_tilde = code

        # Encode distortion
        Y_error, _, jac = self['synthesis'+str(self.num_layers-1)](
            input, y_tilde, jac, last_layer=True, layer=self.num_layers-1)

        return Y_error, y_tilde, z_tilde, y_likelihood, z_likelihood

    def forward(self, input, code=None, jac=None, rec_code=False, rev_ng=False, 
                IDQ=False, detach_enc=False, visual=False, figname='',
                output=None, # Should assign value when self.output_nought==False
                cond_coupling_input=None # Should assign value when self.cond_coupling==True
                                         # When using ANFIC on residual coding, output & cond_coupling_input should both be MC frame
                                                                         
               ):
        if not self.output_nought:
            assert not (output is None), "output should be specified"
        if self.cond_coupling:
            assert not (cond_coupling_input is None), "cond_coupling_input should be specified"

        # Encode
        if IDQ:
            debug("IDQ")
            input = input.add(torch.rand_like(input).sub_(0.5).div_(255.))

        ori_input = input

        if visual:
            save_image(input, figname+"_input.png")
            if not (output is None):
                save_image(output, figname+"_mc_frame.png")
            visualizer.set_mean_var(input)
            visualizer.queue_visual(input, figname+"_ori.png")
            fft_visual(input, figname+"_ori.png")
            logger.open(figname+"_stati.txt")

        jac = [] if jac else None
        input, code, jac = self.encode(
            input, code, jac, visual=visual, figname=figname, cond_coupling_input=cond_coupling_input)

        # Enrtopy coding
        debug('E')

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition)
        # y_tilde = code # No quantize on z2

        # Encode distortion
        # Concat code with condition (MC frame feature)
        if self.cond_coupling:
            cond = self['cond_encoder'+str(self.num_layers-1)](cond_coupling_input)
            cond_code = torch.cat([y_tilde, cond], dim=1)
        #Y_error, _, jac = self['synthesis'+str(self.num_layers-1)](
        x_2, _, jac = self['synthesis'+str(self.num_layers-1)](
            input, cond_code, jac, last_layer=True, layer=self.num_layers-1, visual=visual, figname=figname+"_"+str(self.num_layers-1))

        if visual:
            visualizer.plot_queue(figname+'_for.png', nrow=input.size(0))

            with torch.no_grad(): # Visualize z likelihood
                YLL_perC = y_likelihood[0].clamp_min(
                    1e-9).log2().mean(dim=(1, 2)).neg()
                fig = plt.figure()
                check_range(YLL_perC, "YLL")
                max_idx = YLL_perC.max(0)[1]
                plt.plot(YLL_perC.data.cpu(),
                         label="maxC="+str(max_idx.item()))
                plt.legend()

                plt.savefig(figname+"_YLL.png")
                plt.close(fig)

        # Print feature ; code: z2, no quant ; y_tilde: \hat z2
        #     nrow = 8
        #     save_image(code.div(255*2).add(0.5).flatten(0, 1).unsqueeze(
        #         1), figname+"_feature.png", nrow=nrow)
        #     save_image(max_norm(code).add(0.5).flatten(0, 1).unsqueeze(
        #         1), figname+"_feature_norm.png", nrow=nrow)
        #     save_image(y_tilde.div(255*2).add(0.5).flatten(0, 1).unsqueeze(
        #         1), figname+"_quant.png", nrow=nrow)
        #     save_image(max_norm(y_tilde).add(0.5).flatten(0, 1).unsqueeze(
        #         1), figname+"_quant_norm.png", nrow=nrow)

        #logger.write(check_range(Y_error, "X"+str(self.num_layers-1)))
        logger.write(check_range(x_2, "X"+str(self.num_layers-1)))
        logger.write(check_range(code, "code"+str(self.num_layers-1)))
        logger.write()

        if self.entropy_bottleneck.quant_mode == "SGA" and self.training:
            tau = self.entropy_bottleneck.SGA.tau.item()*0.01
            #input, code, hyper_code = Y_error*tau, y_tilde, z_tilde
            input, code, hyper_code = x_2*tau, y_tilde, z_tilde
        else:
            #input, code, hyper_code = torch.zeros_like(input), y_tilde, z_tilde
            #input, code, hyper_code = x_2, y_tilde, z_tilde # x_2 directly backward
            input, code, hyper_code = output, y_tilde, z_tilde # Correct setting ; MC frame as x_2 when decoding

        # input = Y_error
        #debug(Y_error.shape, code.shape, hyper_code.shape)
        debug(x_2.shape, code.shape, hyper_code.shape)

        if visual:
            logger.write(check_range(input, "X"+str(self.num_layers-1)))
            logger.write(check_range(code, "code"+str(self.num_layers-1)))
            logger.write()

        # Decode
        input, code, jac = self.decode(
            input, code, jac, rec_code=rec_code, visual=visual, figname=figname, cond_coupling_input=cond_coupling_input)

        if visual:
            visualizer.plot_queue(figname+'_rev.png', nrow=input.size(0))
            logger.close()

        # Quality enhancement
        if self.DQ is not None:
            debug("DQ")
            BDQ = input
            input = self.DQ(input)
            if visual:
                save_image(x_2, figname+"_x_2.png")
                save_image(BDQ, figname+"_BDQ.png")
                save_image(input, figname+"_ADQ.png")
            else:
                BDQ = None

            debug('END\n')

        #return input, (y_likelihood, z_likelihood), Y_error, jac, code, BDQ
        return input, (y_likelihood, z_likelihood), x_2, jac, code, BDQ



# ---------------------------------------------- SFTCondANFCoder -------------------------------------------- #

class SFT(nn.Module):
    def __init__(self, in_channels, cond_channels, num_hiddens=64, kernel_size=3):
        super(SFT, self).__init__()
        
        self.mlp_shared = nn.Sequential(
                              Conv2d(cond_channels, num_hiddens, kernel_size=kernel_size),
                              nn.ReLU()
                          )
        self.mlp_gamma = Conv2d(num_hiddens, in_channels, kernel_size=kernel_size)
        self.mlp_beta = Conv2d(num_hiddens, in_channels, kernel_size=kernel_size)

    def forward(self, input, cond):
        actv = self.mlp_shared(cond)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        return input * (1 + gamma) + beta

class SFTResidualBlock(nn.Module):
    """Builds the residual block"""

    def __init__(self, num_filters, cond_channels):
        super(SFTResidualBlock, self).__init__()
        self.sft = SFT(num_filters, cond_channels, num_hiddens=64, kernel_size=3)
        self.bottleneck = nn.Sequential(
            Conv2d(num_filters, num_filters//2, 1, stride=1),
            nn.ReLU(inplace=True),
            Conv2d(num_filters//2, num_filters//2, 3, stride=1),
            nn.ReLU(inplace=True),
            Conv2d(num_filters//2, num_filters, 1, stride=1)
        )

    def forward(self, input, cond):
        x = self.SFT(input, cond)
        return input + self.bottleneck(x)


class SFTAttentionBlock(nn.Module):
    """Builds the non-local attention block"""

    def __init__(self, num_filters, cond_channels, non_local=False, block_size=2):
        super(SFTAttentionBlock, self).__init__()
        self.trunk_branch_resblk1 = SFTResidualBlock(num_filters, cond_channels),
        self.trunk_branch_resblk2 = SFTResidualBlock(num_filters, cond_channels)
        self.trunk_branch_resblk3 = SFTResidualBlock(num_filters, cond_channels)

        nl = [NonLocalBlock(num_filters, block_size)] if non_local else []
        self.nl = nn.Sequential(*nl)

        self.attention_branch_resblk1 = SFTResidualBlock(num_filters, cond_channels)
        self.attention_branch_resblk2 = SFTResidualBlock(num_filters, cond_channels)
        self.attention_branch_resblk3 = SFTResidualBlock(num_filters, cond_channels)

        self.mask = nn.Sequential(
            Conv2d(num_filters, num_filters, 1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, input, cond): 
        trunk_output = self.trunk_branch_resblk1(input, cond)
        trunk_output = self.trunk_branch_resblk2(trunk_output, cond)
        trunk_output = self.trunk_branch_resblk3(trunk_output, cond)
        
        attention_output = self.attention_branch_resblk1(self.nl(input), cond)
        attention_output = self.attention_branch_resblk2(attention_output, cond)
        attention_output = self.attention_branch_resblk3(attention_output, cond)

        attention_output = self.mask(attention_output)
        return input + attention_output * trunk_output


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, num_filters=64, kernel_size=3):
        super(FeatureExtractor, self).__init__()
        self.conv1 = Conv2d(in_channels, num_filters, kernel_size, stride=2)
        self.conv2 = Conv2d(num_filters, num_filters, kernel_size, stride=2)
        self.conv3 = Conv2d(num_filters, num_filters, kernel_size, stride=2)
        self.conv4 = Conv2d(num_filters, num_filters, kernel_size, stride=2)
    def forward(self, input):
        feat_1 = self.conv1(input)
        feat_2 = self.conv2(feat_1)
        feat_3 = self.conv3(feat_2)
        feat_4 = self.conv4(feat_3)
        
        return feat_1, feat_2, feat_3, feat_4


class SFTCondAugmentedNormalizedAnalysisTransform(SFTCondAugmentedNormalizedFlow):
    def __init__(self, in_channels, num_features, num_filters, kernel_size,
                 use_code, distribution, gdn_mode,
                 transpose=False, clamp=1, integerlize=False,
                 use_attn=False,
                 share_cond_net=True, # Share feature extractor for conditioning with SynthesisTransform
                 cond_net=None # Should be given when share_cond_net==True
                ):
        super(SFTCondAugmentedNormalizedAnalysisTransform, self).__init__(use_code, distribution, gdn_mode, transpose, clamp, integerlize)
        self.use_attn = use_attn

        self.conv1 = nn.Sequential(
            Conv2d(in_channels, num_filters, kernel_size, stride=2),
            ANFNorm(num_filters, mode=gdn_mode),
        )
        self.sft1 = SFT(num_filters, num_filters, num_hiddens=64, kernel_size=3)

        self.conv2 = nn.Sequential(
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            ANFNorm(num_filters, mode=gdn_mode)
        )
        self.sft2 = SFT(num_filters, num_filters, num_hiddens=64, kernel_size=3)

        self.conv3 = nn.Sequential(
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            ANFNorm(num_filters, mode=gdn_mode)
        )
        self.sft3 = SFT(num_filters, num_filters, num_hiddens=64, kernel_size=3)

        self.conv4 = Conv2d(num_filters, num_features *
                      (2 if use_code else 1), kernel_size, stride=2)


        self.attention = SFTAttentionBlock(num_features *
                           (2 if use_code else 1), cond_channels=num_filters, non_local=True) if self.use_attn \
                                                                                              else SFT(num_features, num_filters, num_hiddens=64, kernel_size=3)
    
        if share_cond_net:
            assert isinstance(cond_net, nn.Module), "cond_net should be given"
            self.feat_net = cond_net
        else:
            self.feat_net = FeatureExtractor(in_channels, num_filters)


    def net_forward(self, input, sft_cond):
        feat1, feat2, feat3, feat4 = self.feat_net(sft_cond)
        x = self.conv1(torch.cat([input, sft_cond], dim=1))
        x = self.sft1(x, feat1)
        
        x = self.conv2(x)
        x = self.sft2(x, feat2)
        
        x = self.conv3(x)
        x = self.sft3(x, feat3)

        x = self.conv4(x)
        x = self.attention(x, feat4)

        return x


class SFTCondAugmentedNormalizedSynthesisTransform(SFTCondAugmentedNormalizedFlow):
    def __init__(self, out_channels, num_features, num_filters, kernel_size,
                 use_code, distribution, gdn_mode,
                 transpose=True, clamp=1, integerlize=False,
                 use_attn=False,
                 share_cond_net=True, # Share feature extractor for conditioning with AnalysisTransform
                 cond_net=None # Should be given when share_cond_net==True
                ):
        super(SFTCondAugmentedNormalizedSynthesisTransform, self).__init__(use_code, distribution, gdn_mode, transpose, clamp, integerlize)
        self.use_attn = use_attn

        self.attention = SFTAttentionBlock(num_features *
                           (2 if use_code else 1), cond_channels=num_filters, non_local=True) if self.use_attn \
                                                                                              else SFT(num_features, num_filters, num_hiddens=64, kernel_size=3)

        self.conv1 = nn.Sequential(
            ConvTranspose2d(num_features, num_filters, kernel_size, stride=2),
            ANFNorm(num_filters, inverse=True, mode=gdn_mode),
        )
        self.sft1 = SFT(num_filters, num_filters, num_hiddens=64, kernel_size=3)

        self.conv2 = nn.Sequential(
            ConvTranspose2d(num_filters, num_filters, kernel_size, stride=2),
            ANFNorm(num_filters, inverse=True, mode=gdn_mode)
        )
        self.sft2 = SFT(num_filters, num_filters, num_hiddens=64, kernel_size=3)

        self.conv3 = nn.Sequential(
            ConvTranspose2d(num_filters, num_filters, kernel_size, stride=2),
            ANFNorm(num_filters, inverse=True, mode=gdn_mode)
        )
        self.sft3 = SFT(num_filters, num_filters, num_hiddens=64, kernel_size=3)

        self.conv4 = ConvTranspose2d(num_filters, out_channels *
                      (2 if use_code else 1), kernel_size, stride=2)

        if share_cond_net:
            assert isinstance(cond_net, nn.Module), "cond_net should be given"
            self.feat_net = cond_net
        else:
            self.feat_net = FeatureExtractor(in_channels, num_filters)

    def net_forward(self, input, sft_cond):
        feat1, feat2, feat3, feat4 = self.feat_net(sft_cond)
        x = self.attention(input, feat4)

        x = self.conv1(x)
        x = self.sft1(x, feat3)
        
        x = self.conv2(x)
        x = self.sft2(x, feat2)
        
        x = self.conv3(x)
        x = self.sft3(x, feat1)

        x = self.conv4(x)

        return x


class SFTCondAugmentedNormalizedFlowHyperPriorCoder(HyperPriorCoder):
    """SFTCondAugmentedNormalizedFlowHyperPriorCoder"""

    def __init__(self, num_filters, num_features, num_hyperpriors, gdn_mode="standard",
                 in_channels=3, out_channels=3, kernel_size=5, num_layers=1, # Note: out_channels is useless
                 init_code='gaussian', use_DQ=False, share_wei=False, use_code=True, dec_add=False,
                 use_attn=False, use_QE=False,
                 hyper_filters=192, use_mean=False, use_context=False,
                 condition='Gaussian', quant_mode='noise',
                 output_nought=True, # Set False when applying ANFIC for residual coding, which will set output(x_2) as MC frame
                 num_cond_frames:int =1 # Set 1 when only MC frame is for condition ; >1 whwn multi-refertence frames as conditions
                 ):
        super(SFTCondAugmentedNormalizedFlowHyperPriorCoder, self).__init__(
            num_features, num_hyperpriors, use_mean, False, use_context, condition, quant_mode)
        self.num_layers = num_layers
        self.share_wei = share_wei
        self.output_nought=output_nought
        assert num_cond_frames > 0, 'number of conditioning frames must >=1'

        print('self.output_nought = ',self.output_nought)

        if not isinstance(num_filters, list):
            num_filters = [num_filters]
        if len(num_filters) != num_layers:
            num_filters = [num_filters[0]] * num_layers

        if not share_wei:
            self.__delattr__('analysis')
            self.__delattr__('synthesis')

            for i in range(num_layers):
                self.add_module('cond_net_'+str(i), FeatureExtractor(in_channels, num_filters[i]))

                self.add_module('analysis'+str(i), SFTCondAugmentedNormalizedAnalysisTransform(
                    in_channels*(1+num_cond_frames), num_features, num_filters[i], kernel_size, 
                    use_code=use_code and init_code != 'zeros', 
                    distribution=init_code, gdn_mode=gdn_mode, 
                    use_attn=use_attn and i == num_layers-1,
                    share_cond_net=True,
                    cond_net=self['cond_net_'+str(i)]))

                self.add_module('synthesis'+str(i), SFTCondAugmentedNormalizedSynthesisTransform(
                    in_channels, num_features, num_filters[i], kernel_size, 
                    use_code=use_code and i != num_layers-1 and not dec_add, 
                    distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn and i == num_layers-1,
                    share_cond_net=True,
                    cond_net=self['cond_net_'+str(i)]))

                if use_QE:
                    self.add_module(
                        'QE'+str(i), DeQuantizationModule(in_channels, in_channels, 64, 2))
        else: 
            self.analysis = SFTCondAugmentedNormalizedAnalysisTransform(
                in_channels*(1+num_cond_frames), num_features, num_filters[0], kernel_size, 
                use_code=use_code, distribution=init_code, 
                gdn_mode=gdn_mode, use_attn=use_attn,
                share_cond_net=True,
                cond_net=self['cond_net_'+str(i)])
            self.synthesis = SFTCondAugmentedNormalizedSynthesisTransform(
                in_channels, num_features, num_filters[0], kernel_size, 
                use_code=use_code, distribution=init_code, 
                gdn_mode=gdn_mode, use_attn=use_attn,
                share_cond_net=True,
                cond_net=self['cond_net_'+str(i)])

            if use_QE:
                for i in range(num_layers):
                    self.add_module(
                        'QE'+str(i), DeQuantizationModule(in_channels, in_channels, 64, 2))

        self.hyper_analysis = GoogleHyperAnalysisTransform(
            num_features, hyper_filters, num_hyperpriors) if not use_attn else GoogleHyperAnalysisTransform2(num_features, hyper_filters, num_hyperpriors)

        if use_context:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features*2, hyper_filters, num_hyperpriors) if not use_attn else GoogleHyperSynthesisTransform2(num_features*2, hyper_filters, num_hyperpriors)
        elif self.use_mean or "Mixture" in condition:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features*self.conditional_bottleneck.condition_size, hyper_filters, num_hyperpriors) if not use_attn else GoogleHyperSynthesisTransform2(num_features*self.conditional_bottleneck.condition_size, hyper_filters, num_hyperpriors)
        else:
            self.hyper_synthesis = GoogleHyperScaleSynthesisTransform(
                num_features, hyper_filters, num_hyperpriors)

        self.use_QE = use_QE

        for name, m in self.named_children():
            if "ana" in name or "syn" in name:
                m.name = name

        if use_DQ:
            self.DQ = DeQuantizationModule(in_channels, in_channels, 64, 6)
        else:
            self.DQ = None

    def __getitem__(self, key):
        return self.__getattr__(key[:-1] if self.share_wei else key)

    def encode(self, input, code=None, jac=None, visual=False, figname='', cond_coupling_input=None):
        for i in range(self.num_layers):
            debug('F', i)
            # condition 
            cond = cond_coupling_input
            _, code, jac = self['analysis'+str(i)](
                input, cond, code, jac, layer=i, visual=visual, figname=figname+"_"+str(i))

            if i < self.num_layers-1:
                debug('S', i)
                input, _, jac = self['synthesis'+str(i)](
                    input, cond, code, jac, layer=i, visual=visual, figname=figname+"_"+str(i))

            if visual:
                logger.write(check_range(input, "X"+str(i)))
                logger.write(check_range(code, "code"+str(i)))
                logger.write()

                channel_analysis(code, figname+f"_FCA_{i}.png")

        return input, code, jac

    def decode(self, input, code=None, jac=None, rec_code=False, visual=False, figname='', cond_coupling_input=None):
        for i in range(self.num_layers-1, -1, -1):
            debug('rF', i)
            # condition
            cond = cond_coupling_input
            input, _, jac = self['synthesis'+str(i)](
                input, cond, code, jac, rev=True, last_layer=i == self.num_layers-1, layer=i, visual=visual, figname=figname+"_rev_"+str(i))

            if self.use_QE:
                BQE = input
                input = self['QE'+str(i)](input)
                if visual:
                    save_image(BQE, figname+f"_BQE_{i}.png")
                    save_image(input, figname+f"_AQE_{i}.png")

            if i or rec_code or jac is not None:
                debug('RF', i)
                # condition
                cond = cond_coupling_input
                _, code, jac = self['analysis'+str(i)](
                    input, cond, code, jac, layer=i, rev=True, visual=visual, figname=figname+"_rev_"+str(i))
               
            if visual:
                logger.write(check_range(input, "X"+str(i)))
                logger.write(check_range(code, "code"+str(i)))
                logger.write()

        return input, code, jac

    def entropy_model(self, input, code, jac=False):
        # Enrtopy coding
        debug('E')

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition)
        # y_tilde = code

        # Encode distortion
        Y_error, _, jac = self['synthesis'+str(self.num_layers-1)](
            input, y_tilde, jac, last_layer=True, layer=self.num_layers-1)

        return Y_error, y_tilde, z_tilde, y_likelihood, z_likelihood

    def compress(self, input, cond_coupling_input=None, return_hat=False, reverse_input=None): #TODO
        assert not (cond_coupling_input is None), "cond_coupling_input should be specified"
        
        code = None
        jac = None
        input, features, jac = self.encode(
            input, code, jac, visual=visual, figname=figname, cond_coupling_input=cond_coupling_input)

        hyperpriors = self.hyper_analysis(
            features.abs() if self.use_abs else features)

        side_stream, z_hat = self.entropy_bottleneck.compress(
            hyperpriors, return_sym=True)

        condition = self.hyper_synthesis(z_hat)

        ret = self.conditional_bottleneck.compress(
            features, condition=condition, return_sym=return_hat)

        if return_hat:
            jac = None
            stream, y_hat = ret

            # Decode
            if not self.output_nought:
                assert not (reverse_input is None), "reverse_input should be specified"
                input = reverse_input
            else:
                input = torch.zeros_like(input)
                
            x_hat, code, jac = self.decode(
                input, y_hat, jac, cond_coupling_input=cond_coupling_input)
            if self.DQ is not None:
                x_hat = self.DQ(x_hat)

            return x_hat, [stream, side_stream], [hyperpriors.size()]
        else:
            stream = ret
            return [stream, side_stream], [hyperpriors.size()]

    def decompress(self, strings, shape): #TODO
        stream, side_stream = strings
        z_shape = shape

        z_hat = self.entropy_bottleneck.decompress(side_stream, z_shape)

        condition = self.hyper_synthesis(z_hat)

        y_hat = self.conditional_bottleneck.decompress(
            stream, condition.size(), condition=condition)

        reconstructed = self.synthesis(y_hat)

        return reconstructed

    def forward(self, input, code=None, jac=None, rec_code=False, rev_ng=False, 
                IDQ=False, detach_enc=False, visual=False, figname='',
                output=None, # Should assign value when self.output_nought==False
                cond_coupling_input=None # Should assign value
                                         # When using ANFIC on residual coding, output & cond_coupling_input should both be MC frame
               ):
        if not self.output_nought:
            assert not (output is None), "output should be specified"
        
        assert not (cond_coupling_input is None), "cond_coupling_input should be specified"

        # Encode
        if IDQ:
            debug("IDQ")
            input = input.add(torch.rand_like(input).sub_(0.5).div_(255.))

        ori_input = input

        if visual:
            save_image(input, figname+"_input.png")
            if not (output is None):
                save_image(output, figname+"_mc_frame.png")
            visualizer.set_mean_var(input)
            visualizer.queue_visual(input, figname+"_ori.png")
            fft_visual(input, figname+"_ori.png")
            logger.open(figname+"_stati.txt")
            logger.write("min\tmedian\tmean\tvar\tmax\n")

        jac = [] if jac else None
        input, code, jac = self.encode(
            input, code, jac, visual=visual, figname=figname, cond_coupling_input=cond_coupling_input)

        # Enrtopy coding
        debug('E')

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition = self.hyper_synthesis(z_tilde)
        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition)
        # y_tilde = code # No quantize on z2

        # Encode distortion
        x_2, _, jac = self['synthesis'+str(self.num_layers-1)](
            input, cond_coupling_input, y_tilde, jac, last_layer=True, layer=self.num_layers-1, visual=visual, figname=figname+"_"+str(self.num_layers-1))

        if visual:
            visualizer.plot_queue(figname+'_for.png', nrow=input.size(0))

            with torch.no_grad(): # Visualize z likelihood
                YLL_perC = y_likelihood[0].clamp_min(
                    1e-9).log2().mean(dim=(1, 2)).neg()
                fig = plt.figure()
                check_range(YLL_perC, "YLL")
                max_idx = YLL_perC.max(0)[1]
                plt.plot(YLL_perC.data.cpu(),
                         label="maxC="+str(max_idx.item()))
                plt.legend()

                plt.savefig(figname+"_YLL.png")
                plt.close(fig)
            
            # Print feature ; code: z2, no quant ; y_tilde: \hat z2
            #     nrow = 8
            #     save_image(code.div(255*2).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_feature.png", nrow=nrow)
            #     save_image(max_norm(code).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_feature_norm.png", nrow=nrow)
            #     save_image(y_tilde.div(255*2).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_quant.png", nrow=nrow)
            #     save_image(max_norm(y_tilde).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_quant_norm.png", nrow=nrow)

            #logger.write(check_range(Y_error, "X"+str(self.num_layers-1)))
            logger.write(check_range(x_2, "X"+str(self.num_layers-1)))
            logger.write(check_range(code, "code"+str(self.num_layers-1)))
            logger.write()

        if self.entropy_bottleneck.quant_mode == "SGA" and self.training:
            tau = self.entropy_bottleneck.SGA.tau.item()*0.01
            #input, code, hyper_code = Y_error*tau, y_tilde, z_tilde
            input, code, hyper_code = x_2*tau, y_tilde, z_tilde
        else:
            #input, code, hyper_code = torch.zeros_like(input), y_tilde, z_tilde
            #input, code, hyper_code = x_2, y_tilde, z_tilde # x_2 directly backward
            input, code, hyper_code = output, y_tilde, z_tilde # Correct setting ; MC frame as x_2 when decoding

        # input = Y_error
        #debug(Y_error.shape, code.shape, hyper_code.shape)
        debug(x_2.shape, code.shape, hyper_code.shape)

        if visual:
            logger.write(check_range(input, "X"+str(self.num_layers-1)))
            logger.write(check_range(code, "code"+str(self.num_layers-1)))
            logger.write()

        # Decode
        input, code, jac = self.decode(
            input, code, jac, rec_code=rec_code, visual=visual, figname=figname, cond_coupling_input=cond_coupling_input)

        if visual:
            visualizer.plot_queue(figname+'_rev.png', nrow=input.size(0))
            logger.close()

        # Quality enhancement
        if self.DQ is not None:
            debug("DQ")
            BDQ = input
            input = self.DQ(input)
            if visual:
                save_image(x_2, figname+"_x_2.png")
                save_image(BDQ, figname+"_BDQ.png")
                save_image(input, figname+"_ADQ.png")
        else:
            BDQ = None

        debug('END\n')

        return input, (y_likelihood, z_likelihood), x_2, jac, code, BDQ

# ---------------------------------------------- End SFTCondANFCoder -------------------------------------------- #



# ############################################### Models that takes YUV as I/O ############################################
class AnalysisTransformY(nn.Sequential):
    def __init__(self, kernel_size, simplify_gdn=False):
        super(AnalysisTransformY, self).__init__(
              Conv2d(1, 4, kernel_size, stride=2),
              GeneralizedDivisiveNorm(4, simplify=simplify_gdn)
            )

class SynthesisTransformY(nn.Sequential):
    def __init__(self, kernel_size, simplify_gdn=False):
        super(SynthesisTransformY, self).__init__(
            GeneralizedDivisiveNorm(
                4, inverse=True, simplify=simplify_gdn),
            ConvTranspose2d(4, 1, kernel_size, stride=2),
        )


class GoogleAnalysisTransformYUV(nn.Sequential):
    def __init__(self, in_channels, num_features, num_filters, kernel_size, simplify_gdn):
        super(GoogleAnalysisTransformYUV, self).__init__(
            Conv2d(in_channels, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, simplify=simplify_gdn),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, simplify=simplify_gdn),
            Conv2d(num_filters, num_features, kernel_size, stride=2)
        )


class GoogleSynthesisTransformYUV(nn.Sequential):
    def __init__(self, out_channels, num_features, num_filters, kernel_size, simplify_gdn):
        super(GoogleSynthesisTransformYUV, self).__init__(
            ConvTranspose2d(num_features, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(
                num_filters, inverse=True, simplify=simplify_gdn),
            ConvTranspose2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(
                num_filters, inverse=True, simplify=simplify_gdn),
            ConvTranspose2d(num_filters, out_channels, kernel_size, stride=2)
        )


class AugmentedNormalizedAnalysisTransformYUV(AugmentedNormalizedFlow):
    def __init__(self, in_channels, num_features, num_filters, kernel_size, use_code, distribution, gdn_mode, use_attn=False, integerlize=False):
        super(AugmentedNormalizedAnalysisTransformYUV, self).__init__(
            Conv2d(in_channels, num_filters, kernel_size, stride=2),
            ANFNorm(num_filters, mode=gdn_mode),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            ANFNorm(num_filters, mode=gdn_mode),
            Conv2d(num_filters, num_features *
                   (2 if use_code else 1), kernel_size, stride=2),
            AttentionBlock(num_features *
                           (2 if use_code else 1), non_local=True) if use_attn else nn.Identity(),
            use_code=use_code, transpose=False, distribution=distribution, integerlize=integerlize
        )


class AugmentedNormalizedSynthesisTransformYUV(AugmentedNormalizedFlow):
    def __init__(self, out_channels, num_features, num_filters, kernel_size, use_code, distribution, gdn_mode, use_attn=False, integerlize=False):
        super(AugmentedNormalizedSynthesisTransformYUV, self).__init__(
            AttentionBlock(
                num_features, non_local=True) if use_attn else nn.Identity(),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size, stride=2),
            ANFNorm(num_filters, inverse=True, mode=gdn_mode),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size, stride=2),
            ANFNorm(num_filters, inverse=True, mode=gdn_mode),
            ConvTranspose2d(num_filters, out_channels *
                            (2 if use_code else 1), kernel_size, stride=2),
            use_code=use_code, transpose=True, distribution=distribution, integerlize=integerlize
        )

class GoogleHyperPriorCoderYUV(HyperPriorCoder):
    """GoogleHyperPriorCodeYUV"""

    def __init__(self, num_filters, num_features, num_hyperpriors, simplify_gdn=False,
                 in_channels=3, out_channels=3, kernel_size=5, use_mean=False, use_abs=False, use_context=False,
                 condition='Gaussian', quant_mode='noise'):
        super(GoogleHyperPriorCoderYUV, self).__init__(
            num_features, num_hyperpriors, use_mean, use_abs, use_context, condition, quant_mode)

        self.analysis = GoogleAnalysisTransformYUV(
            in_channels, num_features, num_filters, kernel_size, simplify_gdn)

        self.synthesis = GoogleSynthesisTransformYUV(
            out_channels, num_features, num_filters, kernel_size, simplify_gdn)

        self.hyper_analysis = GoogleHyperAnalysisTransform(
            num_features, num_filters, num_hyperpriors)

        if self.use_mean:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features*self.conditional_bottleneck.condition_size, num_filters, num_hyperpriors)
        else:
            self.hyper_synthesis = GoogleHyperScaleSynthesisTransform(
                num_features, num_filters, num_hyperpriors)


class CondAugmentedNormalizedFlowHyperPriorCoderYUV(HyperPriorCoder):
    """CondAugmentedNormalizedFlowHyperPriorCoderYUV ; 
       takes YUV420 as in & output
    """

    def __init__(self, num_filters, num_features, num_hyperpriors, gdn_mode="standard",
                 in_channels=6, out_channels=6, kernel_size=5, num_layers=1, # Note: out_channels is useless
                 init_code='gaussian', use_DQ=False, share_wei=False, use_code=True, dec_add=False,
                 use_attn=False, use_QE=False,
                 hyper_filters=192, use_mean=False, use_context=False,
                 condition='Gaussian', quant_mode='noise',
                 output_nought=True, # Set False when applying ANFIC for residual coding, which will set output(x_2) as MC frame
                 cond_coupling=False, #Set True when applying ANFIC for residual coding, which will take MC frame as condition
                 num_cond_frames:int =1 # Set 1 when only MC frame is for condition ; >1 whwn multi-refertence frames as conditions
                 ):
        super(CondAugmentedNormalizedFlowHyperPriorCoderYUV, self).__init__(
            num_features, num_hyperpriors, use_mean, False, use_context, condition, quant_mode)
        self.num_layers = num_layers
        self.share_wei = share_wei
        self.output_nought=output_nought
        self.cond_coupling = cond_coupling
        assert num_cond_frames > 0, 'number of conditioning frames must >=1'

        print('self.output_nought = ',self.output_nought)
        print('self.cond_coupling = ',self.cond_coupling)

        if not isinstance(num_filters, list):
            num_filters = [num_filters]
        if len(num_filters) != num_layers:
            num_filters = [num_filters[0]] * num_layers

        if not share_wei:
            self.__delattr__('analysis')
            self.__delattr__('synthesis')

            for i in range(num_layers):
                if not self.cond_coupling:
                    self.add_module('analysis'+str(i), AugmentedNormalizedAnalysisTransformYUV(
                        in_channels, num_features, num_filters[i], kernel_size, 
                        use_code=use_code and init_code != 'zeros', 
                        distribution=init_code, gdn_mode=gdn_mode, 
                        use_attn=use_attn and i == num_layers-1))
                    self.add_module('synthesis'+str(i), AugmentedNormalizedSynthesisTransformYUV(
                        in_channels, num_features, num_filters[i], kernel_size, 
                        use_code=use_code and i != num_layers-1 and not dec_add, 
                        distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn and i == num_layers-1))
                else: # For residual coding ; make transform conditional
                    self.add_module('analysis'+str(i), AugmentedNormalizedAnalysisTransformYUV(
                        in_channels*(1+num_cond_frames), num_features, num_filters[i], kernel_size, 
                        use_code=use_code and init_code != 'zeros', 
                        distribution=init_code, gdn_mode=gdn_mode, 
                        use_attn=use_attn and i == num_layers-1))
                    # Not applying condition on synthesis
                    self.add_module('synthesis'+str(i), AugmentedNormalizedSynthesisTransformYUV(
                        in_channels, num_features, num_filters[i], kernel_size, 
                        use_code=use_code and i != num_layers-1 and not dec_add, 
                        distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn and i == num_layers-1))

                if use_QE:
                    self.add_module(
                        'QE'+str(i), DeQuantizationModule(in_channels, in_channels, 64, 2))
        else: 
            self.analysis = AugmentedNormalizedAnalysisTransformYUV(
                in_channels*(1+num_cond_frames), num_features, num_filters[0], kernel_size, use_code=use_code, distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn)
            self.synthesis = AugmentedNormalizedSynthesisTransformYUV(
                in_channels, num_features, num_filters[0], kernel_size, use_code=use_code, distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn)

            if use_QE:
                for i in range(num_layers):
                    self.add_module(
                        'QE'+str(i), DeQuantizationModule(in_channels, in_channels, 64, 2))

        self.hyper_analysis = GoogleHyperAnalysisTransform(
            num_features, hyper_filters, num_hyperpriors) if not use_attn else GoogleHyperAnalysisTransform2(num_features, hyper_filters, num_hyperpriors)

        if use_context:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features*2, hyper_filters, num_hyperpriors) if not use_attn else GoogleHyperSynthesisTransform2(num_features*2, hyper_filters, num_hyperpriors)
        elif self.use_mean or "Mixture" in condition:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features*self.conditional_bottleneck.condition_size, hyper_filters, num_hyperpriors) if not use_attn else GoogleHyperSynthesisTransform2(num_features*self.conditional_bottleneck.condition_size, hyper_filters, num_hyperpriors)
        else:
            self.hyper_synthesis = GoogleHyperScaleSynthesisTransform(
                num_features, hyper_filters, num_hyperpriors)

        self.use_QE = use_QE

        for name, m in self.named_children():
            if "ana" in name or "syn" in name:
                m.name = name

        if use_DQ:
            self.DQ = DeQuantizationModule(in_channels, in_channels, 64, 6)
        else:
            self.DQ = None

    def __getitem__(self, key):
        return self.__getattr__(key[:-1] if self.share_wei else key)

    def encode(self, input, code=None, jac=None, visual=False, figname='', cond_coupling_input=None):
        for i in range(self.num_layers):
            debug('F', i)

            # Concat input with condition (MC frame)
            if self.cond_coupling:
                cond = cond_coupling_input
                cond_input = torch.cat([input, cond], dim=1)
                _, code, jac = self['analysis'+str(i)](
                    cond_input, code, jac, layer=i, visual=visual, figname=figname+"_"+str(i))
            else:
                _, code, jac = self['analysis'+str(i)](
                    input, code, jac, layer=i, visual=visual, figname=figname+"_"+str(i))

            if i < self.num_layers-1:
                debug('S', i)
                input, _, jac = self['synthesis'+str(i)](
                    input, code, jac, layer=i, visual=visual, figname=figname+"_"+str(i))

            if visual:
                logger.write(check_range(input, "X"+str(i)))
                logger.write(check_range(code, "code"+str(i)))
                logger.write()

                channel_analysis(code, figname+f"_FCA_{i}.png")

        return input, code, jac

    def decode(self, input, code=None, jac=None, rec_code=False, visual=False, figname='', cond_coupling_input=None):
        for i in range(self.num_layers-1, -1, -1):
            debug('rF', i)
            
            input, _, jac = self['synthesis'+str(i)](
                input, code, jac, rev=True, last_layer=i == self.num_layers-1, layer=i, visual=visual, figname=figname+"_rev_"+str(i))

            if self.use_QE:
                BQE = input
                input = self['QE'+str(i)](input)
                if visual:
                    save_image(YUV4202RGB(BQE), figname+f"_BQE_{i}.png")
                    save_image(YUV4202RGB(input), figname+f"_AQE_{i}.png")

            if i or rec_code or jac is not None:
                debug('RF', i)
                # Concat input with condition (MC frame)
                if self.cond_coupling:
                    cond = cond_coupling_input
                    cond_input = torch.cat([input, cond], dim=1)
                    _, code, jac = self['analysis'+str(i)](
                        cond_input, code, jac, layer=i, rev=True, visual=visual, figname=figname+"_rev_"+str(i))
                else:
                    _, code, jac = self['analysis'+str(i)](
                        input, code, jac, layer=i, rev=True, visual=visual, figname=figname+"_rev_"+str(i))

            if visual:
                logger.write(check_range(input, "X"+str(i)))
                logger.write(check_range(code, "code"+str(i)))
                logger.write()

        return input, code, jac

    def entropy_model(self, input, code, jac=False):
        # Enrtopy coding
        debug('E')

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition)
        # y_tilde = code

        # Encode distortion
        Y_error, _, jac = self['synthesis'+str(self.num_layers-1)](
            input, y_tilde, jac, last_layer=True, layer=self.num_layers-1)

        return Y_error, y_tilde, z_tilde, y_likelihood, z_likelihood

    def compress(self, input, cond_coupling_input=None, return_hat=False, reverse_input=None):
        if self.cond_coupling:
            assert not (cond_coupling_input is None), "cond_coupling_input should be specified"
        
        code = None
        jac = None
        input, features, jac = self.encode(
            input, code, jac, visual=visual, figname=figname, cond_coupling_input=cond_coupling_input)

        hyperpriors = self.hyper_analysis(
            features.abs() if self.use_abs else features)

        side_stream, z_hat = self.entropy_bottleneck.compress(
            hyperpriors, return_sym=True)

        condition = self.hyper_synthesis(z_hat)

        ret = self.conditional_bottleneck.compress(
            features, condition=condition, return_sym=return_hat)

        if return_hat:
            jac = None
            stream, y_hat = ret

            # Decode
            if not self.output_nought:
                assert not (reverse_input is None), "reverse_input should be specified"
                input = reverse_input
            else:
                input = torch.zeros_like(input)
                
            x_hat, code, jac = self.decode(
                input, y_hat, jac, cond_coupling_input=cond_coupling_input)
            if self.DQ is not None:
                x_hat = self.DQ(x_hat)

            return x_hat, [stream, side_stream], [hyperpriors.size()]
        else:
            stream = ret
            return [stream, side_stream], [hyperpriors.size()]

    def decompress(self, strings, shape): #TODO
        stream, side_stream = strings
        z_shape = shape

        z_hat = self.entropy_bottleneck.decompress(side_stream, z_shape)

        condition = self.hyper_synthesis(z_hat)

        y_hat = self.conditional_bottleneck.decompress(
            stream, condition.size(), condition=condition)

        reconstructed = self.synthesis(y_hat)

        return reconstructed

    def forward(self, input, code=None, jac=None, rec_code=False, rev_ng=False, 
                IDQ=False, detach_enc=False, visual=False, figname='',
                output=None, # Should assign value when self.output_nought==False
                cond_coupling_input=None # Should assign value when self.cond_coupling==True
                                         # When using ANFIC on residual coding, output & cond_coupling_input should both be MC frame
               ):
        if not self.output_nought:
            assert not (output is None), "output should be specified"
        if self.cond_coupling:
            assert not (cond_coupling_input is None), "cond_coupling_input should be specified"

        # Encode
        if IDQ:
            debug("IDQ")
            input = input.add(torch.rand_like(input).sub_(0.5).div_(255.))

        ori_input = input

        if visual:
            save_image(YUV4202RGB(input), figname+"_input.png")
            if not (output is None):
                save_image(YUV4202RGB(output), figname+"_mc_frame.png")
            visualizer.set_mean_var(input)
            visualizer.queue_visual(input, figname+"_ori.png")
            visualizer_yuv.set_mean_var(input)
            visualizer_yuv.queue_visual(input, figname+"_ori_yuv.png")
            fft_visual(input, figname+"_ori.png")
            logger.open(figname+"_stati.txt")
            logger.write("min\tmedian\tmean\tvar\tmax\n")

        jac = [] if jac else None
        input, code, jac = self.encode(
            input, code, jac, visual=visual, figname=figname, cond_coupling_input=cond_coupling_input)

        # Enrtopy coding
        debug('E')

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition = self.hyper_synthesis(z_tilde)
        
        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition)
        # y_tilde = code # No quantize on z2

        # Encode distortion
        x_2, _, jac = self['synthesis'+str(self.num_layers-1)](
            input, y_tilde, jac, last_layer=True, layer=self.num_layers-1, visual=visual, figname=figname+"_"+str(self.num_layers-1))

        if visual:
            visualizer.plot_queue(figname+'_for.png', nrow=input.size(0))
            visualizer_yuv.plot_queue(figname+'_for_yuv.png', nrow=input.size(0))

            with torch.no_grad(): # Visualize z likelihood
                YLL_perC = y_likelihood[0].clamp_min(
                    1e-9).log2().mean(dim=(1, 2)).neg()
                fig = plt.figure()
                check_range(YLL_perC, "YLL")
                max_idx = YLL_perC.max(0)[1]
                plt.plot(YLL_perC.data.cpu(),
                         label="maxC="+str(max_idx.item()))
                plt.legend()

                plt.savefig(figname+"_YLL.png")
                plt.close(fig)
            
            # Print feature ; code: z2, no quant ; y_tilde: \hat z2
            #     nrow = 8
            #     save_image(YUV4202RGB(code).div(255*2).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_feature.png", nrow=nrow)
            #     save_image(max_norm(YUV4202RGB(code)).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_feature_norm.png", nrow=nrow)
            #     save_image(YUV4202RGB(y_tilde).div(255*2).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_quant.png", nrow=nrow)
            #     save_image(max_norm(YUV4202RGB(y_tilde)).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_quant_norm.png", nrow=nrow)

            #logger.write(check_range(Y_error, "X"+str(self.num_layers-1)))
            logger.write(check_range(x_2, "X"+str(self.num_layers-1)))
            logger.write(check_range(code, "code"+str(self.num_layers-1)))
            logger.write()

        if self.entropy_bottleneck.quant_mode == "SGA" and self.training:
            tau = self.entropy_bottleneck.SGA.tau.item()*0.01
            #input, code, hyper_code = Y_error*tau, y_tilde, z_tilde
            input, code, hyper_code = x_2*tau, y_tilde, z_tilde
        else:
            #input, code, hyper_code = torch.zeros_like(input), y_tilde, z_tilde
            #input, code, hyper_code = x_2, y_tilde, z_tilde # x_2 directly backward ; Check reversibility
            input, code, hyper_code = output, y_tilde, z_tilde # Correct setting ; MC frame as x_2 when decoding

        # input = Y_error
        #debug(Y_error.shape, code.shape, hyper_code.shape)
        debug(x_2.shape, code.shape, hyper_code.shape)

        if visual:
            logger.write(check_range(input, "X"+str(self.num_layers-1)))
            logger.write(check_range(code, "code"+str(self.num_layers-1)))
            logger.write()
        # Decode
        input, code, jac = self.decode(
            input, code, jac, rec_code=rec_code, visual=visual, figname=figname, cond_coupling_input=cond_coupling_input)

        if visual:
            visualizer.plot_queue(figname+'_rev.png', nrow=input.size(0))
            visualizer_yuv.plot_queue(figname+'_rev_yuv.png', nrow=input.size(0))
            logger.close()

        # Quality enhancement
        if self.DQ is not None:
            debug("DQ")
            BDQ = input
            input = self.DQ(input)
            if visual:
                save_image(YUV4202RGB(x_2), figname+"_x_2.png")
                save_image(YUV4202RGB(BDQ), figname+"_BDQ.png")
                save_image(YUV4202RGB(input), figname+"_ADQ.png")
        else:
            BDQ = None

        debug('END\n')

        return input, (y_likelihood, z_likelihood), x_2, jac, code, BDQ

""


class CondAugmentedNormalizedFlowHyperPriorCoderPredPriorYUV(CondAugmentedNormalizedFlowHyperPriorCoderYUV):
    def __init__(self, in_channels_predprior=3, num_predprior_filters=None, **kwargs):
        super(CondAugmentedNormalizedFlowHyperPriorCoderPredPriorYUV, self).__init__(**kwargs)

        if num_predprior_filters is None:  # When not specifying, it will align to num_filters
            num_predprior_filters = kwargs['num_filters']

        if self.use_mean or "Mixture" in kwargs["condition"]:
            self.pred_prior = GoogleAnalysisTransformYUV(in_channels_predprior,
                                                      kwargs[
                                                          'num_features'] * self.conditional_bottleneck.condition_size,
                                                      num_predprior_filters,  # num_filters=64,
                                                      kwargs['kernel_size'],  # kernel_size=3,
                                                      simplify_gdn=False)
            self.PA = nn.Sequential(
                nn.Conv2d((kwargs['num_features'] * self.conditional_bottleneck.condition_size) * 2, 640, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(640, 640, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(640, kwargs['num_features'] * self.conditional_bottleneck.condition_size, 1)
            )
        else:
            self.pred_prior = GoogleAnalysisTransformYUV(in_channels_predprior,
                                                      kwargs['num_features'],
                                                      num_predprior_filters,  # num_filters=64,
                                                      kwargs['kernel_size'],  # kernel_size=3,
                                                      simplify_gdn=False)
            self.PA = nn.Sequential(
                nn.Conv2d(kwargs['num_features'] * 2, 640, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(640, 640, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(640, kwargs['num_features'], 1)
            )

    def forward(self, input, code=None, jac=None, rec_code=False, rev_ng=False,
                IDQ=False, detach_enc=False, visual=False, figname='',
                output=None,  # Should assign value when self.output_nought==False
                cond_coupling_input=None,  # Should assign value when self.cond_coupling==True
                pred_prior_input=None  # cond_coupling_input will replace this when None
                ):

        if not self.output_nought:
            assert not (output is None), "output should be specified"
        if self.cond_coupling:
            assert not (cond_coupling_input is None), "cond_coupling_input should be specified"
        if pred_prior_input is None:
            pred_prior_input = cond_coupling_input

        # Encode
        if IDQ:
            debug("IDQ")
            input = input.add(torch.rand_like(input).sub_(0.5).div_(255.))

        ori_input = input

        if visual:
            save_image(YUV4202RGB(input), figname + "_input.png")
            if not (output is None):
                save_image(YUV4202RGB(output), figname + "_mc_frame.png")
            visualizer.set_mean_var(input)
            visualizer.queue_visual(input, figname + "_ori.png")
            visualizer_yuv.set_mean_var(input)
            visualizer_yuv.queue_visual(input, figname + "_ori_yuv.png")
            fft_visual(input, figname + "_ori.png")
            logger.open(figname + "_stati.txt")
            logger.write("min\tmedian\tmean\tvar\tmax\n")

        jac = [] if jac else None
        input, code, jac = self.encode(
            input, code, jac, visual=visual, figname=figname, cond_coupling_input=cond_coupling_input)

        # Enrtopy coding
        debug('E')

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        hp_feat = self.hyper_synthesis(z_tilde)
        pred_feat = self.pred_prior(pred_prior_input)

        condition = self.PA(torch.cat([hp_feat, pred_feat], dim=1))

        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition)

        # y_tilde = code # No quantize on z2

        # Encode distortion
        x_2, _, jac = self['synthesis' + str(self.num_layers - 1)](
            input, y_tilde, jac, last_layer=True, layer=self.num_layers - 1, visual=visual,
            figname=figname + "_" + str(self.num_layers - 1))

        if visual:
            visualizer.plot_queue(figname + '_for.png', nrow=input.size(0))
            visualizer_yuv.plot_queue(figname + '_for_yuv.png', nrow=input.size(0))

            with torch.no_grad():  # Visualize z likelihood
                YLL_perC = y_likelihood[0].clamp_min(
                    1e-9).log2().mean(dim=(1, 2)).neg()
                fig = plt.figure()
                check_range(YLL_perC, "YLL")
                max_idx = YLL_perC.max(0)[1]
                plt.plot(YLL_perC.data.cpu(),
                         label="maxC=" + str(max_idx.item()))
                plt.legend()

                plt.savefig(figname + "_YLL.png")
                plt.close(fig)

            # Print feature ; code: z2, no quant ; y_tilde: \hat z2
            #     nrow = 8
            #     save_image(YUV4202RGB(code).div(255*2).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_feature.png", nrow=nrow)
            #     save_image(max_norm(YUV4202RGB(code)).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_feature_norm.png", nrow=nrow)
            #     save_image(YUV4202RGB(y_tilde).div(255*2).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_quant.png", nrow=nrow)
            #     save_image(max_norm(YUV4202RGB(y_tilde)).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_quant_norm.png", nrow=nrow)

            # logger.write(check_range(Y_error, "X"+str(self.num_layers-1)))
            logger.write(check_range(x_2, "X" + str(self.num_layers - 1)))
            logger.write(check_range(code, "code" + str(self.num_layers - 1)))
            logger.write()

        if self.entropy_bottleneck.quant_mode == "SGA" and self.training:
            tau = self.entropy_bottleneck.SGA.tau.item() * 0.01
            # input, code, hyper_code = Y_error*tau, y_tilde, z_tilde
            input, code, hyper_code = x_2 * tau, y_tilde, z_tilde
        else:
            # input, code, hyper_code = torch.zeros_like(input), y_tilde, z_tilde
            # input, code, hyper_code = x_2, y_tilde, z_tilde # x_2 directly backward ; Check reversibility
            input, code, hyper_code = output, y_tilde, z_tilde  # Correct setting ; MC frame as x_2 when decoding

        # input = Y_error
        # debug(Y_error.shape, code.shape, hyper_code.shape)
        debug(x_2.shape, code.shape, hyper_code.shape)

        if visual:
            logger.write(check_range(input, "X" + str(self.num_layers - 1)))
            logger.write(check_range(code, "code" + str(self.num_layers - 1)))
            logger.write()
        # Decode
        input, code, jac = self.decode(
            input, code, jac, rec_code=rec_code, visual=visual, figname=figname,
            cond_coupling_input=cond_coupling_input)

        if visual:
            visualizer.plot_queue(figname + '_rev.png', nrow=input.size(0))
            visualizer_yuv.plot_queue(figname + '_rev_yuv.png', nrow=input.size(0))
            logger.close()

        # Quality enhancement
        if self.DQ is not None:
            debug("DQ")
            BDQ = input
            input = self.DQ(input)
            if visual:
                save_image(YUV4202RGB(x_2), figname + "_x_2.png")
                save_image(YUV4202RGB(BDQ), figname + "_BDQ.png")
                save_image(YUV4202RGB(input), figname + "_ADQ.png")
        else:
            BDQ = None

        debug('END\n')

        return input, (y_likelihood, z_likelihood), x_2, jac, code, BDQ

class AugmentedNormalizedFlowHyperPriorCoder(HyperPriorCoder):
    """AugmentedNormalizedFlowHyperPriorCoder"""

    def __init__(self, num_filters, num_features, num_hyperpriors, gdn_mode="standard",
                 in_channels=3, out_channels=3, kernel_size=5, num_layers=1,
                 init_code='gaussian', use_DQ=False, share_wei=False, use_code=True, dec_add=False,
                 use_attn=False, use_QE=False,
                 hyper_filters=192, use_mean=False, use_context=False,
                 condition='Gaussian', quant_mode='noise'):
        super(AugmentedNormalizedFlowHyperPriorCoder, self).__init__(
            num_features, num_hyperpriors, use_mean, False, use_context, condition, quant_mode)
        self.num_layers = num_layers
        self.share_wei = share_wei
        if not isinstance(num_filters, list):
            num_filters = [num_filters]
        if len(num_filters) != num_layers:
            num_filters = [num_filters[0]] * num_layers

        if not share_wei:
            self.__delattr__('analysis')
            self.__delattr__('synthesis')

            for i in range(num_layers):
                self.add_module('analysis'+str(i), AugmentedNormalizedAnalysisTransform(
                    in_channels, num_features, num_filters[i], kernel_size, use_code=use_code and init_code != 'zeros', distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn and i == num_layers-1))
                self.add_module('synthesis'+str(i), AugmentedNormalizedSynthesisTransform(
                    in_channels, num_features, num_filters[i], kernel_size, use_code=use_code and i != num_layers-1 and not dec_add, distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn and i == num_layers-1))

                if use_QE:
                    self.add_module(
                        'QE'+str(i), DeQuantizationModule(in_channels, in_channels, 64, 2))
        else:
            self.analysis = AugmentedNormalizedAnalysisTransform(
                in_channels, num_features, num_filters[0], kernel_size, use_code=use_code, distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn)
            self.synthesis = AugmentedNormalizedSynthesisTransform(
                in_channels, num_features, num_filters[0], kernel_size, use_code=use_code, distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn)

            if use_QE:
                for i in range(num_layers):
                    self.add_module(
                        'QE'+str(i), DeQuantizationModule(in_channels, in_channels, 64, 2))

        self.hyper_analysis = GoogleHyperAnalysisTransform(
            num_features, hyper_filters, num_hyperpriors) if not use_attn else GoogleHyperAnalysisTransform2(num_features, hyper_filters, num_hyperpriors)

        if use_context:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features*2, hyper_filters, num_hyperpriors) if not use_attn else GoogleHyperSynthesisTransform2(num_features*2, hyper_filters, num_hyperpriors)
        elif self.use_mean or "Mixture" in condition:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features*self.conditional_bottleneck.condition_size, hyper_filters, num_hyperpriors) if not use_attn else GoogleHyperSynthesisTransform2(num_features*self.conditional_bottleneck.condition_size, hyper_filters, num_hyperpriors)
        else:
            self.hyper_synthesis = GoogleHyperScaleSynthesisTransform(
                num_features, hyper_filters, num_hyperpriors)

        self.use_QE = use_QE

        for name, m in self.named_children():
            if "ana" in name or "syn" in name:
                m.name = name

        if use_DQ:
            self.DQ = DeQuantizationModule(in_channels, in_channels, 64, 6)
        else:
            self.DQ = None

    def __getitem__(self, key):
        return self.__getattr__(key[:-1] if self.share_wei else key)

    def encode(self, input, code=None, jac=None, visual=False, figname=''):
        for i in range(self.num_layers):
            debug('F', i)
            _, code, jac = self['analysis'+str(i)](
                input, code, jac, layer=i, visual=visual, figname=figname+"_"+str(i))

            if i < self.num_layers-1:
                debug('S', i)
                input, _, jac = self['synthesis'+str(i)](
                    input, code, jac, layer=i, visual=visual, figname=figname+"_"+str(i))

            if visual:
                logger.write(check_range(input, "X"+str(i)))
                logger.write(check_range(code, "code"+str(i)))
                logger.write()

                channel_analysis(code, figname+f"_FCA_{i}.png")

        return input, code, jac

    def decode(self, input, code=None, jac=None, rec_code=False, visual=False, figname=''):
        for i in range(self.num_layers-1, -1, -1):
            debug('rF', i)
            input, _, jac = self['synthesis'+str(i)](
                input, code, jac, rev=True, last_layer=i == self.num_layers-1, layer=i, visual=visual, figname=figname+"_rev_"+str(i))

            if self.use_QE:
                BQE = input
                input = self['QE'+str(i)](input)
                if visual:
                    save_image(BQE, figname+f"_BQE_{i}.png")
                    save_image(input, figname+f"_AQE_{i}.png")

            if i or rec_code or jac is not None:
                debug('RF', i)
                _, code, jac = self['analysis' +
                                    str(i)](input, code, jac, rev=True, layer=i, visual=visual, figname=figname+"_rev_"+str(i))

            if visual:
                logger.write(check_range(input, "X"+str(i)))
                logger.write(check_range(code, "code"+str(i)))
                logger.write()

        return input, code, jac

    def entropy_model(self, input, code, jac=False):
        # Enrtopy coding
        debug('E')

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition)
        # y_tilde = code

        # Encode distortion
        Y_error, _, jac = self['synthesis'+str(self.num_layers-1)](
            input, y_tilde, jac, last_layer=True, layer=self.num_layers-1)

        return Y_error, y_tilde, z_tilde, y_likelihood, z_likelihood

    def forward(self, input, code=None, jac=None, rec_code=False, rev_ng=False, IDQ=False, detach_enc=False, visual=False, figname=''):
        # Encode
        if IDQ:
            debug("IDQ")
            input = input.add(torch.rand_like(input).sub_(0.5).div_(255.))

        ori_input = input

        if visual:
            visualizer.set_mean_var(input)
            visualizer.queue_visual(input, figname+"_ori.png")
            fft_visual(input, figname+"_ori.png")
            logger.open(figname+"_stati.txt")

        jac = [] if jac else None
        input, code, jac = self.encode(
            input, code, jac, visual=visual, figname=figname)

        # Enrtopy coding
        debug('E')

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition)
        # y_tilde = code

        # Encode distortion
        Y_error, _, jac = self['synthesis'+str(self.num_layers-1)](
            input, y_tilde, jac, last_layer=True, layer=self.num_layers-1, visual=visual, figname=figname+"_"+str(self.num_layers-1))

        if visual:
            visualizer.plot_queue(figname+'_for.png', nrow=input.size(0))

            with torch.no_grad():
                YLL_perC = y_likelihood[0].clamp_min(
                    1e-9).log2().mean(dim=(1, 2)).neg()
                fig = plt.figure()
                check_range(YLL_perC, "YLL")
                max_idx = YLL_perC.max(0)[1]
                plt.plot(YLL_perC.data.cpu(),
                         label="maxC="+str(max_idx.item()))
                plt.legend()

                plt.savefig(figname+"_YLL.png")
                plt.close(fig)

            #     nrow = 8
            #     save_image(code.div(255*2).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_feature.png", nrow=nrow)
            #     save_image(max_norm(code).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_feature_norm.png", nrow=nrow)
            #     save_image(y_tilde.div(255*2).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_quant.png", nrow=nrow)
            #     save_image(max_norm(y_tilde).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_quant_norm.png", nrow=nrow)

            logger.write(check_range(Y_error, "X"+str(self.num_layers-1)))
            logger.write(check_range(code, "code"+str(self.num_layers-1)))
            logger.write()

        if self.entropy_bottleneck.quant_mode == "SGA" and self.training:
            tau = self.entropy_bottleneck.SGA.tau.item()*0.01
            input, code, hyper_code = Y_error*tau, y_tilde, z_tilde
        else:
            input, code, hyper_code = torch.zeros_like(input), y_tilde, z_tilde
        # input, code, hyper_code = Y_error, y_tilde, z_tilde
        # input = Y_error
        debug(Y_error.shape, code.shape, hyper_code.shape)

        if visual:
            logger.write(check_range(input, "X"+str(self.num_layers-1)))
            logger.write(check_range(code, "code"+str(self.num_layers-1)))
            logger.write()

        # Decode
        input, code, jac = self.decode(
            input, code, jac, rec_code=rec_code, visual=visual, figname=figname)

        if visual:
            visualizer.plot_queue(figname+'_rev.png', nrow=input.size(0))
            logger.close()

        if self.DQ is not None:
            debug("DQ")
            BDQ = input
            input = self.DQ(input)
            if visual:
                save_image(BDQ, figname+"_BDQ.png")
                save_image(input, figname+"_ADQ.png")
        else:
            BDQ = None

        debug('END\n')

        return input, (y_likelihood, z_likelihood), Y_error, jac, code, BDQ


class AugmentedNormalizedIntegerDiscreteFlowHyperPriorCoder(HyperPriorCoder):
    """AugmentedNormalizedFlowHyperPriorCoder"""

    def __init__(self, num_filters, num_features, num_hyperpriors, gdn_mode="standard",
                 in_channels=3, out_channels=3, kernel_size=5, num_layers=1,
                 init_code='gaussian', use_DQ=False, share_wei=False, use_code=True, dec_add=False,
                 use_attn=False, use_QE=False, integerlize=True,
                 hyper_filters=192, use_mean=False, use_context=False,
                 condition='Gaussian', quant_mode='noise'):
        super(AugmentedNormalizedIntegerDiscreteFlowHyperPriorCoder, self).__init__(
            num_features, num_hyperpriors, use_mean, False, use_context, condition, quant_mode)
        self.num_layers = num_layers
        self.share_wei = share_wei
        if not isinstance(num_filters, list):
            num_filters = [num_filters]
        if len(num_filters) != num_layers:
            num_filters = [num_filters[0]] * num_layers
        self.integerlize = integerlize

        if not share_wei:
            self.__delattr__('analysis')
            self.__delattr__('synthesis')

            for i in range(num_layers):
                self.add_module('analysis'+str(i), AugmentedNormalizedAnalysisTransform(
                    in_channels, num_features, num_filters[i], kernel_size, use_code=use_code and init_code != 'zeros', distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn and i == num_layers-1, integerlize=integerlize))
                self.add_module('synthesis'+str(i), AugmentedNormalizedSynthesisTransform(
                    in_channels, num_features, num_filters[i], kernel_size, use_code=use_code and i != num_layers-1 and not dec_add, distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn and i == num_layers-1, integerlize=integerlize))

                if use_QE:
                    self.add_module(
                        'QE'+str(i), DeQuantizationModule(in_channels, in_channels, 64, 2))
        else:
            self.analysis = AugmentedNormalizedAnalysisTransform(
                in_channels, num_features, num_filters[0], kernel_size, use_code=use_code, distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn)
            self.synthesis = AugmentedNormalizedSynthesisTransform(
                in_channels, num_features, num_filters[0], kernel_size, use_code=use_code, distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn)

            if use_QE:
                for i in range(num_layers):
                    self.add_module(
                        'QE'+str(i), DeQuantizationModule(in_channels, in_channels, 64, 2))

        self.hyper_analysis = GoogleHyperAnalysisTransform(
            num_features, hyper_filters, num_hyperpriors) if not use_attn else GoogleHyperAnalysisTransform2(num_features, hyper_filters, num_hyperpriors)

        if use_context:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features*2, hyper_filters, num_hyperpriors) if not use_attn else GoogleHyperSynthesisTransform2(num_features*2, hyper_filters, num_hyperpriors)
        elif self.use_mean or "Mixture" in condition:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features*self.conditional_bottleneck.condition_size, hyper_filters, num_hyperpriors) if not use_attn else GoogleHyperSynthesisTransform2(num_features*self.conditional_bottleneck.condition_size, hyper_filters, num_hyperpriors)
        else:
            self.hyper_synthesis = GoogleHyperScaleSynthesisTransform(
                num_features, hyper_filters, num_hyperpriors)

        self.use_QE = use_QE

        for name, m in self.named_children():
            if "ana" in name or "syn" in name:
                m.name = name

        if use_DQ:
            self.DQ = DeQuantizationModule(in_channels, in_channels, 64, 6)
        else:
            self.DQ = None

    def __getitem__(self, key):
        return self.__getattr__(key[:-1] if self.share_wei else key)

    def encode(self, input, code=None, jac=None, visual=False, figname=''):
        for i in range(self.num_layers):
            debug('F', i)
            _, code, jac = self['analysis'+str(i)](
                input, code, jac, layer=i, visual=visual, figname=figname+"_"+str(i))

            if i < self.num_layers-1:
                debug('S', i)
                input, _, jac = self['synthesis'+str(i)](
                    input, code, jac, layer=i, visual=visual, figname=figname+"_"+str(i))

            if visual:
                logger.write(check_range(input, "X"+str(i)))
                logger.write(check_range(code, "code"+str(i)))
                logger.write()

                channel_analysis(code, figname+f"_FCA_{i}.png")

        return input, code, jac

    def decode(self, input, code=None, jac=None, rec_code=False, visual=False, figname=''):
        for i in range(self.num_layers-1, -1, -1):
            debug('rF', i)
            input, _, jac = self['synthesis'+str(i)](
                input, code, jac, rev=True, last_layer=i == self.num_layers-1, layer=i, visual=visual, figname=figname+"_rev_"+str(i))

            if self.use_QE:
                BQE = input
                input = quantize(self['QE'+str(i)](input))
                if visual:
                    save_image(BQE, figname+f"_BQE_{i}.png")
                    save_image(input, figname+f"_AQE_{i}.png")

            if i or rec_code or jac is not None:
                debug('RF', i)
                _, code, jac = self['analysis' +
                                    str(i)](input, code, jac, rev=True, layer=i, visual=visual, figname=figname+"_rev_"+str(i))

            if visual:
                logger.write(check_range(input, "X"+str(i)))
                logger.write(check_range(code, "code"+str(i)))
                logger.write()

        return input, code, jac

    def entropy_model(self, input, code, jac=False):
        # Enrtopy coding
        debug('E')

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition)
        # y_tilde = code

        # Encode distortion
        Y_error, _, jac = self['synthesis'+str(self.num_layers-1)](
            input, y_tilde, jac, last_layer=True, layer=self.num_layers-1)

        return Y_error, y_tilde, z_tilde, y_likelihood, z_likelihood

    def forward(self, input, code=None, jac=None, rec_code=False, rev_ng=False, IDQ=False, detach_enc=False, visual=False, figname=''):
        # Encode
        if IDQ:
            debug("IDQ")
            input = input.add(torch.rand_like(input).sub_(0.5))

        ori_input = input

        if visual:
            # visualizer.set_mean_var(input)
            visualizer.queue_visual(input.div(255.), figname+"_ori.png")
            fft_visual(input.div(255.), figname+"_ori.png")
            logger.open(figname+"_stati.txt")

        jac = [] if jac else None
        input, code, jac = self.encode(
            input, code, jac, visual=visual, figname=figname)

        # Enrtopy coding
        debug('E')

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition)
        # y_tilde = code

        # Encode distortion
        Y_error, _, jac = self['synthesis'+str(self.num_layers-1)](
            input, y_tilde, jac, last_layer=True, layer=self.num_layers-1, visual=visual, figname=figname+"_"+str(self.num_layers-1))

        if visual:
            visualizer.plot_queue(figname+'_for.png', nrow=input.size(0))

            with torch.no_grad():
                YLL_perC = y_likelihood[0].clamp_min(
                    1e-9).log2().mean(dim=(1, 2)).neg()
                fig = plt.figure()
                check_range(YLL_perC, "YLL")
                max_idx = YLL_perC.max(0)[1]
                plt.plot(YLL_perC.data.cpu(),
                         label="maxC="+str(max_idx.item()))
                plt.legend()

                plt.savefig(figname+"_YLL.png")
                plt.close(fig)

            #     nrow = 8
            #     save_image(code.div(255*2).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_feature.png", nrow=nrow)
            #     save_image(max_norm(code).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_feature_norm.png", nrow=nrow)
            #     save_image(y_tilde.div(255*2).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_quant.png", nrow=nrow)
            #     save_image(max_norm(y_tilde).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_quant_norm.png", nrow=nrow)

            logger.write(check_range(Y_error, "X"+str(self.num_layers-1)))
            logger.write(check_range(code, "code"+str(self.num_layers-1)))
            logger.write()

        if self.entropy_bottleneck.quant_mode == "SGA" and self.training:
            tau = self.entropy_bottleneck.SGA.tau.item()*0.01
            input, code, hyper_code = Y_error*tau, y_tilde, z_tilde
        else:
            input, code, hyper_code = torch.zeros_like(input), y_tilde, z_tilde
        # input, code, hyper_code = Y_error, y_tilde, z_tilde
        # input = Y_error
        debug(Y_error.shape, code.shape, hyper_code.shape)

        if visual:
            logger.write(check_range(input, "X"+str(self.num_layers-1)))
            logger.write(check_range(code, "code"+str(self.num_layers-1)))
            logger.write()

        # Decode
        input, code, jac = self.decode(
            input, code, jac, rec_code=rec_code, visual=visual, figname=figname)

        if visual:
            visualizer.plot_queue(figname+'_rev.png', nrow=input.size(0))
            logger.close()

        if self.DQ is not None:
            debug("DQ")
            BDQ = input
            input = quantize(self.DQ(input))
            if visual:
                save_image(BDQ.div(255.), figname+"_BDQ.png")
                save_image(input.div(255.), figname+"_ADQ.png")
        else:
            BDQ = None

        debug('END\n')

        return input, (y_likelihood, z_likelihood), Y_error, jac, code, BDQ



class AugmentedNormalizedFlowHyperPriorCoder2(HyperPriorCoder):
    """AugmentedNormalizedFlowHyperPriorCoder2"""

    def __init__(self, num_filters, num_features, num_hyperpriors, gdn_mode="standard",
                 in_channels=3, out_channels=3, kernel_size=5, num_layers=1,
                 init_code='gaussian', use_DQ=False, share_wei=False, use_code=True, dec_add=False,
                 use_attn=False, input_norm="shift", use_syntax=False, use_AQ=False, syntax_prior="None",
                 hyper_filters=192, use_mean=False, use_context=False, use_conditionalconv=False, ch_wise=False,
                 condition='Gaussian', quant_mode='noise'):
        super(AugmentedNormalizedFlowHyperPriorCoder2, self).__init__(
            num_features, num_hyperpriors, use_mean, False, use_context, condition, quant_mode)
        from torch_compression.modules.context_model import ContextModel2
        self.conditional_bottleneck = ContextModel2(
            num_features, num_features*2, self.conditional_bottleneck.entropy_model, kernel_size=5)
        self.num_layers = num_layers
        self.share_wei = share_wei
        if not isinstance(num_filters, list):
            num_filters = [num_filters]
        if len(num_filters) != num_layers:
            num_filters = [num_filters[0]] * num_layers

        if not share_wei:
            self.__delattr__('analysis')
            self.__delattr__('synthesis')

            for i in range(num_layers):
                self.add_module('analysis'+str(i), AugmentedNormalizedAnalysisTransform(
                    in_channels, num_features, num_filters[i], kernel_size, use_code=use_code and init_code != 'zeros', distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn and i == num_layers-1))
                self.add_module('synthesis'+str(i), AugmentedNormalizedSynthesisTransform(
                    in_channels, num_features, num_filters[i], kernel_size, use_code=use_code and i != num_layers-1 and not dec_add, distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn and i == num_layers-1))
        else:
            self.analysis = AugmentedNormalizedAnalysisTransform(
                in_channels, num_features, num_filters[0], kernel_size, use_code=use_code, distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn)
            self.synthesis = AugmentedNormalizedSynthesisTransform(
                in_channels, num_features, num_filters[0], kernel_size, use_code=use_code, distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn)

        self.hyper_analysis = GoogleHyperAnalysisTransform(
            num_features, hyper_filters, num_hyperpriors) if not use_attn else GoogleHyperAnalysisTransform2(num_features, hyper_filters, num_hyperpriors)

        if use_context:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features*2, hyper_filters, num_hyperpriors) if not use_attn else GoogleHyperSynthesisTransform2(num_features*2, hyper_filters, num_hyperpriors)
        elif self.use_mean or "Mixture" in condition:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features*self.conditional_bottleneck.condition_size, hyper_filters, num_hyperpriors) if not use_attn else GoogleHyperSynthesisTransform2(num_features*self.conditional_bottleneck.condition_size, hyper_filters, num_hyperpriors)
        else:
            self.hyper_synthesis = GoogleHyperScaleSynthesisTransform(
                num_features, hyper_filters, num_hyperpriors)

        for name, m in self.named_children():
            if "ana" in name or "syn" in name:
                m.name = name

        if use_DQ:
            self.DQ = DeQuantizationModule(3, 3, 64, 6)
        else:
            self.DQ = None

        self.AdaptiveQuant = AdaptiveQuant2(
            hyper_filters, num_features, num_hyperpriors, use_conditionalconv, ch_wise)
        self.num_bitstreams += 1

    def __getitem__(self, key):
        return self.__getattr__(key[:-1] if self.share_wei else key)

    def encode(self, input, code=None, jac=None, visual=False, figname=''):
        for i in range(self.num_layers):
            debug('F', i)
            _, code, jac = self['analysis'+str(i)](
                input, code, jac, layer=i, visual=visual, figname=figname+"_"+str(i))

            if i < self.num_layers-1:
                debug('S', i)
                input, _, jac = self['synthesis'+str(i)](
                    input, code, jac, layer=i, visual=visual, figname=figname+"_"+str(i))

            if visual:
                logger.write(check_range(input, "X"+str(i)))
                logger.write(check_range(code, "code"+str(i)))
                logger.write()

        return input, code, jac

    def decode(self, input, code=None, jac=None, rec_code=False, visual=False, figname=''):
        for i in range(self.num_layers-1, -1, -1):
            debug('rF', i)
            input, _, jac = self['synthesis'+str(i)](
                input, code, jac, rev=True, last_layer=i == self.num_layers-1, layer=i, visual=visual, figname=figname+"_rev_"+str(i))

            if i or rec_code or jac is not None:
                debug('RF', i)
                _, code, jac = self['analysis' +
                                    str(i)](input, code, jac, rev=True, layer=i, visual=visual, figname=figname+"_rev_"+str(i))

            if visual:
                logger.write(check_range(input, "X"+str(i)))
                logger.write(check_range(code, "code"+str(i)))
                logger.write()

        return input, code, jac

    def entropy_model(self, input, code, jac=False):
        # Enrtopy coding
        debug('E')

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition)
        # y_tilde = code

        # Encode distortion
        Y_error, _, jac = self['synthesis'+str(self.num_layers-1)](
            input, y_tilde, jac, last_layer=True, layer=self.num_layers-1)

        return Y_error, y_tilde, z_tilde, y_likelihood, z_likelihood

    def forward(self, input, code=None, jac=None, rec_code=False, rev_ng=False, IDQ=False, detach_enc=False, visual=False, figname=''):
        # Encode
        ori_input = input

        if visual:
            visualizer.set_mean_var(input)
            visualizer.queue_visual(input, figname+"_ori.png")
            logger.open(figname+"_stati.txt")

        jac = [] if jac else None
        if detach_enc:
            with torch.no_grad():
                input, code, jac = self.encode(
                    input, code, jac, visual=visual, figname=figname)
        else:
            input, code, jac = self.encode(
                input, code, jac, visual=visual, figname=figname)

        # Enrtopy coding
        debug('E')

        scale_factor, s_likelihood = self.AdaptiveQuant(code)

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition, scale_factor=scale_factor)
        # y_tilde = code

        # Encode distortion
        Y_error, _, jac = self['synthesis'+str(self.num_layers-1)](
            input, y_tilde, jac, last_layer=True, layer=self.num_layers-1, visual=visual, figname=figname+"_"+str(self.num_layers-1))

        if visual:
            visualizer.plot_queue(figname+'_for.png', nrow=input.size(0))

            with torch.no_grad():
                YLL_perC = y_likelihood.clamp(
                    1e-9).log2().mean(dim=(2, 3)).neg()
                fig = plt.figure()
                check_range(YLL_perC, "YLL")
                C = YLL_perC.size(1)
                for YLL in YLL_perC:
                    max_idx = YLL.max(0)[1]
                    plt.bar(torch.arange(C), YLL.data.cpu(),
                            label="maxC="+str(max_idx.item()))
                plt.legend()
                # plt.show()
                plt.savefig(figname+"_YLL.png")
                plt.close(fig)

                nrow = 8
                save_image(code.div(255*2).add(0.5).flatten(0, 1).unsqueeze(
                    1), figname+"_feature.png", nrow=nrow)
                save_image(max_norm(code).add(0.5).flatten(0, 1).unsqueeze(
                    1), figname+"_feature_norm.png", nrow=nrow)
                save_image(y_tilde.div(255*2).add(0.5).flatten(0, 1).unsqueeze(
                    1), figname+"_quant.png", nrow=nrow)
                save_image(max_norm(y_tilde).add(0.5).flatten(0, 1).unsqueeze(
                    1), figname+"_quant_norm.png", nrow=nrow)

            logger.write(check_range(Y_error, "X"+str(self.num_layers-1)))
            logger.write(check_range(code, "code"+str(self.num_layers-1)))
            logger.write(check_range(scale_factor, "SF"))
            logger.write()

        input, code, hyper_code = torch.zeros_like(input), y_tilde, z_tilde
        # input, code, hyper_code = Y_error, y_tilde, z_tilde
        # input = Y_error
        debug(Y_error.shape, code.shape, hyper_code.shape)

        if visual:
            logger.write(check_range(input, "X"+str(self.num_layers-1)))
            logger.write(check_range(code, "code"+str(self.num_layers-1)))
            logger.write()

        # Decode
        if rev_ng:
            with torch.no_grad():
                input, code, jac = self.decode(
                    input, code, jac, rec_code=rec_code, visual=visual, figname=figname)
        else:
            input, code, jac = self.decode(
                input, code, jac, rec_code=rec_code, visual=visual, figname=figname)

        if visual:
            visualizer.plot_queue(figname+'_rev.png', nrow=input.size(0))
            logger.close()

        if self.DQ is not None:
            debug("DQ")
            BDQ = input
            input = self.DQ(input)
            if visual:
                save_image(BDQ, figname+"_BDQ.png")
                save_image(input, figname+"_ADQ.png")
        else:
            BDQ = None

        debug('END\n')

        lls = (y_likelihood, z_likelihood)
        if self.AdaptiveQuant is not None:
            lls += (s_likelihood,)

        return input, lls, Y_error, jac, code, BDQ


class AugmentedNormalizedHyperPriorCoder2(HyperPriorCoder):
    """AugmentedNormalizedHyperPriorCoder"""

    def __init__(self, num_filters, num_features, num_hyperpriors, gdn_mode="standard",
                 in_channels=3, out_channels=3, kernel_size=5, num_layers=1,
                 init_code='gaussian', use_DQ=False, share_wei=False, use_code=True, dec_add=False, use_context=False,
                 condition='Gaussian', quant_mode='noise'):
        super(AugmentedNormalizedHyperPriorCoder2, self).__init__(
            num_features, num_hyperpriors, True, False, use_context, condition, quant_mode)
        self.num_layers = num_layers
        self.share_wei = share_wei
        if not isinstance(num_filters, list):
            num_filters = [num_filters]
        if len(num_filters) != num_layers:
            num_filters = [num_filters[0]] * num_layers

        if not share_wei:
            self.__delattr__('analysis')
            self.__delattr__('synthesis')

            for i in range(num_layers):
                self.add_module('analysis'+str(i), AugmentedNormalizedAnalysisTransform(
                    in_channels, num_features, num_filters[i], kernel_size, use_code=use_code and init_code != 'zeros', distribution=init_code, gdn_mode=gdn_mode))
                self.add_module('synthesis'+str(i), AugmentedNormalizedSynthesisTransform(
                    in_channels, num_features, num_filters[i], kernel_size, use_code=use_code and i != num_layers-1 and not dec_add, distribution=init_code, gdn_mode=gdn_mode))

            self.add_module('hyper_analysis', AugmentedNormalizedHyperAnalysisTransform(
                num_features, num_filters[i], num_hyperpriors, use_code=use_code and init_code != 'zeros', distribution=init_code))
            self.add_module('hyper_synthesis', AugmentedNormalizedHyperSynthesisTransform(
                num_features, num_filters[i], num_hyperpriors, use_code=use_code or i == num_layers-1, distribution=init_code))
        else:
            self.analysis = AugmentedNormalizedAnalysisTransform(
                in_channels, num_features, num_filters[0], kernel_size, use_code=use_code, distribution=init_code, gdn_mode=gdn_mode)
            self.synthesis = AugmentedNormalizedSynthesisTransform(
                in_channels, num_features, num_filters[0], kernel_size, use_code=use_code, distribution=init_code, gdn_mode=gdn_mode)
            self.hyper_analysis = AugmentedNormalizedHyperAnalysisTransform(
                num_features, num_filters[0], num_hyperpriors, use_code=True, distribution=init_code)
            self.hyper_synthesis = AugmentedNormalizedHyperSynthesisTransform(
                num_features, num_filters[0], num_hyperpriors, use_code=use_code, distribution=init_code)

        for name, m in self.named_children():
            if "ana" in name or "syn" in name:
                m.name = name

        if use_DQ:
            self.DQ = DeQuantizationModule(3, 3, 64, 6)
        else:
            self.DQ = None

    def __getitem__(self, key):
        return self.__getattr__(key[:-1] if self.share_wei else key)

    def encode(self, input, code=None, jac=None, visual=False, figname=''):
        for i in range(self.num_layers):
            debug('F', i)
            _, code, jac = self['analysis'+str(i)](input, code, jac, layer=i)

            if i < self.num_layers-1:
                debug('S', i)
                input, _, jac = self['synthesis'+str(i)](
                    input, code, jac, layer=i, visual=visual, figname=figname+"_"+str(i))

        return input, code, jac

    def decode(self, input, code=None, jac=None, rec_code=False, visual=False, figname=''):
        for i in range(self.num_layers-1, -1, -1):
            debug('rF', i)
            input, _, jac = self['synthesis'+str(i)](
                input, code, jac, rev=True, last_layer=i == self.num_layers-1, layer=i, visual=visual, figname=figname+"_rev_"+str(i))

            if i or rec_code or jac is not None:
                debug('RF', i)
                _, code, jac = self['analysis' +
                                    str(i)](input, code, jac, rev=True, layer=i)

        return input, code, jac

    def forward(self, input, code=None, hyper_code=None, jac=None, rev_ng=False, rec_code=False, IDQ=False, visual=False, figname=''):
        # Encode

        if visual:
            visualizer.set_mean_var(input)
            visualizer.queue_visual(input, figname+"_ori.png")

        jac = [] if jac else None
        input, code, jac = self.encode(
            input, code, jac, visual=visual, figname=figname)

        # Enrtopy coding
        debug('E')
        _, hyper_code, jac = self.hyper_analysis(code, hyper_code, jac)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition, jac = self.hyper_synthesis.get_condition(
            z_tilde, jac, layer=self.num_layers-1)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition)
        # y_tilde = code

        # Encode distortion
        Y_error, _, jac = self['synthesis'+str(self.num_layers-1)](
            input, y_tilde, jac, last_layer=True, layer=self.num_layers-1, visual=visual, figname=figname+"_"+str(self.num_layers-1))

        if visual:
            visualizer.plot_queue(figname+'_for.png')

        input, code, hyper_code = torch.zeros_like(input), y_tilde, z_tilde
        # input, code, hyper_code = Y_error, y_tilde, z_tilde
        debug(Y_error.shape, code.shape, hyper_code.shape)
        if (jac is not None) and not rev_ng:
            jac.append(jac[-1].neg())

        if (jac is not None) or rec_code:
            if rev_ng:
                with torch.no_grad():
                    _, hyper_code, _ = self.hyper_analysis(
                        code, hyper_code, None, rev=True)
            else:
                _, hyper_code, jac = self.hyper_analysis(
                    code, hyper_code, jac, rev=True)

        # Decode
        if rev_ng:
            with torch.no_grad():
                input, code, _ = self.decode(
                    input, code, None, rec_code, visual=visual, figname=figname)
        else:
            input, code, jac = self.decode(
                input, code, jac, rec_code, visual=visual, figname=figname)

        if visual:
            visualizer.plot_queue(figname+'_rev.png')
        debug('END\n')

        if self.DQ is not None:
            debug("DQ")
            BDQ = input
            input = self.DQ(input)
            if visual:
                save_image(BDQ, figname+"_BDQ.png")
                save_image(input, figname+"_ADQ.png")
        else:
            BDQ = None

        if jac is not None:
            jac = torch.stack(jac, 1)

        return input, (y_likelihood, z_likelihood), Y_error, jac, BDQ


class AugmentedNormalizedHyperPriorCoder(HyperPriorCoder):
    """AugmentedNormalizedHyperPriorCoder"""

    def __init__(self, num_filters, num_features, num_hyperpriors, gdn_mode="standard",
                 in_channels=3, out_channels=3, kernel_size=5, num_layers=1,
                 init_code='gaussian', use_DQ=False, share_wei=False, use_code=True, dec_add=False, use_mean=False, use_context=False,
                 condition='Gaussian', quant_mode='noise', 
                 output_nought=True, # Set False when applying ANFIC for residual coding, which will set output(x_2) as MC frame
                 cond_coupling=False #Set True when applying ANFIC for residual coding, which will take MC frame as condition
                 ):

        super(AugmentedNormalizedHyperPriorCoder, self).__init__(
            num_features, num_hyperpriors, True, False, use_context, condition, quant_mode)
        self.num_layers = num_layers
        self.share_wei = share_wei
        self.output_nought=output_nought
        self.cond_coupling = cond_coupling

        if not isinstance(num_filters, list):
            num_filters = [num_filters]
        if len(num_filters) != num_layers:
            num_filters = [num_filters[0]] * num_layers

        if not share_wei:
            self.__delattr__('analysis')
            self.__delattr__('synthesis')
            self.__delattr__('hyper_analysis')
            self.__delattr__('hyper_synthesis')

            for i in range(num_layers):
                if not self.cond_coupling:# Default
                    self.add_module('analysis'+str(i), AugmentedNormalizedAnalysisTransform(
                        in_channels, num_features, num_filters[i], kernel_size, 
                        use_code=use_code and init_code != 'zeros', distribution=init_code, gdn_mode=gdn_mode)
                        )
                    self.add_module('synthesis'+str(i), AugmentedNormalizedSynthesisTransform(
                        in_channels, num_features, num_filters[i], kernel_size, 
                        use_code=use_code and i != num_layers-1 and not dec_add, distribution=init_code, gdn_mode=gdn_mode)
                        )
                else: # For residual coding ; make transform conditional
                    self.add_module('analysis'+str(i), AugmentedNormalizedAnalysisTransform(
                        in_channels*2, num_features, num_filters[i], kernel_size, 
                        use_code=use_code and init_code != 'zeros', distribution=init_code, gdn_mode=gdn_mode)
                        )
                    self.add_module('synthesis'+str(i), AugmentedNormalizedSynthesisTransform(
                        in_channels*2, num_features, num_filters[i], kernel_size, 
                        use_code=use_code and i != num_layers-1 and not dec_add, distribution=init_code, gdn_mode=gdn_mode)
                        )
                    # For making down-scaled MC frame feature for synthesis conditioning
                    self.add_module('cond_encoder'+str(i), GoogleAnalysisTransform(
                        in_channels, num_features, num_filters[i], kernel_size
                        ))

                self.add_module('hyper_analysis'+str(i), AugmentedNormalizedHyperAnalysisTransform(
                    num_features, num_filters[i], num_hyperpriors, use_code=use_code and init_code != 'zeros', distribution=init_code))
                self.add_module('hyper_synthesis'+str(i), AugmentedNormalizedHyperSynthesisTransform(
                    num_features, num_filters[i], num_hyperpriors, use_code=use_code or i == num_layers-1, distribution=init_code))
        else:
            #TODO: Double analysis & synthesis input channels when self.cond_coupling==True
            self.analysis = AugmentedNormalizedAnalysisTransform(
                in_channels, num_features, num_filters[0], kernel_size, use_code=use_code, distribution=init_code, gdn_mode=gdn_mode)
            self.synthesis = AugmentedNormalizedSynthesisTransform(
                in_channels, num_features, num_filters[0], kernel_size, use_code=use_code, distribution=init_code, gdn_mode=gdn_mode)
            self.hyper_analysis = AugmentedNormalizedHyperAnalysisTransform(
                num_features, num_filters[0], num_hyperpriors, use_code=True, distribution=init_code)
            self.hyper_synthesis = AugmentedNormalizedHyperSynthesisTransform(
                num_features, num_filters[0], num_hyperpriors, use_code=use_code, distribution=init_code)

        for name, m in self.named_children():
            if "ana" in name or "syn" in name:
                m.name = name

        if use_DQ:
            self.DQ = DeQuantizationModule(3, 3, 64, 6)
        else:
            self.DQ = None

    def __getitem__(self, key):
        return self.__getattr__(key[:-1] if self.share_wei else key)

    def encode(self, input, code=None, hyper_code=None, jac=None, visual=False, figname='', cond_coupling_input=None):
        for i in range(self.num_layers):
            debug('F', i)
            # Concat input with condition (MC frame)
            if self.cond_coupling:
                cond = cond_coupling_input
                cond_input = torch,cat([input, cond], dim=1)
                _, code, jac = self['analysis'+str(i)](cond_input, code, jac, layer=i)
            else:
                _, code, jac = self['analysis'+str(i)](input, code, jac, layer=i)
            _, hyper_code, jac = self['hyper_analysis' +
                                      str(i)](code, hyper_code, jac, layer=i)

            if i < self.num_layers-1:
                debug('S', i)
                # Concat code with condition (MC frame feature)
                if self.cond_coupling:
                    cond = self['cond_encoder'+str(i)](cond_coupling_input)
                    cond_code = torch,cat([code, cond], dim=1)
                    input, _, jac = self['synthesis'+str(i)](
                        input, cond_code, jac, layer=i, visual=visual, figname=figname+"_"+str(i))
                code, _, jac = self['hyper_synthesis' +
                                    str(i)](code, hyper_code, jac, layer=i)

        return input, code, hyper_code, jac

    def decode(self, input, code=None, hyper_code=None, jac=None, rec_code=False, visual=False, figname='', cond_coupling_input=None):
        for i in range(self.num_layers-1, -1, -1):
            debug('rF', i)
            if i < self.num_layers-1:
                debug('RS', i)
                code, _, jac = self['hyper_synthesis' +
                                    str(i)](code, hyper_code, jac, rev=True, layer=i)
 
            # Concat code with condition (MC frame feature)
            if self.cond_coupling:
                cond = self['cond_encoder'+str(i)](cond_coupling_input)
                cond_code = torch,cat([code, cond], dim=1)
                input, _, jac = self['synthesis'+str(i)](
                    input, cond_code, jac, rev=True, last_layer=i == self.num_layers-1, layer=i, visual=visual, figname=figname+"_rev_"+str(i))
            else:
                input, _, jac = self['synthesis'+str(i)](
                    input, code, jac, rev=True, last_layer=i == self.num_layers-1, layer=i, visual=visual, figname=figname+"_rev_"+str(i))

            if i or rec_code or jac is not None:
                debug('RF', i)
                _, hyper_code, jac = self['hyper_analysis' +
                                          str(i)](code, hyper_code, jac, rev=True, layer=i)
                # Concat input (x) with condition (MC frame)
                if self.cond_coupling:
                    cond = cond_coupling_input
                    cond_input = torch,cat([input, cond], dim=1)
                    _, code, jac = self['analysis' +
                                        str(i)](cond_input, code, jac, rev=True, layer=i)
                else:
                    _, code, jac = self['analysis' +
                                        str(i)](input, code, jac, rev=True, layer=i)

        return input, code, hyper_code, jac

    def forward(self, input, code=None, hyper_code=None, jac=None, rev_ng=False, 
                rec_code=False, IDQ=False, detach_enc=False, visual=False, figname='',
                output=None, # Should assign value when self.output_nought==False
                cond_coupling_input=None # Should assign value when self.cond_coupling==True
                                         # When using ANFIC on residual coding, output & cond_coupling_input should both be MC frame
                ):
        if not self.output_nought:
            assert not (output is None), "output should be specified"
        if self.cond_coupling:
            assert not (cond_coupling_input is None), "cond_coupling_input should be specified"

        # Encode
        if IDQ:
            debug("IDQ")
            input = input.add(torch.rand_like(input).sub_(0.5).div_(255.))

        if visual:
            visualizer.set_mean_var(input)
            visualizer.queue_visual(input, figname+"_ori.png")
            logger.open(figname+"_stati.txt")

        jac = [] if jac else None
        input, code, hyper_code, jac = self.encode(
            input, code, hyper_code, jac, visual=visual, figname=figname, cond_coupling_input=cond_coupling_input)

        # Enrtopy coding
        debug('E')
        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition, jac = self['hyper_synthesis' +
                              str(self.num_layers-1)].get_condition(z_tilde, jac, layer=self.num_layers-1)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition)
        # y_tilde = code

        # Encode distortion
        
        # Concat code with condition (MC frame feature)
        if self.cond_coupling:
            cond = self['cond_encoder'+str(self.num_layers-1)](cond_coupling_input)
            y_tilde = torch,cat([y_tilde, cond], dim=1)
        # Last synthesis transform
        Y_error, _, jac = self['synthesis'+str(self.num_layers-1)](
            input, y_tilde, jac, last_layer=True, layer=self.num_layers-1, visual=visual, figname=figname+"_"+str(self.num_layers-1))

        if visual:
            visualizer.plot_queue(figname+'_for.png')

            logger.write(check_range(Y_error, "X"+str(self.num_layers-1)))
            logger.write(check_range(code, "code"+str(self.num_layers-1)))
            logger.write()

        # input, code, hyper_code = torch.zeros_like(input), y_tilde, z_tilde
        input, code, hyper_code = output, y_tilde, z_tilde # Take MC frame for decoding
        debug(Y_error.shape, code.shape, hyper_code.shape)

        if jac is not None and not rev_ng:
            jac.append(jac[-1].neg())

        if visual:
            logger.write(check_range(input, "X"+str(self.num_layers-1)))
            logger.write(check_range(code, "code"+str(self.num_layers-1)))
            logger.write()

        # Decode
        if rev_ng:
            with torch.no_grad():
                input, code, hyper_code, _ = self.decode(
                    input, code, hyper_code, None, rec_code, visual=visual, figname=figname, 
                    cond_coupling_input=cond_coupling_input
                    )
        else:
            input, code, hyper_code, _ = self.decode(
                input, code, hyper_code, None, rec_code, visual=visual, figname=figname, 
                cond_coupling_input=cond_coupling_input
                )

        if visual:
            visualizer.plot_queue(figname+'_rev.png')
            logger.close()

        if self.DQ is not None:
            debug("DQ")
            BDQ = input
            input = self.DQ(input)
            if visual:
                save_image(BDQ, figname+"_BDQ.png")
                save_image(input, figname+"_ADQ.png")
        else:
            BDQ = None

        if jac is not None:
            jac = torch.stack(jac, 1)

        debug('END\n')

        return input, (y_likelihood, z_likelihood), Y_error, jac, BDQ


class Coarse2FineSynthesis(nn.Module):

    def __init__(self, out_channels, num_features, num_filters, kernel_size, simplify_gdn, use_mean):
        super(Coarse2FineSynthesis, self).__init__()
        self.deconv = nn.Sequential(
            ConvTranspose2d(num_features, num_filters,
                            kernel_size, stride=2),
            GeneralizedDivisiveNorm(
                num_filters, inverse=True, simplify=simplify_gdn),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size, stride=2),
            GeneralizedDivisiveNorm(
                num_filters, inverse=True, simplify=simplify_gdn),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size, stride=2),
            GeneralizedDivisiveNorm(
                num_filters, inverse=True, simplify=simplify_gdn)
        )

        self.SideInfoRecon = nn.Sequential(
            ConvTranspose2d(num_features*2*(2 if use_mean else 1), num_filters,
                            kernel_size, stride=2),
            nn.LeakyReLU(inplace=True),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size, stride=2),
            nn.LeakyReLU(inplace=True),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size, stride=2),
            nn.LeakyReLU(inplace=True)
        )

        self.fusion = nn.Sequential(
            ResidualBlock(num_filters*2),
            ConvTranspose2d(num_filters*2, num_filters,
                            kernel_size, stride=2),
            Conv2d(num_filters, num_filters//4, kernel_size=3),
            Conv2d(num_filters//4, out_channels, kernel_size=1)
        )

    def forward(self, features, condition, condition2):
        hyper_align = F.interpolate(
            condition2, condition.size()[-2:], mode='bicubic', align_corners=False)

        sideinfo = self.SideInfoRecon(
            torch.cat([condition, hyper_align], dim=1))

        deconv1 = self.deconv(features)

        reconstructed = self.fusion(torch.cat([deconv1, sideinfo], dim=1))

        return reconstructed


class Coarse2FineHyperPriorCoder(GoogleHyperPriorCoder):
    """Coarse2FineHyperPriorCoder"""

    def __init__(self, num_filters, num_features, num_hyperpriors, use_IAR=True,
                 simplify_gdn=False, in_channels=3, out_channels=3, kernel_size=5, use_mean=False, use_abs=False,
                 condition='Gaussian', quant_mode='noise'):
        assert num_features == num_hyperpriors, (num_features, num_hyperpriors)
        super(Coarse2FineHyperPriorCoder, self).__init__(
            num_filters, num_features, num_hyperpriors, simplify_gdn, in_channels, out_channels, kernel_size,
            use_mean, use_abs, condition, quant_mode)

        self.hyper_analysis2 = nn.Sequential(
            Conv2d(num_hyperpriors, num_filters, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            Conv2d(num_filters, num_filters, kernel_size=1, stride=2),
            nn.ReLU(inplace=True),
            Conv2d(num_filters, num_hyperpriors, kernel_size=1, stride=1)
        )

        if self.use_mean:
            self.hyper_synthesis2 = nn.Sequential(
                ConvTranspose2d(num_hyperpriors, num_filters,
                                kernel_size=1, stride=1),
                nn.ReLU(inplace=True),
                ConvTranspose2d(num_filters, num_filters * 3 // 2,
                                kernel_size=1, stride=2),
                nn.ReLU(inplace=True),
                ConvTranspose2d(num_filters * 3 // 2, num_hyperpriors*self.conditional_bottleneck.condition_size,
                                kernel_size=3, stride=1)
            )
        else:
            self.hyper_synthesis2 = nn.Sequential(
                ConvTranspose2d(num_hyperpriors, num_filters,
                                kernel_size=1, stride=1, parameterizer=None),
                nn.ReLU(inplace=True),
                ConvTranspose2d(num_filters, num_filters,
                                kernel_size=1, stride=2, parameterizer=None),
                nn.ReLU(inplace=True),
                ConvTranspose2d(num_filters, num_hyperpriors,
                                kernel_size=1, stride=1, parameterizer=None)
            )

        self.use_IAR = use_IAR
        if self.use_IAR:
            self.synthesis = Coarse2FineSynthesis(
                out_channels, num_features, num_filters, kernel_size, simplify_gdn, use_mean)

        self.divisor = 128

    def forward(self, input):
        features = self.analysis(input)

        hyperpriors = self.hyper_analysis(
            features.abs() if self.use_abs else features)

        hyperpriors2 = self.hyper_analysis2(
            hyperpriors.abs() if self.use_abs else hyperpriors)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyperpriors2)

        condition2 = self.hyper_synthesis2(z_tilde)

        y_tilde2, y_likelihood2 = self.conditional_bottleneck(
            hyperpriors, condition=condition2)

        condition = self.hyper_synthesis(y_tilde2)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            features, condition=condition)

        if self.use_IAR:
            reconstructed = self.synthesis(y_tilde, condition, condition2)
        else:
            reconstructed = self.synthesis(y_tilde)

        return reconstructed, (y_likelihood, y_likelihood2, z_likelihood)


class GoogleContextCoder(ContextCoder):
    """GoogleContextCoder"""

    def __init__(self, num_filters, num_features, num_hyperpriors, simplify_gdn=False,
                 in_channels=3, out_channels=3, kernel_size=5, use_mean=False, use_abs=False,
                 condition='Gaussian', quant_mode='noise'):
        super(GoogleContextCoder, self).__init__(
            num_features, num_hyperpriors, use_mean, use_abs, condition, quant_mode)

        self.analysis = GoogleAnalysisTransform(
            in_channels, num_features, num_filters, kernel_size, simplify_gdn)

        self.synthesis = GoogleSynthesisTransform(
            out_channels, num_features, num_filters, kernel_size, simplify_gdn)

        self.hyper_analysis = GoogleHyperAnalysisTransform(
            num_features, num_filters, num_hyperpriors)

        self.hyper_synthesis = GoogleHyperSynthesisTransform(
            num_features*2, num_filters, num_hyperpriors)


class GoogleContextCoder2(GoogleContextCoder):
    """GoogleContextCoder2"""

    def __init__(self, num_filters, num_features, num_hyperpriors, simplify_gdn=False,
                 in_channels=3, out_channels=3, kernel_size=5, use_mean=False, use_abs=False,
                 condition='Gaussian', quant_mode='noise'):
        super(GoogleContextCoder2, self).__init__(
            num_filters, num_features, num_hyperpriors, simplify_gdn, in_channels, out_channels, kernel_size, use_mean, use_abs, condition, quant_mode)
        self.post = DeQuantizationModule(3, 3, 64, 6)

    def forward(self, input):
        rec, ll = super().forward(input)
        return self.post(rec), ll


class _ShortCutBlock(nn.Module):
    """Basic ShortCutBlock"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, transposed, simplify_gdn=False):
        super(_ShortCutBlock, self).__init__()
        self.shortcut = stride != 1
        conv = ConvTranspose2d if transposed else Conv2d
        if self.shortcut:
            if kernel_size == 5:
                self.residual = nn.Sequential(
                    conv(in_channels, out_channels,
                         kernel_size=3, stride=stride),
                    nn.LeakyReLU(inplace=True),
                    conv(out_channels, out_channels, kernel_size=3),
                    GeneralizedDivisiveNorm(
                        out_channels, inverse=transposed, simplify=simplify_gdn)
                )
            else:
                self.residual = nn.Sequential(
                    conv(in_channels, out_channels,
                         kernel_size=3, stride=stride),
                    GeneralizedDivisiveNorm(
                        out_channels, inverse=transposed, simplify=simplify_gdn)
                )
            self.shortcut = conv(in_channels, out_channels, 1, stride)
        else:
            assert in_channels == out_channels
            if kernel_size == 5:
                self.residual = nn.Sequential(
                    conv(in_channels, out_channels, kernel_size=3),
                    nn.LeakyReLU(inplace=True),
                    conv(out_channels, out_channels, kernel_size=3),
                    nn.LeakyReLU(inplace=True)
                )
            else:
                self.residual = nn.Sequential(
                    conv(in_channels, out_channels, kernel_size=3),
                    nn.LeakyReLU(inplace=True)
                )

    def forward(self, input):
        if self.shortcut:
            return self.residual(input) + self.shortcut(input)
        else:
            return self.residual(input) + input


class ShortCutBlock(_ShortCutBlock):
    """ShortCutBlock"""

    def __init__(self, in_channels, out_channels, kernel_size, simplify_gdn=False, stride=1):
        super(ShortCutBlock, self).__init__(
            in_channels, out_channels, kernel_size, stride, False, simplify_gdn)


class TransposedShortCutBlock(_ShortCutBlock):
    """TransposedShortCutBlock"""

    def __init__(self, in_channels, out_channels, kernel_size, simplify_gdn=False, stride=1):
        super(TransposedShortCutBlock, self).__init__(
            in_channels, out_channels, kernel_size, stride, True, simplify_gdn)


class CSTKAnalysisTransform(nn.Sequential):
    """CSTKAnalysisTransform"""

    def __init__(self, in_channels, num_filters, num_features, kernel_size, simplify_gdn):
        super(CSTKAnalysisTransform, self).__init__(
            ShortCutBlock(in_channels, num_filters, kernel_size,
                          simplify_gdn=simplify_gdn, stride=2),
            ShortCutBlock(num_filters, num_filters, kernel_size),
            ShortCutBlock(num_filters, num_filters, kernel_size,
                          simplify_gdn=simplify_gdn, stride=2),
            AttentionBlock(num_filters),
            ShortCutBlock(num_filters, num_filters, kernel_size),
            ShortCutBlock(num_filters, num_filters, kernel_size,
                          simplify_gdn=simplify_gdn, stride=2),
            ShortCutBlock(num_filters, num_filters, kernel_size),
            AttentionBlock(num_filters),
            Conv2d(num_filters, num_features, 3, stride=2)
        )


class CSTKSynthesisTransform(nn.Sequential):
    """CSTKSynthesisTransform"""

    def __init__(self, out_channels, num_filters, num_features, kernel_size, simplify_gdn):
        super(CSTKSynthesisTransform, self).__init__(
            TransposedShortCutBlock(
                num_features, num_features, kernel_size),
            AttentionBlock(num_features),
            TransposedShortCutBlock(
                num_features, num_filters, kernel_size, simplify_gdn=simplify_gdn, stride=2),
            TransposedShortCutBlock(num_filters, num_filters, kernel_size),
            TransposedShortCutBlock(
                num_filters, num_filters, kernel_size, simplify_gdn=simplify_gdn, stride=2),
            AttentionBlock(num_filters),
            TransposedShortCutBlock(num_filters, num_filters, kernel_size),
            TransposedShortCutBlock(
                num_filters, num_filters, kernel_size, simplify_gdn=simplify_gdn, stride=2),
            TransposedShortCutBlock(num_filters, num_filters, kernel_size),
            ConvTranspose2d(num_filters, out_channels, 3, stride=2)
        )


class CSTKHyperAnalysisTransform(nn.Sequential):
    def __init__(self, num_features, num_filters, num_hyperpriors):
        super(CSTKHyperAnalysisTransform, self).__init__(
            Conv2d(num_features, num_filters, 3, stride=1),
            nn.LeakyReLU(inplace=True),
            Conv2d(num_filters, num_filters, 3),
            nn.LeakyReLU(inplace=True),
            Conv2d(num_filters, num_filters, 3, stride=2),
            nn.LeakyReLU(inplace=True),
            Conv2d(num_filters, num_filters, 3),
            nn.LeakyReLU(inplace=True),
            Conv2d(num_filters, num_hyperpriors, 3, stride=2)
        )


class CSTKHyperSynthesisTransform(nn.Sequential):
    def __init__(self, num_features, num_filters, num_hyperpriors):
        super(CSTKHyperSynthesisTransform, self).__init__(
            ConvTranspose2d(num_hyperpriors, num_filters, 3, stride=1),
            nn.LeakyReLU(inplace=True),
            ConvTranspose2d(num_filters, num_filters, 3, stride=2),
            nn.LeakyReLU(inplace=True),
            ConvTranspose2d(num_filters, num_filters * 3 // 2, 3),
            nn.LeakyReLU(inplace=True),
            ConvTranspose2d(num_filters * 3 // 2,
                            num_filters * 3 // 2, 3, stride=2),
            nn.LeakyReLU(inplace=True),
            ConvTranspose2d(num_filters * 3 // 2, num_features, 3, stride=1)
        )


class CSTKContextCoder(ContextCoder):
    """CSTKContextCoder"""

    def __init__(self, num_filters, num_features, num_hyperpriors, simplify_gdn=False,
                 in_channels=3, out_channels=3, kernel_size=5, use_mean=False, use_abs=False,
                 condition='GaussianMixtureModel', quant_mode='noise'):
        super(CSTKContextCoder, self).__init__(
            num_features, num_hyperpriors, use_mean, use_abs, condition, quant_mode)

        self.analysis = CSTKAnalysisTransform(
            in_channels, num_features, num_filters, kernel_size, simplify_gdn)

        self.synthesis = CSTKSynthesisTransform(
            out_channels, num_filters, num_features, kernel_size, simplify_gdn)

        self.hyper_analysis = CSTKHyperAnalysisTransform(
            num_features, num_filters, num_hyperpriors)

        self.hyper_synthesis = CSTKHyperSynthesisTransform(
            num_features*2, num_filters, num_hyperpriors)


class AugmentedNormalizedCSTKAnalysisTransform(AugmentedNormalizedFlow):
    def __init__(self, in_channels, num_features, num_filters, kernel_size, use_code, distribution, gdn_mode, use_attn=False):
        super(AugmentedNormalizedCSTKAnalysisTransform, self).__init__(
            ShortCutBlock(in_channels, num_filters, kernel_size,
                          simplify_gdn=gdn_mode, stride=2),
            ShortCutBlock(num_filters, num_filters, kernel_size),
            ShortCutBlock(num_filters, num_filters, kernel_size,
                          simplify_gdn=gdn_mode, stride=2),
            # AttentionBlock(num_filters),
            ShortCutBlock(num_filters, num_filters, kernel_size),
            ShortCutBlock(num_filters, num_filters, kernel_size,
                          simplify_gdn=gdn_mode, stride=2),
            ShortCutBlock(num_filters, num_filters, kernel_size),
            # AttentionBlock(num_filters),
            Conv2d(num_filters, num_features, 3, stride=2),
            use_code=use_code, transpose=False, distribution=distribution
        )


class AugmentedNormalizedCSTKSynthesisTransform(AugmentedNormalizedFlow):
    def __init__(self, out_channels, num_features, num_filters, kernel_size, use_code, distribution, gdn_mode, use_attn=False):
        super(AugmentedNormalizedCSTKSynthesisTransform, self).__init__(
            TransposedShortCutBlock(
                num_features, num_features, kernel_size),
            # AttentionBlock(num_features),
            TransposedShortCutBlock(
                num_features, num_filters, kernel_size, simplify_gdn=gdn_mode, stride=2),
            TransposedShortCutBlock(num_filters, num_filters, kernel_size),
            TransposedShortCutBlock(
                num_filters, num_filters, kernel_size, simplify_gdn=gdn_mode, stride=2),
            # AttentionBlock(num_filters),
            TransposedShortCutBlock(num_filters, num_filters, kernel_size),
            TransposedShortCutBlock(
                num_filters, num_filters, kernel_size, simplify_gdn=gdn_mode, stride=2),
            TransposedShortCutBlock(num_filters, num_filters, kernel_size),
            ConvTranspose2d(num_filters, out_channels, 3, stride=2),
            use_code=use_code, transpose=True, distribution=distribution
        )


class AugmentedNormalizedFlowCSTKCoder(HyperPriorCoder):
    """AugmentedNormalizedFlowCSTKCoder"""

    def __init__(self, num_filters, num_features, num_hyperpriors, gdn_mode="standard",
                 in_channels=3, out_channels=3, kernel_size=5, num_layers=1,
                 init_code='gaussian', use_DQ=False, share_wei=False, use_code=True, dec_add=False,
                 use_attn=False,
                 hyper_filters=192, use_mean=False, use_context=False,
                 condition='Gaussian', quant_mode='noise'):
        super(AugmentedNormalizedFlowCSTKCoder, self).__init__(
            num_features, num_hyperpriors, use_mean, False, use_context, condition, quant_mode)
        self.num_layers = num_layers
        self.share_wei = share_wei
        if not isinstance(num_filters, list):
            num_filters = [num_filters]
        if len(num_filters) != num_layers:
            num_filters = [num_filters[0]] * num_layers

        if not share_wei:
            self.__delattr__('analysis')
            self.__delattr__('synthesis')

            for i in range(num_layers):
                self.add_module('analysis'+str(i), AugmentedNormalizedCSTKAnalysisTransform(
                    in_channels, num_features, num_filters[i], kernel_size, use_code=use_code and init_code != 'zeros', distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn and i == num_layers-1))
                self.add_module('synthesis'+str(i), AugmentedNormalizedCSTKSynthesisTransform(
                    in_channels, num_features, num_filters[i], kernel_size, use_code=use_code and i != num_layers-1 and not dec_add, distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn and i == num_layers-1))
        else:
            self.analysis = AugmentedNormalizedAnalysisTransform(
                in_channels, num_features, num_filters[0], kernel_size, use_code=use_code, distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn)
            self.synthesis = AugmentedNormalizedSynthesisTransform(
                in_channels, num_features, num_filters[0], kernel_size, use_code=use_code, distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn)

        self.hyper_analysis = CSTKHyperAnalysisTransform(
            num_features, hyper_filters, num_hyperpriors)

        self.hyper_synthesis = CSTKHyperSynthesisTransform(
            num_features*2, hyper_filters, num_hyperpriors)

        for name, m in self.named_children():
            if "ana" in name or "syn" in name:
                m.name = name

        if use_DQ:
            self.DQ = DeQuantizationModule(3, 3, 128, 6)
        else:
            self.DQ = None

    def __getitem__(self, key):
        return self.__getattr__(key[:-1] if self.share_wei else key)

    def encode(self, input, code=None, jac=None, visual=False, figname=''):
        for i in range(self.num_layers):
            debug('F', i)
            _, code, jac = self['analysis'+str(i)](
                input, code, jac, layer=i, visual=visual, figname=figname+"_"+str(i))

            if i < self.num_layers-1:
                debug('S', i)
                input, _, jac = self['synthesis'+str(i)](
                    input, code, jac, layer=i, visual=visual, figname=figname+"_"+str(i))

            if visual:
                logger.write(check_range(input, "X"+str(i)))
                logger.write(check_range(code, "code"+str(i)))
                logger.write()

        return input, code, jac

    def decode(self, input, code=None, jac=None, rec_code=False, visual=False, figname=''):
        for i in range(self.num_layers-1, -1, -1):
            debug('rF', i)
            input, _, jac = self['synthesis'+str(i)](
                input, code, jac, rev=True, last_layer=i == self.num_layers-1, layer=i, visual=visual, figname=figname+"_rev_"+str(i))

            if i or rec_code or jac is not None:
                debug('RF', i)
                _, code, jac = self['analysis' +
                                    str(i)](input, code, jac, rev=True, layer=i, visual=visual, figname=figname+"_rev_"+str(i))

            if visual:
                logger.write(check_range(input, "X"+str(i)))
                logger.write(check_range(code, "code"+str(i)))
                logger.write()

        return input, code, jac

    def entropy_model(self, input, code, jac=False):
        # Enrtopy coding
        debug('E')

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition)
        # y_tilde = code

        # Encode distortion
        Y_error, _, jac = self['synthesis'+str(self.num_layers-1)](
            input, y_tilde, jac, last_layer=True, layer=self.num_layers-1)

        return Y_error, y_tilde, z_tilde, y_likelihood, z_likelihood

    def forward(self, input, code=None, jac=None, rec_code=False, rev_ng=False, IDQ=False, detach_enc=False, visual=False, figname=''):
        # Encode
        ori_input = input

        if visual:
            visualizer.set_mean_var(input)
            visualizer.queue_visual(input, figname+"_ori.png")
            logger.open(figname+"_stati.txt")

        jac = [] if jac else None
        if detach_enc:
            with torch.no_grad():
                input, code, jac = self.encode(
                    input, code, jac, visual=visual, figname=figname)
        else:
            input, code, jac = self.encode(
                input, code, jac, visual=visual, figname=figname)

        # Enrtopy coding
        debug('E')

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition)
        # y_tilde = code

        # Encode distortion
        Y_error, _, jac = self['synthesis'+str(self.num_layers-1)](
            input, y_tilde, jac, last_layer=True, layer=self.num_layers-1, visual=visual, figname=figname+"_"+str(self.num_layers-1))

        if visual:
            visualizer.plot_queue(figname+'_for.png', nrow=input.size(0))

            with torch.no_grad():
                YLL_perC = y_likelihood.clamp(
                    1e-9).log2().mean(dim=(2, 3)).neg()
                fig = plt.figure()
                check_range(YLL_perC, "YLL")
                C = YLL_perC.size(1)
                for YLL in YLL_perC:
                    max_idx = YLL.max(0)[1]
                    plt.bar(torch.arange(C), YLL.data.cpu(),
                            label="maxC="+str(max_idx.item()))
                plt.legend()
                # plt.show()
                plt.savefig(figname+"_YLL.png")
                plt.close(fig)

                nrow = 8
                save_image(code.div(255*2).add(0.5).flatten(0, 1).unsqueeze(
                    1), figname+"_feature.png", nrow=nrow)
                save_image(max_norm(code).add(0.5).flatten(0, 1).unsqueeze(
                    1), figname+"_feature_norm.png", nrow=nrow)
                save_image(y_tilde.div(255*2).add(0.5).flatten(0, 1).unsqueeze(
                    1), figname+"_quant.png", nrow=nrow)
                save_image(max_norm(y_tilde).add(0.5).flatten(0, 1).unsqueeze(
                    1), figname+"_quant_norm.png", nrow=nrow)

            logger.write(check_range(Y_error, "X"+str(self.num_layers-1)))
            logger.write(check_range(code, "code"+str(self.num_layers-1)))
            logger.write()

        input, code, hyper_code = torch.zeros_like(input), y_tilde, z_tilde
        # input, code, hyper_code = Y_error, y_tilde, z_tilde
        # input = Y_error
        debug(Y_error.shape, code.shape, hyper_code.shape)

        if visual:
            logger.write(check_range(input, "X"+str(self.num_layers-1)))
            logger.write(check_range(code, "code"+str(self.num_layers-1)))
            logger.write()

        # Decode
        if rev_ng:
            with torch.no_grad():
                input, code, jac = self.decode(
                    input, code, jac, rec_code=rec_code, visual=visual, figname=figname)
        else:
            input, code, jac = self.decode(
                input, code, jac, rec_code=rec_code, visual=visual, figname=figname)

        if visual:
            visualizer.plot_queue(figname+'_rev.png', nrow=input.size(0))
            logger.close()

        if self.DQ is not None:
            debug("DQ")
            BDQ = input
            input = self.DQ(input)
            if visual:
                save_image(BDQ, figname+"_BDQ.png")
                save_image(input, figname+"_ADQ.png")
        else:
            BDQ = None

        debug('END\n')

        return input, (y_likelihood, z_likelihood), Y_error, jac, code, BDQ


class AugmentedNormalizedFlowCSTKCoder2(HyperPriorCoder):
    """AugmentedNormalizedFlowCSTKCoder"""

    def __init__(self, num_filters, num_features, num_hyperpriors, gdn_mode="standard",
                 in_channels=3, out_channels=3, kernel_size=5, num_layers=1,
                 init_code='gaussian', use_DQ=False, share_wei=False, use_code=True, dec_add=False,
                 use_attn=False, input_norm="shift", use_syntax=False, use_AQ=False, syntax_prior="None",
                 hyper_filters=192, use_mean=False, use_context=False, use_conditionalconv=False, ch_wise=False,
                 condition='Gaussian', quant_mode='noise'):
        super(AugmentedNormalizedFlowCSTKCoder2, self).__init__(
            num_features, num_hyperpriors, use_mean, False, use_context, condition, quant_mode)
        from torch_compression.modules.context_model import ContextModel2
        self.conditional_bottleneck = ContextModel2(
            num_features, num_features*2, self.conditional_bottleneck.entropy_model, kernel_size=5)
        self.num_layers = num_layers
        self.share_wei = share_wei
        if not isinstance(num_filters, list):
            num_filters = [num_filters]
        if len(num_filters) != num_layers:
            num_filters = [num_filters[0]] * num_layers

        if not share_wei:
            self.__delattr__('analysis')
            self.__delattr__('synthesis')

            for i in range(num_layers):
                self.add_module('analysis'+str(i), AugmentedNormalizedCSTKAnalysisTransform(
                    in_channels, num_features, num_filters[i], kernel_size, use_code=use_code and init_code != 'zeros', distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn and i == num_layers-1))
                self.add_module('synthesis'+str(i), AugmentedNormalizedCSTKSynthesisTransform(
                    in_channels, num_features, num_filters[i], kernel_size, use_code=use_code and i != num_layers-1 and not dec_add, distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn and i == num_layers-1))
        else:
            self.analysis = AugmentedNormalizedAnalysisTransform(
                in_channels, num_features, num_filters[0], kernel_size, use_code=use_code, distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn)
            self.synthesis = AugmentedNormalizedSynthesisTransform(
                in_channels, num_features, num_filters[0], kernel_size, use_code=use_code, distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn)

        self.hyper_analysis = CSTKHyperAnalysisTransform(
            num_features, hyper_filters, num_hyperpriors)

        self.hyper_synthesis = CSTKHyperSynthesisTransform(
            num_features*2, hyper_filters, num_hyperpriors)

        for name, m in self.named_children():
            if "ana" in name or "syn" in name:
                m.name = name

        if use_DQ:
            self.DQ = DeQuantizationModule(3, 3, 64, 6)
            if use_syntax:
                self.syntax = SyntaxCoder(
                    3, 32, 64, (3, 64, 3, 3), prior=syntax_prior)
                self.num_bitstreams += 1
            else:
                self.syntax = None
        else:
            self.DQ = None

        self.AdaptiveQuant = AdaptiveQuant2(
            hyper_filters, num_features, num_hyperpriors, use_conditionalconv, ch_wise)
        self.num_bitstreams += 1

    def __getitem__(self, key):
        return self.__getattr__(key[:-1] if self.share_wei else key)

    def encode(self, input, code=None, jac=None, visual=False, figname=''):
        for i in range(self.num_layers):
            debug('F', i)
            _, code, jac = self['analysis'+str(i)](
                input, code, jac, layer=i, visual=visual, figname=figname+"_"+str(i))

            if i < self.num_layers-1:
                debug('S', i)
                input, _, jac = self['synthesis'+str(i)](
                    input, code, jac, layer=i, visual=visual, figname=figname+"_"+str(i))

            if visual:
                logger.write(check_range(input, "X"+str(i)))
                logger.write(check_range(code, "code"+str(i)))
                logger.write()

        return input, code, jac

    def decode(self, input, code=None, jac=None, rec_code=False, visual=False, figname=''):
        for i in range(self.num_layers-1, -1, -1):
            debug('rF', i)
            input, _, jac = self['synthesis'+str(i)](
                input, code, jac, rev=True, last_layer=i == self.num_layers-1, layer=i, visual=visual, figname=figname+"_rev_"+str(i))

            if i or rec_code or jac is not None:
                debug('RF', i)
                _, code, jac = self['analysis' +
                                    str(i)](input, code, jac, rev=True, layer=i, visual=visual, figname=figname+"_rev_"+str(i))

            if visual:
                logger.write(check_range(input, "X"+str(i)))
                logger.write(check_range(code, "code"+str(i)))
                logger.write()

        return input, code, jac

    def entropy_model(self, input, code, jac=False):
        # Enrtopy coding
        debug('E')

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition)
        # y_tilde = code

        # Encode distortion
        Y_error, _, jac = self['synthesis'+str(self.num_layers-1)](
            input, y_tilde, jac, last_layer=True, layer=self.num_layers-1)

        return Y_error, y_tilde, z_tilde, y_likelihood, z_likelihood

    def forward(self, input, code=None, jac=None, rec_code=False, rev_ng=False, IDQ=False, detach_enc=False, visual=False, figname=''):
        # Encode
        ori_input = input

        if visual:
            visualizer.set_mean_var(input)
            visualizer.queue_visual(input, figname+"_ori.png")
            logger.open(figname+"_stati.txt")

        jac = [] if jac else None
        if detach_enc:
            with torch.no_grad():
                input, code, jac = self.encode(
                    input, code, jac, visual=visual, figname=figname)
        else:
            input, code, jac = self.encode(
                input, code, jac, visual=visual, figname=figname)

        # Enrtopy coding
        debug('E')

        scale_factor, s_likelihood = self.AdaptiveQuant(code)

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition, scale_factor=scale_factor)
        # y_tilde = code

        # Encode distortion
        Y_error, _, jac = self['synthesis'+str(self.num_layers-1)](
            input, y_tilde, jac, last_layer=True, layer=self.num_layers-1, visual=visual, figname=figname+"_"+str(self.num_layers-1))

        if visual:
            visualizer.plot_queue(figname+'_for.png', nrow=input.size(0))

            with torch.no_grad():
                YLL_perC = y_likelihood.clamp(
                    1e-9).log2().mean(dim=(2, 3)).neg()
                fig = plt.figure()
                check_range(YLL_perC, "YLL")
                C = YLL_perC.size(1)
                for YLL in YLL_perC:
                    max_idx = YLL.max(0)[1]
                    plt.bar(torch.arange(C), YLL.data.cpu(),
                            label="maxC="+str(max_idx.item()))
                plt.legend()
                # plt.show()
                plt.savefig(figname+"_YLL.png")
                plt.close(fig)

                nrow = 8
                save_image(code.div(255*2).add(0.5).flatten(0, 1).unsqueeze(
                    1), figname+"_feature.png", nrow=nrow)
                save_image(max_norm(code).add(0.5).flatten(0, 1).unsqueeze(
                    1), figname+"_feature_norm.png", nrow=nrow)
                save_image(y_tilde.div(255*2).add(0.5).flatten(0, 1).unsqueeze(
                    1), figname+"_quant.png", nrow=nrow)
                save_image(max_norm(y_tilde).add(0.5).flatten(0, 1).unsqueeze(
                    1), figname+"_quant_norm.png", nrow=nrow)

            logger.write(check_range(Y_error, "X"+str(self.num_layers-1)))
            logger.write(check_range(code, "code"+str(self.num_layers-1)))
            logger.write(check_range(scale_factor, "SF"))
            logger.write()

        input, code, hyper_code = torch.zeros_like(input), y_tilde, z_tilde
        # input, code, hyper_code = Y_error, y_tilde, z_tilde
        # input = Y_error
        debug(Y_error.shape, code.shape, hyper_code.shape)

        if visual:
            logger.write(check_range(input, "X"+str(self.num_layers-1)))
            logger.write(check_range(code, "code"+str(self.num_layers-1)))
            logger.write()

        # Decode
        if rev_ng:
            with torch.no_grad():
                input, code, jac = self.decode(
                    input, code, jac, rec_code=rec_code, visual=visual, figname=figname)
        else:
            input, code, jac = self.decode(
                input, code, jac, rec_code=rec_code, visual=visual, figname=figname)

        if visual:
            visualizer.plot_queue(figname+'_rev.png', nrow=input.size(0))
            logger.close()

        if self.DQ is not None:
            debug("DQ")
            BDQ = input
            if self.syntax is not None:
                weight, w_likelihood = self.syntax(ori_input - input)
            else:
                weight = None
            input = self.DQ(input, weight)
            if visual:
                save_image(BDQ, figname+"_BDQ.png")
                save_image(input, figname+"_ADQ.png")
        else:
            BDQ = None

        debug('END\n')

        lls = (y_likelihood, z_likelihood)
        if self.AdaptiveQuant is not None:
            lls += (s_likelihood,)
        if self.syntax is not None:
            lls += (w_likelihood,)

        return input, lls, Y_error, jac, code, BDQ


class GoogleHyperSynthesisTransformLight(nn.Sequential):
    def __init__(self, num_features, num_filters, num_hyperpriors):
        super(GoogleHyperSynthesisTransformLight, self).__init__(
            ConvTranspose2d(num_hyperpriors, num_filters,
                            kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters, num_features,
                            kernel_size=3, stride=1)
        )


class RNNEntropyModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_filters):
        super(RNNEntropyModel, self).__init__()

        self.conv_series_1 = nn.Sequential(
            Conv2d(in_channels, num_filters, 3, bias=True),
            nn.ReLU(inplace=True),
            Conv2d(num_filters, num_filters, 3, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv_series_2 = nn.Sequential(
            Conv2d(num_filters, num_filters, 3),
            #ConvTranspose2d(num_filters, num_filters, 3, stride=2)
            nn.ReLU(inplace=True),
            Conv2d(num_filters, out_channels, 3)
        )

        self.rnn_cell = ConvLSTMCell(num_filters, num_filters, (3, 3), True)

    def forward(self, inputs, state=None):
        if state is None:
            state = self.rnn_cell.init_hidden(inputs.size(0), inputs.size()[-2:])
        x = self.conv_series_1(inputs)
        state = self.rnn_cell(x, state)
        x = self.conv_series_2(state[0])

        return x, state


from FrEIA.modules import IRevNetDownsampling

class GroupContextHPCoder(GoogleHyperPriorCoder):
    def __init__(self, **kwargs):
        super(GroupContextHPCoder, self).__init__(**kwargs)

        #self.rnn_unit = RNNEntropyModel(kwargs['num_features'] * 2, kwargs['num_features'] * 2, 96)
        self.rnn_unit = RNNEntropyModel(kwargs['num_features'] * 2, kwargs['num_features'] * 2, kwargs['num_filters'])
        self.f_downsampler = IRevNetDownsampling([[kwargs['num_features']]], legacy_backend=True)
        #self.c_downsampler = IRevNetDownsampling([[kwargs['num_features']]])

        self.hyper_synthesis = GoogleHyperSynthesisTransformLight(kwargs['num_features'],
                                                                  kwargs['num_filters'],
                                                                  kwargs['num_hyperpriors'])
        #self.hyper_synthesis = GoogleHyperSynthesisTransformLight(kwargs['num_features'],
        #                                                          96,
        #                                                          kwargs['num_hyperpriors'])

    def compress(self, input, return_hat=False):
        raise NotImplementedError

    def decompress(self, strings, shape):
        raise NotImplementedError

    def forward(self, input, group=4):
        features = self.analysis(input)

        hyperpriors = self.hyper_analysis(features.abs() if self.use_abs else features)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyperpriors)

        condition = self.hyper_synthesis(z_tilde)

        hidden_states = None
        #condition = self.c_downsampler((condition,))[0][0]
        features = torch.chunk(self.f_downsampler((features,))[0][0], group, dim=1)
        y_tilde = []
        y_likelihood = []

        for idx in range(group):
            if len(y_tilde) == 0:
                #context_input = torch.zeros_like(condition[:, :condition.size(1)//group])
                context_input = torch.zeros_like(condition)
            else:
                context_input = y_tilde[-1]
            
            local_condition, hidden_states = self.rnn_unit(torch.cat([context_input, condition], dim=1), hidden_states)
            local_y_tilde, local_y_likelihood = self.conditional_bottleneck(features[idx], condition=local_condition)
            y_tilde.append(local_y_tilde)
            y_likelihood.append(local_y_likelihood)

        y_tilde = self.f_downsampler((torch.cat(y_tilde, dim=1),), rev=True)[0][0]
        y_likelihood = self.f_downsampler((torch.cat(y_likelihood, dim=1),), rev=True)[0][0]

        reconstructed = self.synthesis(y_tilde)

        return reconstructed, (y_likelihood, z_likelihood)

class AugmentedNormalizedAnalysisTransformS(AugmentedNormalizedFlow):
    def __init__(self, in_channels, num_features, num_filters, kernel_size, use_code, distribution, gdn_mode, use_attn=False, integerlize=False):
        super(AugmentedNormalizedAnalysisTransformS, self).__init__(
            Conv2d(in_channels, num_filters, kernel_size, stride=1),
            ANFNorm(num_filters, mode=gdn_mode),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            ANFNorm(num_filters, mode=gdn_mode),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            ANFNorm(num_filters, mode=gdn_mode),
            Conv2d(num_filters, num_features *
                   (2 if use_code else 1), kernel_size, stride=2),
            AttentionBlock(num_features *
                           (2 if use_code else 1), non_local=True) if use_attn else nn.Identity(),
            use_code=use_code, transpose=False, distribution=distribution, integerlize=integerlize
        )

class AugmentedNormalizedSynthesisTransformS(AugmentedNormalizedFlow):
    def __init__(self, out_channels, num_features, num_filters, kernel_size, use_code, distribution, gdn_mode, use_attn=False, integerlize=False):
        super(AugmentedNormalizedSynthesisTransformS, self).__init__(
            AttentionBlock(
                num_features, non_local=True) if use_attn else nn.Identity(),
            ConvTranspose2d(num_features, num_filters,
                            kernel_size, stride=2),
            ANFNorm(num_filters, inverse=True, mode=gdn_mode),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size, stride=2),
            ANFNorm(num_filters, inverse=True, mode=gdn_mode),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size, stride=2),
            ANFNorm(num_filters, inverse=True, mode=gdn_mode),
            ConvTranspose2d(num_filters, out_channels *
                            (2 if use_code else 1), kernel_size, stride=1),
            use_code=use_code, transpose=True, distribution=distribution, integerlize=integerlize
        )

class AugmentedNormalizedFlowHyperPriorCoder420(HyperPriorCoder):
    """AugmentedNormalizedFlowHyperPriorCoder420"""

    def __init__(self, num_filters, num_features, num_hyperpriors, gdn_mode="standard",
                 in_channels=6, out_channels=6, kernel_size=5, num_layers=1,
                 init_code='gaussian', use_DQ=False, share_wei=False, use_code=True, dec_add=False,
                 use_attn=False, use_QE=False,
                 hyper_filters=192, use_mean=False, use_context=False,
                 condition='Gaussian', quant_mode='noise'):
        super(AugmentedNormalizedFlowHyperPriorCoder420, self).__init__(
            num_features, num_hyperpriors, use_mean, False, use_context, condition, quant_mode)
        self.num_layers = num_layers
        self.share_wei = share_wei
        if not isinstance(num_filters, list):
            num_filters = [num_filters]
        if len(num_filters) != num_layers:
            num_filters = [num_filters[0]] * num_layers

        if not share_wei:
            self.__delattr__('analysis')
            self.__delattr__('synthesis')

            for i in range(num_layers):
                self.add_module('analysis'+str(i), AugmentedNormalizedAnalysisTransformS(
                    in_channels, num_features, num_filters[i], kernel_size, use_code=use_code and init_code != 'zeros', distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn and i == num_layers-1))
                self.add_module('synthesis'+str(i), AugmentedNormalizedSynthesisTransformS(
                    in_channels, num_features, num_filters[i], kernel_size, use_code=use_code and i != num_layers-1 and not dec_add, distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn and i == num_layers-1))

                if use_QE:
                    self.add_module(
                        'QE'+str(i), DeQuantizationModule(in_channels, in_channels, 64, 2))
        else:
            self.analysis = AugmentedNormalizedAnalysisTransformS(
                in_channels, num_features, num_filters[0], kernel_size, use_code=use_code, distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn)
            self.synthesis = AugmentedNormalizedSynthesisTransformS(
                in_channels, num_features, num_filters[0], kernel_size, use_code=use_code, distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn)

        self.hyper_analysis = GoogleHyperAnalysisTransform(
            num_features, hyper_filters, num_hyperpriors) if not use_attn else GoogleHyperAnalysisTransform2(num_features, hyper_filters, num_hyperpriors)

        if use_context:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features*2, hyper_filters, num_hyperpriors) if not use_attn else GoogleHyperSynthesisTransform2(num_features*2, hyper_filters, num_hyperpriors)
        elif self.use_mean or "Mixture" in condition:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features*self.conditional_bottleneck.condition_size, hyper_filters, num_hyperpriors) if not use_attn else GoogleHyperSynthesisTransform2(num_features*self.conditional_bottleneck.condition_size, hyper_filters, num_hyperpriors)
        else:
            self.hyper_synthesis = GoogleHyperScaleSynthesisTransform(
                num_features, hyper_filters, num_hyperpriors)

        for name, m in self.named_children():
            if "ana" in name or "syn" in name:
                m.name = name

        if use_DQ:
            self.DQ = DeQuantizationModule(in_channels, in_channels, 64, 6)
        else:
            self.DQ = None

    def __getitem__(self, key):
        return self.__getattr__(key[:-1] if self.share_wei else key)

    def encode(self, input, code=None, jac=None, visual=False, figname=''):
        codes = []
        for i in range(self.num_layers):
            debug('F', i)
            _, code, jac = self['analysis'+str(i)](
                input, code, jac, layer=i, visual=visual, figname=figname+"_"+str(i))

            if i < self.num_layers-1:
                debug('S', i)
                input, _, jac = self['synthesis'+str(i)](
                    input, code, jac, layer=i, visual=visual, figname=figname+"_"+str(i))

            if visual:
                logger.write(check_range(input, "X"+str(i)))
                logger.write(check_range(code, "code"+str(i)))
                logger.write()

                # channel_analysis(code, figname+f"_FCA_{i}.png")
                codes.append(code)
                save_image(code.div(255*2).add(0.5).flatten(0, 1).unsqueeze(
                    1), figname+f"_feature_{i}.png", nrow=8)
                save_image(max_norm(code).add(0.5).flatten(0, 1).unsqueeze(
                    1), figname+f"_feature_norm_{i}.png", nrow=8)

        if visual:
            channel_analysis_queue(codes, figname)

        return input, code, jac

    def decode(self, input, code=None, jac=None, rec_code=False, visual=False, figname=''):
        for i in range(self.num_layers-1, -1, -1):
            debug('rF', i)
            input, _, jac = self['synthesis'+str(i)](
                input, code, jac, rev=True, last_layer=i == self.num_layers-1, layer=i, visual=visual, figname=figname+"_rev_"+str(i))

            if i or rec_code or jac is not None:
                debug('RF', i)
                _, code, jac = self['analysis' +
                                    str(i)](input, code, jac, rev=True, layer=i, visual=visual, figname=figname+"_rev_"+str(i))

            if visual:
                logger.write(check_range(input, "X"+str(i)))
                logger.write(check_range(code, "code"+str(i)))
                logger.write()

        return input, code, jac

    def entropy_model(self, input, code, jac=False):
        # Enrtopy coding
        debug('E')

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition)
        # y_tilde = code

        # Encode distortion
        Y_error, _, jac = self['synthesis'+str(self.num_layers-1)](
            input, y_tilde, jac, last_layer=True, layer=self.num_layers-1)

        return Y_error, y_tilde, z_tilde, y_likelihood, z_likelihood

    def forward(self, input, code=None, jac=None, rec_code=False, rev_ng=False, IDQ=False, detach_enc=False, visual=False, figname=''):
        # Encode
        if IDQ:
            debug("IDQ")
            input = input.add(torch.rand_like(input).sub_(0.5).div_(255.))

        ori_input = input

        if visual:
            visualizer.set_mean_var(input)
            visualizer.queue_visual(input, figname+"_ori.png")
            # fft_visual(input, figname+"_ori.png")
            logger.open(figname+"_stati.txt")
            logger.write(check_range(input, "input"))
            logger.write()

        jac = [] if jac else None
        input, code, jac = self.encode(
            input, code, jac, visual=visual, figname=figname)

        # Enrtopy coding
        debug('E')

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition)
        # y_tilde = code

        # Encode distortion
        Y_error, _, jac = self['synthesis'+str(self.num_layers-1)](
            input, y_tilde, jac, last_layer=True, layer=self.num_layers-1, visual=visual, figname=figname+"_"+str(self.num_layers-1))

        if visual:
            visualizer.plot_queue(figname+'_for.png', nrow=input.size(0))

            with torch.no_grad():
                YLL_perC = y_likelihood[0].clamp_min(
                    1e-9).log2().mean(dim=(1, 2)).neg()
                fig = plt.figure()
                check_range(YLL_perC, "YLL")
                max_idx = YLL_perC.max(0)[1]
                plt.plot(YLL_perC.data.cpu(),
                         label="maxC="+str(max_idx.item()))
                plt.legend()

                plt.savefig(figname+"_YLL.png")
                plt.close(fig)

                nrow = 8
                order = YLL_perC.argsort(dim=0)
                save_image(code[:, order].div(255*2).add(0.5).flatten(0, 1).unsqueeze(
                    1), figname+"_feature.png", nrow=nrow)
                save_image(max_norm(code[:, order]).add(0.5).flatten(0, 1).unsqueeze(
                    1), figname+"_feature_norm.png", nrow=nrow)
                save_image(y_tilde[:, order].div(255*2).add(0.5).flatten(0, 1).unsqueeze(
                    1), figname+"_quant.png", nrow=nrow)
                save_image(max_norm(y_tilde[:, order]).add(0.5).flatten(0, 1).unsqueeze(
                    1), figname+"_quant_norm.png", nrow=nrow)

            logger.write(check_range(Y_error, "X"+str(self.num_layers-1)))
            logger.write(check_range(code, "code"+str(self.num_layers-1)))
            logger.write()

        input, code, hyper_code = torch.zeros_like(input), y_tilde, z_tilde
        # input, code, hyper_code = Y_error, y_tilde, z_tilde
        # input = Y_error
        debug(Y_error.shape, code.shape, hyper_code.shape)

        if visual:
            logger.write(check_range(input, "X"+str(self.num_layers-1)))
            logger.write(check_range(code, "code"+str(self.num_layers-1)))
            logger.write()

        # Decode
        input, code, jac = self.decode(
            input, code, jac, rec_code=rec_code, visual=visual, figname=figname)

        if visual:
            visualizer.plot_queue(figname+'_rev.png', nrow=input.size(0))
            logger.close()

        if self.DQ is not None:
            debug("DQ")
            BDQ = input
            input = self.DQ(input)
            if visual:
                save_image(YUV4202RGB(BDQ), figname+"_BDQ.png")
                save_image(YUV4202RGB(input), figname+"_ADQ.png")
        else:
            BDQ = None

        debug('END\n')

        return input, (y_likelihood, z_likelihood), Y_error, jac, code, BDQ


class GroupContextCondANFIC(CondAugmentedNormalizedFlowHyperPriorCoder):
    def __init__(self, **kwargs):
        super(GroupContextCondANFIC, self).__init__(**kwargs)

        #self.rnn_unit = RNNEntropyModel(kwargs['num_features'] * 2, kwargs['num_features'] * 2, 96)
        self.rnn_unit = RNNEntropyModel(kwargs['num_features'] * 2, kwargs['num_features'] * 2, kwargs['num_filters'])
        self.f_downsampler = IRevNetDownsampling([[kwargs['num_features']]], legacy_backend=True)
        #self.c_downsampler = IRevNetDownsampling([[kwargs['num_features']]])

        self.hyper_synthesis = GoogleHyperSynthesisTransformLight(kwargs['num_features'],
                                                                  kwargs['hyper_filters'],
                                                                  kwargs['num_hyperpriors'])
        #self.hyper_synthesis = GoogleHyperSynthesisTransformLight(kwargs['num_features'],
        #                                                          96,
        #                                                          kwargs['num_hyperpriors'])

    def forward(self, input, code=None, jac=None, rec_code=False, rev_ng=False,
                IDQ=False, detach_enc=False, visual=False, figname='',
                output=None,  # Should assign value when self.output_nought==False
                cond_coupling_input=None  # Should assign value when self.cond_coupling==True
                # When using ANFIC on residual coding, output & cond_coupling_input should both be MC frame
                ):

        if not self.output_nought:
            assert not (output is None), "output should be specified"
        if self.cond_coupling:
            assert not (cond_coupling_input is None), "cond_coupling_input should be specified"

        # Encode
        if IDQ:
            debug("IDQ")
            input = input.add(torch.rand_like(input).sub_(0.5).div_(255.))

        ori_input = input

        if visual:
            save_image(input, figname + "_input.png")
            if not (output is None):
                save_image(output, figname + "_mc_frame.png")
            visualizer.set_mean_var(input)
            visualizer.queue_visual(input, figname + "_ori.png")
            fft_visual(input, figname + "_ori.png")
            logger.open(figname + "_stati.txt")
            logger.write("min\tmedian\tmean\tvar\tmax\n")

        jac = [] if jac else None
        input, code, jac = self.encode(
            input, code, jac, visual=visual, figname=figname, cond_coupling_input=cond_coupling_input)

        # Enrtopy coding
        debug('E')

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition = self.hyper_synthesis(z_tilde)

        hidden_states = None
        #condition = self.c_downsampler((condition,))[0][0]
        code = torch.chunk(self.f_downsampler((code,))[0][0], 4, dim=1)
        y_tilde = []
        y_likelihood = []

        for idx in range(4):
            if len(y_tilde) == 0:
                #context_input = torch.zeros_like(condition[:, :condition.size(1)//4])
                context_input = torch.zeros_like(condition)
            else:
                context_input = y_tilde[-1]

            local_condition, hidden_states = self.rnn_unit(torch.cat([context_input, condition], dim=1), hidden_states)
            local_y_tilde, local_y_likelihood = self.conditional_bottleneck(code[idx], condition=local_condition)
            y_tilde.append(local_y_tilde)
            y_likelihood.append(local_y_likelihood)

        y_tilde = self.f_downsampler((torch.cat(y_tilde, dim=1),), rev=True)[0][0]
        y_likelihood = self.f_downsampler((torch.cat(y_likelihood, dim=1),), rev=True)[0][0]

        # y_tilde = code # No quantize on z2

        # Encode distortion
        # Concat code with condition (MC frame feature)
        # if self.cond_coupling:
        #    cond = self['cond_encoder'+str(self.num_layers-1)](cond_coupling_input)
        #    y_tilde = torch.cat([y_tilde, cond], dim=1)
        # Y_error, _, jac = self['synthesis'+str(self.num_layers-1)](
        x_2, _, jac = self['synthesis' + str(self.num_layers - 1)](
            input, y_tilde, jac, last_layer=True, layer=self.num_layers - 1, visual=visual,
            figname=figname + "_" + str(self.num_layers - 1))

        if visual:
            visualizer.plot_queue(figname + '_for.png', nrow=input.size(0))

            with torch.no_grad():  # Visualize z likelihood
                YLL_perC = y_likelihood[0].clamp_min(
                    1e-9).log2().mean(dim=(1, 2)).neg()
                fig = plt.figure()
                check_range(YLL_perC, "YLL")
                max_idx = YLL_perC.max(0)[1]
                plt.plot(YLL_perC.data.cpu(),
                         label="maxC=" + str(max_idx.item()))
                plt.legend()

                plt.savefig(figname + "_YLL.png")
                plt.close(fig)

            # Print feature ; code: z2, no quant ; y_tilde: \hat z2
            #     nrow = 8
            #     save_image(code.div(255*2).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_feature.png", nrow=nrow)
            #     save_image(max_norm(code).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_feature_norm.png", nrow=nrow)
            #     save_image(y_tilde.div(255*2).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_quant.png", nrow=nrow)
            #     save_image(max_norm(y_tilde).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_quant_norm.png", nrow=nrow)

            # logger.write(check_range(Y_error, "X"+str(self.num_layers-1)))
            logger.write(check_range(x_2, "X" + str(self.num_layers - 1)))
            logger.write(check_range(code, "code" + str(self.num_layers - 1)))
            logger.write()

        if self.entropy_bottleneck.quant_mode == "SGA" and self.training:
            tau = self.entropy_bottleneck.SGA.tau.item() * 0.01
            # input, code, hyper_code = Y_error*tau, y_tilde, z_tilde
            input, code, hyper_code = x_2 * tau, y_tilde, z_tilde
        else:
            # input, code, hyper_code = torch.zeros_like(input), y_tilde, z_tilde
            # input, code, hyper_code = x_2, y_tilde, z_tilde # x_2 directly backward
            input, code, hyper_code = output, y_tilde, z_tilde  # Correct setting ; MC frame as x_2 when decoding

        # input = Y_error
        # debug(Y_error.shape, code.shape, hyper_code.shape)
        debug(x_2.shape, code.shape, hyper_code.shape)

        if visual:
            logger.write(check_range(input, "X" + str(self.num_layers - 1)))
            logger.write(check_range(code, "code" + str(self.num_layers - 1)))
            logger.write()

        # Decode
        input, code, jac = self.decode(
            input, code, jac, rec_code=rec_code, visual=visual, figname=figname,
            cond_coupling_input=cond_coupling_input)

        if visual:
            visualizer.plot_queue(figname + '_rev.png', nrow=input.size(0))
            logger.close()

        # Quality enhancement
        if self.DQ is not None:
            debug("DQ")
            BDQ = input
            input = self.DQ(input)
            if visual:
                save_image(x_2, figname + "_x_2.png")
                save_image(BDQ, figname + "_BDQ.png")
                save_image(input, figname + "_ADQ.png")
        else:
            BDQ = None

        debug('END\n')

        # return input, (y_likelihood, z_likelihood), Y_error, jac, code, BDQ
        return input, (y_likelihood, z_likelihood), x_2, jac, code, BDQ


class CondAugmentedNormalizedFlowHyperPriorCoderPredPrior(CondAugmentedNormalizedFlowHyperPriorCoder):
    def __init__(self, in_channels_predprior=3, num_predprior_filters=None, **kwargs):
        super(CondAugmentedNormalizedFlowHyperPriorCoderPredPrior, self).__init__(**kwargs)

        if num_predprior_filters is None:  # When not specifying, it will align to num_filters
            num_predprior_filters = kwargs['num_filters']

        if self.use_mean or "Mixture" in kwargs["condition"]:
            self.pred_prior = GoogleAnalysisTransform(in_channels_predprior,
                                                      kwargs[
                                                          'num_features'] * self.conditional_bottleneck.condition_size,
                                                      num_predprior_filters,  # num_filters=64,
                                                      kwargs['kernel_size'],  # kernel_size=3,
                                                      simplify_gdn=False)
            self.PA = nn.Sequential(
                nn.Conv2d((kwargs['num_features'] * self.conditional_bottleneck.condition_size) * 2, 640, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(640, 640, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(640, kwargs['num_features'] * self.conditional_bottleneck.condition_size, 1)
            )
        else:
            self.pred_prior = GoogleAnalysisTransform(in_channels_predprior,
                                                      kwargs['num_features'],
                                                      num_predprior_filters,  # num_filters=64,
                                                      kwargs['kernel_size'],  # kernel_size=3,
                                                      simplify_gdn=False)
            self.PA = nn.Sequential(
                nn.Conv2d(kwargs['num_features'] * 2, 640, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(640, 640, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(640, kwargs['num_features'], 1)
            )

    def forward(self, input, code=None, jac=None, rec_code=False, rev_ng=False,
                IDQ=False, detach_enc=False, visual=False, figname='',
                output=None,  # Should assign value when self.output_nought==False
                cond_coupling_input=None,  # Should assign value when self.cond_coupling==True
                pred_prior_input=None  # cond_coupling_input will replace this when None
                ):

        if not self.output_nought:
            assert not (output is None), "output should be specified"
        if self.cond_coupling:
            assert not (cond_coupling_input is None), "cond_coupling_input should be specified"
        if pred_prior_input is None:
            pred_prior_input = cond_coupling_input

        # Encode
        if IDQ:
            debug("IDQ")
            input = input.add(torch.rand_like(input).sub_(0.5).div_(255.))

        ori_input = input

        if visual:
            save_image(input, figname + "_input.png")
            if not (output is None):
                save_image(output, figname + "_mc_frame.png")
            visualizer.set_mean_var(input)
            visualizer.queue_visual(input, figname + "_ori.png")
            fft_visual(input, figname + "_ori.png")
            logger.open(figname + "_stati.txt")
            logger.write("min\tmedian\tmean\tvar\tmax\n")

        jac = [] if jac else None
        input, code, jac = self.encode(
            input, code, jac, visual=visual, figname=figname, cond_coupling_input=cond_coupling_input)

        # Enrtopy coding
        debug('E')

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        hp_feat = self.hyper_synthesis(z_tilde)
        pred_feat = self.pred_prior(pred_prior_input)

        condition = self.PA(torch.cat([hp_feat, pred_feat], dim=1))

        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition)

        # y_tilde = code # No quantize on z2

        # Encode distortion
        # Concat code with condition (MC frame feature)
        # if self.cond_coupling:
        #    cond = self['cond_encoder'+str(self.num_layers-1)](cond_coupling_input)
        #    y_tilde = torch.cat([y_tilde, cond], dim=1)
        # Y_error, _, jac = self['synthesis'+str(self.num_layers-1)](
        x_2, _, jac = self['synthesis' + str(self.num_layers - 1)](
            input, y_tilde, jac, last_layer=True, layer=self.num_layers - 1, visual=visual,
            figname=figname + "_" + str(self.num_layers - 1))

        if visual:
            visualizer.plot_queue(figname + '_for.png', nrow=input.size(0))

            with torch.no_grad():  # Visualize z likelihood
                YLL_perC = y_likelihood[0].clamp_min(
                    1e-9).log2().mean(dim=(1, 2)).neg()
                fig = plt.figure()
                check_range(YLL_perC, "YLL")
                max_idx = YLL_perC.max(0)[1]
                plt.plot(YLL_perC.data.cpu(),
                         label="maxC=" + str(max_idx.item()))
                plt.legend()

                plt.savefig(figname + "_YLL.png")
                plt.close(fig)

            # Print feature ; code: z2, no quant ; y_tilde: \hat z2
            #     nrow = 8
            #     save_image(code.div(255*2).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_feature.png", nrow=nrow)
            #     save_image(max_norm(code).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_feature_norm.png", nrow=nrow)
            #     save_image(y_tilde.div(255*2).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_quant.png", nrow=nrow)
            #     save_image(max_norm(y_tilde).add(0.5).flatten(0, 1).unsqueeze(
            #         1), figname+"_quant_norm.png", nrow=nrow)

            # logger.write(check_range(Y_error, "X"+str(self.num_layers-1)))
            logger.write(check_range(x_2, "X" + str(self.num_layers - 1)))
            logger.write(check_range(code, "code" + str(self.num_layers - 1)))
            logger.write()

        if self.entropy_bottleneck.quant_mode == "SGA" and self.training:
            tau = self.entropy_bottleneck.SGA.tau.item() * 0.01
            # input, code, hyper_code = Y_error*tau, y_tilde, z_tilde
            input, code, hyper_code = x_2 * tau, y_tilde, z_tilde
        else:
            # input, code, hyper_code = torch.zeros_like(input), y_tilde, z_tilde
            # input, code, hyper_code = x_2, y_tilde, z_tilde # x_2 directly backward
            input, code, hyper_code = output, y_tilde, z_tilde  # Correct setting ; MC frame as x_2 when decoding

        # input = Y_error
        # debug(Y_error.shape, code.shape, hyper_code.shape)
        debug(x_2.shape, code.shape, hyper_code.shape)

        if visual:
            logger.write(check_range(input, "X" + str(self.num_layers - 1)))
            logger.write(check_range(code, "code" + str(self.num_layers - 1)))
            logger.write()

        # Decode
        input, code, jac = self.decode(
            input, code, jac, rec_code=rec_code, visual=visual, figname=figname,
            cond_coupling_input=cond_coupling_input)

        if visual:
            visualizer.plot_queue(figname + '_rev.png', nrow=input.size(0))
            logger.close()

        # Quality enhancement
        if self.DQ is not None:
            debug("DQ")
            BDQ = input
            input = self.DQ(input)
            if visual:
                save_image(x_2, figname + "_x_2.png")
                save_image(BDQ, figname + "_BDQ.png")
                save_image(input, figname + "_ADQ.png")
        else:
            BDQ = None

        debug('END\n')

        # return input, (y_likelihood, z_likelihood), Y_error, jac, code, BDQ
        return input, (y_likelihood, z_likelihood), x_2, jac, code, BDQ


class CondAugmentedNormalizedFlowHyperPriorCoderPredPrior1LConcat(HyperPriorCoder):
    """CondAugmentedNormalizedFlowHyperPriorCoderPredPrior1LConcat"""

    def __init__(self, num_filters, num_features, num_hyperpriors, hyper_filters=192, simplify_gdn=False,
                 in_channels=3, out_channels=3, kernel_size=5, use_mean=False, use_context=False,
                 condition='Gaussian', quant_mode='noise', 
                 in_channels_predprior=3,
                 num_predprior_filters=None):
        super(CondAugmentedNormalizedFlowHyperPriorCoderPredPrior1LConcat, self).__init__(
            num_features, num_hyperpriors, use_mean, False, use_context, condition, quant_mode)

        self.analysis = GoogleAnalysisTransform(
            in_channels*2, num_features, num_filters, kernel_size, simplify_gdn)

        self.synthesis = GoogleSynthesisTransform(
            out_channels, num_features, num_filters, kernel_size, simplify_gdn)

        self.hyper_analysis = GoogleHyperAnalysisTransform(
            num_features, num_filters, num_hyperpriors)

        if self.use_mean:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features*self.conditional_bottleneck.condition_size, hyper_filters, num_hyperpriors)
        else:
            self.hyper_synthesis = GoogleHyperScaleSynthesisTransform(
                num_features, hyper_filters, num_hyperpriors)
         
        if num_predprior_filters is None: # When not specifying, it will align to num_filters
            num_predprior_filters = num_filters

        self.pred_prior = GoogleAnalysisTransform(in_channels_predprior,
                                                  num_features*self.conditional_bottleneck.condition_size,
                                                  num_predprior_filters,
                                                  kernel_size,
                                                  simplify_gdn=False
                                                 )
        self.PA = nn.Sequential(
                    nn.Conv2d((num_features*self.conditional_bottleneck.condition_size)*2, 640, 1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(640, 640, 1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(640, num_features*self.conditional_bottleneck.condition_size, 1)
                  )       

        self.synthesis2 = nn.Sequential(
                            Conv2d(6, 64, 3, stride=1),
                            nn.ReLU(inplace=True),
                            ResidualBlock(64), 
                            ResidualBlock(64), 
                            Conv2d(64, 3, 3, stride=1),
                          )
        self.DQ = DeQuantizationModule(in_channels, in_channels, 64, 6)

    def forward(self, input, output=None, cond_coupling_input=None, pred_prior_input=None):

        assert not (output is None), "output should be specified"
        assert not (cond_coupling_input is None), "cond_coupling_input should be specified"
        if pred_prior_input is None:
            pred_prior_input = cond_coupling_input
        
        features = self.analysis(torch.cat([input, cond_coupling_input], dim=1))

        hyperpriors = self.hyper_analysis(
            features.abs() if self.use_abs else features)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyperpriors)
        
        hp_feat = self.hyper_synthesis(z_tilde)
        pred_feat = self.pred_prior(pred_prior_input)
        
        condition = self.PA(torch.cat([hp_feat, pred_feat], dim=1))

        y_tilde, y_likelihood = self.conditional_bottleneck(
            features, condition=condition)

        decoded = self.synthesis(y_tilde)

        reconstructed = self.synthesis2(torch.cat([decoded, output], dim=1))

        reconstructed = self.DQ(reconstructed)

        return reconstructed, (y_likelihood, z_likelihood), None, None, None, decoded

# Google transforms with 2 shortcut blocks
class ShortCutAnalysisTransform(nn.Sequential):
    """ShortCutAnalysisTransform"""

    def __init__(self, in_channels, num_filters, num_features, kernel_size, simplify_gdn):
        super(ShortCutAnalysisTransform, self).__init__(
            Conv2d(in_channels, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, simplify=simplify_gdn),
            ShortCutBlock(num_filters, num_filters, kernel_size),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, simplify=simplify_gdn),
            ShortCutBlock(num_filters, num_filters, kernel_size),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, simplify=simplify_gdn),
            Conv2d(num_filters, num_features, kernel_size, stride=2)
        )


class ShortCutSynthesisTransform(nn.Sequential):
    def __init__(self, out_channels, num_features, num_filters, kernel_size, simplify_gdn):
        super(ShortCutSynthesisTransform, self).__init__(
            ConvTranspose2d(num_features, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(
                num_filters, inverse=True, simplify=simplify_gdn),
            ConvTranspose2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(
                num_filters, inverse=True, simplify=simplify_gdn),
            TransposedShortCutBlock(num_filters, num_filters, kernel_size),
            ConvTranspose2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(
                num_filters, inverse=True, simplify=simplify_gdn),
            TransposedShortCutBlock(num_filters, num_filters, kernel_size),
            ConvTranspose2d(num_filters, out_channels, kernel_size, stride=2)
        )


class ShortCutHyperPriorCoder(HyperPriorCoder):
    """ShortCutHyperPriorCoder"""

    def __init__(self, num_filters, num_features, num_hyperpriors, simplify_gdn=False,
                 in_channels=3, out_channels=3, kernel_size=5, use_mean=False, use_abs=False, use_context=False,
                 condition='Gaussian', quant_mode='noise'):
        super(ShortCutHyperPriorCoder, self).__init__(
            num_features, num_hyperpriors, use_mean, use_abs, use_context, condition, quant_mode)
        
        self.analysis = ShortCutAnalysisTransform(
            in_channels, num_features, num_filters, kernel_size, simplify_gdn)

        self.synthesis = ShortCutSynthesisTransform(
            out_channels, num_features, num_filters, kernel_size, simplify_gdn)

        self.hyper_analysis = GoogleHyperAnalysisTransform(
            num_features, num_filters, num_hyperpriors)

        if self.use_mean:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features*self.conditional_bottleneck.condition_size, num_filters, num_hyperpriors)
        else:
            self.hyper_synthesis = GoogleHyperScaleSynthesisTransform(
                num_features, num_filters, num_hyperpriors)


# Transforms with 2 shortcut blocks
class AugmentedNormalizedShortCutAnalysisTransform(AugmentedNormalizedFlow):
    def __init__(self, in_channels, num_features, num_filters, kernel_size, use_code, distribution, gdn_mode, use_attn=False, integerlize=False):
        super(AugmentedNormalizedShortCutAnalysisTransform, self).__init__(
            Conv2d(in_channels, num_filters, kernel_size, stride=2),
            ANFNorm(num_filters, mode=gdn_mode),
            ShortCutBlock(num_filters, num_filters, kernel_size),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            ANFNorm(num_filters, mode=gdn_mode),
            ShortCutBlock(num_filters, num_filters, kernel_size),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            ANFNorm(num_filters, mode=gdn_mode),
            Conv2d(num_filters, num_features *
                   (2 if use_code else 1), kernel_size, stride=2),
            AttentionBlock(num_features *
                           (2 if use_code else 1), non_local=True) if use_attn else nn.Identity(),
            use_code=use_code, transpose=False, distribution=distribution, integerlize=integerlize
        )


class AugmentedNormalizedShortCutSynthesisTransform(AugmentedNormalizedFlow):
    def __init__(self, out_channels, num_features, num_filters, kernel_size, use_code, distribution, gdn_mode, use_attn=False, integerlize=False):
        super(AugmentedNormalizedShortCutSynthesisTransform, self).__init__(
            AttentionBlock(
                num_features, non_local=True) if use_attn else nn.Identity(),
            ConvTranspose2d(num_features, num_filters,
                            kernel_size, stride=2),
            ANFNorm(num_filters, inverse=True, mode=gdn_mode),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size, stride=2),
            ANFNorm(num_filters, inverse=True, mode=gdn_mode),
            TransposedShortCutBlock(num_filters, num_filters, kernel_size),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size, stride=2),
            ANFNorm(num_filters, inverse=True, mode=gdn_mode),
            TransposedShortCutBlock(num_filters, num_filters, kernel_size),
            ConvTranspose2d(num_filters, out_channels *
                            (2 if use_code else 1), kernel_size, stride=2),
            use_code=use_code, transpose=True, distribution=distribution, integerlize=integerlize
        )


class CondAugmentedNormalizedFlowShortCutCoderPredPrior(CondAugmentedNormalizedFlowHyperPriorCoderPredPrior):
    def __init__(self, **kwargs):
        super(CondAugmentedNormalizedFlowShortCutCoderPredPrior, self).__init__(**kwargs)

        if not isinstance(kwargs['num_filters'], list):
            num_filters = [kwargs['num_filters']]
        if len(num_filters) != kwargs['num_layers']:
            num_filters = [num_filters[0]] * kwargs['num_layers']

        in_channels     = kwargs['in_channels']
        num_features    = kwargs['num_features']
        kernel_size     = kwargs['kernel_size']
        use_code        = kwargs['use_code']
        dec_add         = kwargs['dec_add']
        init_code       = kwargs['init_code']
        gdn_mode        = kwargs['gdn_mode']
        use_attn        = kwargs['use_attn']
        num_layers      = kwargs['num_layers']
        share_wei       = kwargs['share_wei']
        num_cond_frames = kwargs['num_cond_frames']

        if not share_wei:
             for i in range(num_layers):
                self.__delattr__('analysis'+str(i))
                self.__delattr__('synthesis'+str(i))
            
                if not self.cond_coupling:
                    self.add_module('analysis'+str(i), AugmentedNormalizedShortCutAnalysisTransform(
                        in_channels, num_features, num_filters[i], kernel_size, 
                        use_code=use_code and init_code != 'zeros', 
                        distribution=init_code, gdn_mode=gdn_mode, 
                        use_attn=use_attn and i == num_layers-1))
                    self.add_module('synthesis'+str(i), AugmentedNormalizedShortCutSynthesisTransform(
                        in_channels, num_features, num_filters[i], kernel_size, 
                        use_code=use_code and i != num_layers-1 and not dec_add, 
                        distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn and i == num_layers-1))
                else: 
                    self.add_module('analysis'+str(i), AugmentedNormalizedShortCutAnalysisTransform(
                        in_channels*(1+num_cond_frames), num_features, num_filters[i], kernel_size, 
                        use_code=use_code and init_code != 'zeros', 
                        distribution=init_code, gdn_mode=gdn_mode, 
                        use_attn=use_attn and i == num_layers-1))
                    self.add_module('synthesis'+str(i), AugmentedNormalizedShortCutSynthesisTransform(
                        in_channels, num_features, num_filters[i], kernel_size, 
                        use_code=use_code and i != num_layers-1 and not dec_add, 
                        distribution=init_code, gdn_mode=gdn_mode, use_attn=use_attn and i == num_layers-1))
        
        if kwargs['num_predprior_filters'] is None:  
            num_predprior_filters = kwargs['num_filters']
        else:
            num_predprior_filters = kwargs['num_predprior_filters']

        if self.use_mean or "Mixture" in kwargs["condition"]:
            self.pred_prior = ShortCutAnalysisTransform(
                                                      kwargs['in_channels_predprior'],
                                                      kwargs['num_features'] * self.conditional_bottleneck.condition_size,
                                                      num_predprior_filters,
                                                      kwargs['kernel_size'],
                                                      simplify_gdn=False)
        else:
            self.pred_prior = ShortCutAnalysisTransform(
                                                      kwargs['in_channels_predprior'],
                                                      kwargs['num_features'],
                                                      num_predprior_filters,
                                                      kwargs['kernel_size'],
                                                      simplify_gdn=False)


__CODER_TYPES__ = {"GoogleFactorizedCoder": GoogleFactorizedCoder, "GoogleIDFPriorCoder": GoogleIDFPriorCoder,
                   "GoogleHyperPriorCoder": GoogleHyperPriorCoder, "GoogleHyperPriorCoder2": GoogleHyperPriorCoder2,
                   "GoogleHyperPriorCoderYUV": GoogleHyperPriorCoderYUV,
                   "ANFHyperPriorCoder": AugmentedNormalizedFlowHyperPriorCoder,
                   "CondANFHyperPriorCoder": CondAugmentedNormalizedFlowHyperPriorCoder,
                   "CondANFHyperPriorCoder2": CondAugmentedNormalizedFlowHyperPriorCoder2,
                   "CondANFHyperPriorCoderYUV": CondAugmentedNormalizedFlowHyperPriorCoderYUV,
                   "SFTCondANFHyperPriorCoder": SFTCondAugmentedNormalizedFlowHyperPriorCoder,
                   "Coarse2FineHyperPriorCoder": Coarse2FineHyperPriorCoder,
                   "GoogleContextCoder": GoogleContextCoder, "GoogleContextCoder2": GoogleContextCoder2, 
                   "CSTKContextCoder": CSTKContextCoder,
                   "GroupContextHPCoder": GroupContextHPCoder, "GroupContextCondANFIC": GroupContextCondANFIC,
                   "CondANFHyperPriorCoderPredPrior": CondAugmentedNormalizedFlowHyperPriorCoderPredPrior,
                   "CondANFHyperPriorCoderPredPrior1LConcat": CondAugmentedNormalizedFlowHyperPriorCoderPredPrior1LConcat,
                   "CondANFHyperPriorCoderPredPriorYUV": CondAugmentedNormalizedFlowHyperPriorCoderPredPriorYUV,
                   "ShortCutHyperPriorCoder": ShortCutHyperPriorCoder,
                   "CondANFShortCutCoderPredPrior": CondAugmentedNormalizedFlowShortCutCoderPredPrior,
                   }


def get_coder(coder_type, num_features, num_filters, num_hyperpriors=None, simplify_gdn=False,
              use_mean=True, use_abs=False, condition='Gaussian', quant_mode='noise',
              output_nought=True, cond_coupling=False, num_cond_frames=1, use_DQ=True):
    """get_coder"""
    coder = __CODER_TYPES__[
        coder_type] if coder_type in __CODER_TYPES__ else coder_type
    assert coder in __CODER_TYPES__.values()
    from functools import partial
    if coder == GoogleFactorizedCoder:
        return partial(coder, num_features=num_features, num_filters=num_filters,
                       simplify_gdn=simplify_gdn, quant_mode=quant_mode)
    elif coder == AugmentedNormalizedFlowHyperPriorCoder:
        return partial(coder, num_features=num_features, num_filters=num_filters, num_hyperpriors=num_hyperpriors,
                        use_mean=use_mean, condition=condition, quant_mode=quant_mode,
                        num_layers=2, #L=2
                        use_DQ=use_DQ,
                        use_code=False, # Shift only (in affine coupling)
                      )
    elif coder == CondAugmentedNormalizedFlowHyperPriorCoder or coder == CondAugmentedNormalizedFlowHyperPriorCoderPredPrior:
        return partial(coder, num_features=num_features, num_filters=num_filters, num_hyperpriors=num_hyperpriors,
                        use_mean=use_mean, condition=condition, quant_mode=quant_mode,
                        num_layers=2, #L=2
                        use_DQ=use_DQ,
                        use_code=False, # Shift only (in affine coupling)
                        output_nought=output_nought,
                        cond_coupling=cond_coupling,
                        num_cond_frames=num_cond_frames
                      )
    elif coder == CondAugmentedNormalizedFlowHyperPriorCoder2:
        return partial(coder, num_features=num_features, num_filters=num_filters, num_hyperpriors=num_hyperpriors,
                        use_mean=use_mean, condition=condition, quant_mode=quant_mode,
                        num_layers=2, #L=2
                        use_DQ=use_DQ,
                        simplify_gdn=simplify_gdn,
                        use_code=False, # Shift only (in affine coupling)
                        output_nought=output_nought, cond_coupling=cond_coupling
                      )
    elif coder == CondAugmentedNormalizedFlowHyperPriorCoderYUV or coder == CondAugmentedNormalizedFlowHyperPriorCoderPredPriorYUV:
        return partial(coder, num_features=num_features, num_filters=num_filters, num_hyperpriors=num_hyperpriors,
                        use_mean=use_mean, condition=condition, quant_mode=quant_mode,
                        num_layers=2, #L=2
                        use_DQ=use_DQ,
                        use_code=False, # Shift only (in affine coupling)
                        output_nought=output_nought,
                        cond_coupling=cond_coupling,
                        num_cond_frames=num_cond_frames
                      )
    elif coder == SFTCondAugmentedNormalizedFlowHyperPriorCoder:
        return partial(coder, num_features=num_features, num_filters=num_filters, num_hyperpriors=num_hyperpriors,
                        use_mean=use_mean, condition=condition, quant_mode=quant_mode,
                        num_layers=2, #L=2
                        use_DQ=use_DQ,
                        use_code=False, # Shift only (in affine coupling)
                        output_nought=output_nought,
                        num_cond_frames=num_cond_frames
                      )
    elif coder == GroupContextCondANFIC:
        return partial(coder, num_features=num_features, num_filters=num_filters, num_hyperpriors=num_hyperpriors,
                       use_mean=use_mean, condition=condition, quant_mode=quant_mode, hyper_filters=num_filters,
                       num_layers=2,  # L=2
                       use_DQ=use_DQ,
                       use_code=False,  # Shift only (in affine coupling)
                       output_nought=output_nought,
                       cond_coupling=cond_coupling,
                       num_cond_frames=num_cond_frames
                       )
    elif coder == GoogleIDFPriorCoder:
        return partial(coder, num_features=num_features, num_filters=num_filters,
                       simplify_gdn=simplify_gdn, quant_mode=quant_mode)
    else:
        return partial(coder, num_features=num_features, num_filters=num_filters, num_hyperpriors=num_hyperpriors,
                       simplify_gdn=simplify_gdn, use_mean=use_mean, use_abs=use_abs, condition=condition, quant_mode=quant_mode)
