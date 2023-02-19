import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import os

import torch_compression as trc
import torch_compression.torchac.torchac as ac
from torch_compression.modules.ConvRNN import ConvLSTM2d
from torch_compression.modules.functional import space_to_depth, depth_to_space
from torch_compression.modules.MIMT import MIMTEncoder, MIMTDecoder, PatchEmbed, PatchUnEmbed
from torch_compression.util.math import lower_bound

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
        return self.conv2d_forward(input, self.weight*self._mask)


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
        self.condition_size = self.entropy_model.condition_size
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

        if not (self.training or isinstance(self.entropy_model, trc.MixtureModelConditional)):
            output = self.entropy_model.quantize(
                input, self.entropy_model.quant_mode if self.training else "round",
                self.entropy_model.mean
            )
            
        likelihood = self.entropy_model._likelihood(output)

        #if isinstance(self.entropy_model, trc.GaussianConditional):
        #    from torch_compression.modules.entropy_models import estimate_bpp
        #    self.entropy_model._set_condition(condition0)
        #    output0 = self.entropy_model.quantize(
        #        input, self.entropy_model.quant_mode if self.training else "round")
        #    likelihood0 = self.entropy_model._likelihood(output0)
        #    print(
        #        estimate_bpp(likelihood, num_pixels=256*256).mean().item() <= \
        #        estimate_bpp(likelihood0, num_pixels=256*256).mean().item()
        #    )

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


####################################################################


class ResBlocks(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResBlocks, self).__init__(
            ResBlock(in_channels, out_channels, kernel_size),
            ResBlock(out_channels, out_channels, kernel_size),
            ResBlock(out_channels, out_channels, kernel_size)
        )
                
    def forward(self, input):
        return super().forward(input) + input


class ResBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResBlock, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, (kernel_size - 1) // 2), 
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, (kernel_size - 1) // 2), 
        )
    
    def forward(self, input):
        return super().forward(input) + input


from torchvision.utils import make_grid
import seaborn as sns
import matplotlib.pyplot as plt
import math

class ScoreBasedContextModel(nn.Module):
    def __init__(self, num_features, num_phi_features, entropy_model, n=8, num_masks=None, sep_PA=False, pred_condition_res=False):
        super(ScoreBasedContextModel, self).__init__()
        self.num_features = num_features

        assert isinstance(
            entropy_model, trc.SymmetricConditional), type(entropy_model)
        self.entropy_model = entropy_model

        if num_masks is None:
            self.num_masks = num_features
        else:
            self.num_masks = num_masks


        self.sep_PA = sep_PA
        self.pred_condition_res = pred_condition_res
        if self.sep_PA:
            self.PA0 = nn.Sequential(
                nn.Conv2d(num_phi_features, 640, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(640, 640, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(640, num_features * self.entropy_model.condition_size, 1)
            )
            if self.pred_condition_res:
                num_phi_features += num_features * self.entropy_model.condition_size

        self.PA = nn.Sequential(
            nn.Conv2d(num_phi_features + num_features + self.num_masks + 1, num_features * (self.entropy_model.condition_size + 1), 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * (self.entropy_model.condition_size + 1), num_features * (self.entropy_model.condition_size + 1), 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * (self.entropy_model.condition_size + 1), num_features * self.entropy_model.condition_size, 3, 1, 1)
        )

        self.score_est_net =  nn.Sequential(
            nn.Conv2d(num_features * self.entropy_model.condition_size + self.num_masks + 1, 128, 3, 1, 1), # input: mean, scale, mask
            ResBlocks(128, 128, 3),
            nn.Conv2d(128, self.num_masks, 3, 1, 1),
            nn.Sigmoid()
        )
        
        self.n = n
        self.sampler = [math.sin((i*math.pi)/(2*self.n)) for i in range(0, self.n+1)]

    def sample_top_k(self, score, k, mask=None): # for each latents in a batch, select top K elements by score only from those un-masked.
        masked_score = score.where((1-mask).bool(), torch.Tensor([-1.]).to(score.device))

        # Flatten score for topk selection along a batch
        flatten_masked_score = masked_score.flatten(1)
        sample_at_ = torch.topk(flatten_masked_score, k, dim=1, largest=True).indices

        # Mark sampled locations as 1
        sample_at_ = torch.zeros_like(flatten_masked_score).scatter_(1, sample_at_, 1)

        # Reshape back the sampling mask
        sample_at_ = sample_at_.view(score.shape)
        
        return sample_at_

    def get_num_samples(self, i, score):
        num_elements = np.prod(score.shape[1:])
        return int(self.sampler[i]*num_elements) - int(self.sampler[i-1]*num_elements)

    def PA_forward(self, i, priors, ctx, mask, step_map, overall_condition):
        if self.sep_PA:
            if i == 1:
                condition = self.PA0(priors)
                condition0 = condition
                if self.pred_condition_res:
                    overall_condition = condition
            else:
                if self.pred_condition_res:
                    _priors = priors
                    priors = torch.cat([priors, overall_condition], dim=1)
                
                condition = self.PA(torch.cat([priors, ctx, mask, step_map], dim=1))
                
                if self.pred_condition_res:
                    priors = _priors
                    mu_r, sigma_r = condition.chunk(2, dim=1)
                    mu, sigma = overall_condition.chunk(2, dim=1)
                    mu += mu_r
                    sigma *= F.softplus(1 + sigma_r)
                    condition = torch.cat([mu, sigma], dim=1)
        else:
            condition = self.PA(torch.cat([priors, ctx, mask, step_map], dim=1))


    def quant(self, input):
        if self.entropy_model.quant_mode == 'estUN_outR' and self.training:
            def _training_quant(x):
                n = torch.round(x) - x
                n = n.clone().detach()
                return x + n

            if self.entropy_model.mean is not None:
                input_res = input - self.entropy_model.mean
                output = _training_quant(input_res) + self.entropy_model.mean 
            else:
                output = _training_quant(input)
        else:
            output = self.entropy_model.quantize(
                                input, 
                                self.entropy_model.quant_mode if self.training else "round",
                                None if self.training else self.entropy_model.mean
                     )
        return output

    def forward(self, input, condition, visual=False, visual_prefix='./tmp'):
        priors = condition # To ensure the encapsulation, use "condition" as keyword arg but it should be priors

        b, c, h, w = input.shape

        ctx = torch.zeros_like(input)
        mask = torch.zeros((b, self.num_masks, h, w)).to(input.device)
        overall_condition = torch.zeros((b, c*2, h, w)).to(input.device)

        next_ctx = torch.zeros_like(input)
        next_mask = torch.zeros((b, self.num_masks, h, w)).to(input.device)
        next_overall_condition = torch.zeros((b, c*2, h, w)).to(input.device)
        
        for i in range(1, self.n+1):
            ctx = next_ctx
            mask = next_mask
            overall_condition = next_overall_condition

            step_map = torch.ones((b, 1, h, w)).to(input.device) * float(i) / self.n
            condition = self.PA_forward(i, priors, ctx, mask, step_map, overall_condition)
            
            self.entropy_model._set_condition(condition)
            
            score = self.score_est_net(torch.cat([condition, mask, step_map], dim=1))
            
            # Sample top K scored elements
            num_samples = self.get_num_samples(i, score)
            sample_at_ = self.sample_top_k(score, num_samples, mask).to(input.device)
            
            # Update context by which samples selected
            current_quant = self.quant(input)
             
            
            if sample_at_.shape[1] != self.num_features:
                #assert torch.sum(sample_at_ * overall_condition) < 1e-5
                next_overall_condition = mask * overall_condition + (1 - mask) * condition
            else:
                _mask = torch.cat([mask]*2, dim=1)
                next_overall_condition = _mask * overall_condition + (1 - _mask) * condition
            
            # Quantize input with current condition ; mask out unseen context and update
            next_ctx = ctx + sample_at_ * current_quant
            next_mask = mask + sample_at_

            if visual:
                def visual_feat(feat_map, save_name='./tmp/tmp.png'):
                    heatmap = make_grid([feat_map[:, i, :, :] for i in range(feat_map.size(1))], nrow=8)[0, :, :].cpu().numpy()
                    plt.figure()
                    rate_heatmap = sns.heatmap(heatmap, xticklabels=False, yticklabels=False, square=True, cmap='Greys')
                    plt.savefig(save_name, dpi=400)
                    plt.close()
                
                def save_info(prefix, i, name, info):
                    path = os.path.join(visual_prefix, name)
                    os.makedirs(path, exist_ok=True)
                    visual_feat(info, os.path.join(path, 'iter'+str(i)+'.png'))

                save_info(visual_prefix, i, 'updated_mask', next_mask)
                save_info(visual_prefix, i, 'current_mask', sample_at_)
                save_info(visual_prefix, i, 'current_context', sample_at_ * current_quant)
                save_info(visual_prefix, i, 'updated_context', next_ctx)
                save_info(visual_prefix, i, 'current_mean', sample_at_ * condition.chunk(2, dim=1)[0])
                save_info(visual_prefix, i, 'current_scale', sample_at_ * condition.chunk(2, dim=1)[1])
                save_info(visual_prefix, i, 'current_mean_full', condition.chunk(2, dim=1)[0])
                save_info(visual_prefix, i, 'current_scale_full', condition.chunk(2, dim=1)[1])
                save_info(visual_prefix, i, 'updated_mean', next_overall_condition.chunk(2, dim=1)[0])
                save_info(visual_prefix, i, 'updated_scale', next_overall_condition.chunk(2, dim=1)[1])
                save_info(visual_prefix, i, 'prev_mean_current_masked', sample_at_ * overall_condition.chunk(2, dim=1)[0])
                save_info(visual_prefix, i, 'prev_scale_current_masked', sample_at_ * overall_condition.chunk(2, dim=1)[1])

        
        assert torch.sum(next_mask.float() - torch.ones_like(next_mask)) < 1e-5
        output = next_ctx
        self.entropy_model._set_condition(next_overall_condition)

        if self.entropy_model.quant_mode == 'estUN_outR' and self.training:
            output_noise = self.entropy_model.quantize(input, 'noise')
            likelihood = self.entropy_model._likelihood(output_noise)
        else:
            likelihood = self.entropy_model._likelihood(output)
        

        return output, likelihood


class GetEntropyNet(nn.Module):
    def __init__(self, entropy_model, num_masks=1):
        super().__init__()
        self.entropy_model = entropy_model

        self.num_masks = num_masks
    
    def forward(self, input):
        condition, _ = input[:, :-(self.num_masks+1), :, :], input[:, -(self.num_masks+1): , :, :]
        mean, scale = condition.chunk(2, dim=1)
        entropy = -self.entropy_model._likelihood(mean).log2()

        if self.num_masks == 1:
            entropy = entropy.mean(dim=1, keepdim=True)
        return entropy


class EntropyFirstContextModel(ScoreBasedContextModel):
    def __init__(self, *args, **kwargs):
        super(EntropyFirstContextModel, self).__init__(*args, **kwargs)
        self.score_est_net = GetEntropyNet(self.entropy_model, self.num_masks) 

    def sample_top_k(self, score, k, mask=None): # for each latents in a batch, select top K elements by score only from those un-masked.
        masked_score = score.where((1-mask).bool(), torch.Tensor([np.inf]).to(score.device))

        # Flatten score for topk selection along a batch
        flatten_masked_score = masked_score.flatten(1)
        sample_at_ = torch.topk(flatten_masked_score, k, dim=1, largest=False).indices

        # Mark sampled locations as 1
        sample_at_ = torch.zeros_like(flatten_masked_score).scatter_(1, sample_at_, 1)

        # Reshape back the sampling mask
        sample_at_ = sample_at_.view(score.shape)
        
        return sample_at_


class MIMTCORE(nn.Module):
    def __init__(self, num_features, num_phi_features, entropy_model, n=8, num_masks=None, sep_PA=False, pred_condition_res=False):
        super(MIMTCORE, self).__init__()
        self.num_features = num_features
        assert isinstance(entropy_model, trc.SymmetricConditional), type(entropy_model)
        self.entropy_model = entropy_model

        if num_masks is None:
            self.num_masks = num_features
        else:
            self.num_masks = num_masks

        self.n = n
        self.sampler = [math.sin((i*math.pi)/(2*self.n)) for i in range(0, self.n+1)]

    def get_num_samples(self, i, score):
        num_elements = np.prod(score.shape[1:])
        return int(self.sampler[i]*num_elements) - int(self.sampler[i-1]*num_elements)

    def sample_top_k(self, score, k, mask=None): # for each latents in a batch, select top K elements by score only from those un-masked.
        masked_score = score.where((1-mask).bool(), torch.Tensor([np.inf]).to(score.device))

        # Flatten score for topk selection along a batch
        flatten_masked_score = masked_score.flatten(1)
        sample_at_ = torch.topk(flatten_masked_score, k, dim=1, largest=False).indices

        # Mark sampled locations as 1
        sample_at_ = torch.zeros_like(flatten_masked_score).scatter_(1, sample_at_, 1)

        # Reshape back the sampling mask
        sample_at_ = sample_at_.view(score.shape)
        
        return sample_at_

    def quant(self, input):
        if self.entropy_model.quant_mode == 'estUN_outR' and self.training:
            def _training_quant(x):
                n = torch.round(x) - x
                n = n.clone().detach()
                return x + n

            if self.entropy_model.mean is not None:
                input_res = input - self.entropy_model.mean
                output = _training_quant(input_res) + self.entropy_model.mean 
            else:
                output = _training_quant(input)
        else:
            output = self.entropy_model.quantize(
                                input, 
                                self.entropy_model.quant_mode if self.training else "round",
                                None if self.training else self.entropy_model.mean
                     )
        return output


class MIMTContextModel(MIMTCORE):
    def __init__(self, num_priors=2, training_mask_ratio=1./math.e, *args, **kwargs):
        super(MIMTContextModel, self).__init__(*args, **kwargs)
        
        # delattr(self, 'PA')
        self.num_priors = num_priors
        self.training_mask_ratio = training_mask_ratio
        assert self.training_mask_ratio >= 0. and self.training_mask_ratio <= 1., ValueError

        self.MIMT_encoder = MIMTEncoder(input_dim=self.num_features*2, l_dim=768,
                                        num_heads=8, window_size=4,
                                        mlp_ratio=1., 
                                        qkv_bias=True, qk_scale=None, 
                                        drop=0., attn_drop=0.,drop_path=0., norm_layer=nn.LayerNorm, 
                                        use_checkpoint=False, 
                                        shift_first=False, num_sep=2, num_joint=4)

        self.MIMT_decoder = MIMTDecoder(input_dim=self.num_features, l_dim=768, 
                                        num_heads=8, window_size=4,
                                        mlp_ratio=1., 
                                        qkv_bias=True, qk_scale=None, 
                                        drop=0., attn_drop=0.,drop_path=0., norm_layer=nn.LayerNorm, 
                                        use_checkpoint=False, 
                                        shift_first=False, num_dec=2)
        
        
    def PA_forward(self, i, priors, ctx, mask, step_map, overall_condition):
        priors = priors.chunk(self.num_priors, dim=1)
        
        joint = self.MIMT_encoder(priors)
        result = self.MIMT_decoder(joint, ctx, mask)

        return result

    def forward(self, input, condition, visual=False, visual_prefix='./tmp', return_mask=False):
        if self.training:
            b, c, h, w = input.shape
            i = 1
            
            #########
            #self.training_mask_ratio = 0.
            #########
            mask = (torch.rand((b, self.num_masks, h, w)).to(input.device) <= self.training_mask_ratio).float()

            output = self.quant(input) 
            ctx = output * mask

            condition = self.PA_forward(i, condition, ctx, mask, None, None)
            
            self.entropy_model._set_condition(condition)
            likelihood = self.entropy_model._likelihood(output)
            
            if return_mask:
                return output, likelihood, 1. - mask
            else:
                return output, likelihood
        else:
            # input shape 1, 128, 68, 120
            # conditional 1, 512, 68, 120 means and scales(hyperprior and temporalprior) # 4 times of input channels
            # output of PA_forward(condition): 1, 256, 68, 120

            B, C, H, W = input.shape
            mask = torch.zeros((B, 1, H, W)).to(input.device)
            final_likelihood = torch.zeros_like(input)
            final_y_hat = torch.zeros_like(input)

            for i in range(1, self.n+1):
                # print(f"iter {i}")

                ctx = final_y_hat
                mu_sigma = self.PA_forward(i, condition, ctx, mask, None, None)
                self.entropy_model._set_condition(mu_sigma)
                score = -(self.entropy_model._likelihood(mu_sigma.chunk(2, dim=1)[0]).log2()).sum(dim=1) 

                # Sample top K scored elements
                num_samples = self.get_num_samples(i, score) # i=1~8
                new_additional_mask = self.sample_top_k(score, num_samples, mask[:,0,:,:]).to(input.device).unsqueeze(1) 
                mask += new_additional_mask
                y_hat = self.quant(input)
                final_likelihood += self.entropy_model._likelihood(y_hat) * new_additional_mask ###
                final_y_hat += y_hat * new_additional_mask
                # print(f"iteration{i}")

            return final_y_hat, final_likelihood



class SwinTransformerPAModel(nn.Module):
    def __init__(self, num_features, num_priors, entropy_model):
        super(SwinTransformerPAModel, self).__init__()
        
        self.num_priors = num_priors
        self.num_features = num_features

        self.MIMT_encoder = MIMTEncoder(input_dim=self.num_features*2, l_dim=768,
                                        num_heads=8, window_size=4,
                                        mlp_ratio=1., 
                                        qkv_bias=True, qk_scale=None, 
                                        drop=0., attn_drop=0.,drop_path=0., norm_layer=nn.LayerNorm, 
                                        use_checkpoint=False, 
                                        shift_first=False, num_sep=2, num_joint=4)

        self.patch_unembed = PatchUnEmbed()
        self.entropy_model = entropy_model

        self.conv = nn.Conv2d(in_channels=768*self.num_priors, out_channels=self.num_features*2, kernel_size=1, stride=1, padding=0)


    def quant(self, input):
        if self.entropy_model.quant_mode == 'estUN_outR' and self.training:
            def _training_quant(x):
                n = torch.round(x) - x
                n = n.clone().detach()
                return x + n

            if self.entropy_model.mean is not None:
                input_res = input - self.entropy_model.mean
                output = _training_quant(input_res) + self.entropy_model.mean 
            else:
                output = _training_quant(input)
        else:
            output = self.entropy_model.quantize(
                                input, 
                                self.entropy_model.quant_mode if self.training else "round",
                                None if self.training else self.entropy_model.mean
                     )
        return output


    def forward(self, input, condition, visual=False, visual_prefix='./tmp'):
        priors = condition.chunk(self.num_priors, dim=1)
        
        encoded = self.MIMT_encoder(priors)
        encoded = [self.patch_unembed(_encoded, input.shape[2:]) for _encoded in encoded]
        encoded = torch.cat(encoded, dim=1)
        condition = self.conv(encoded)
        
        self.entropy_model._set_condition(condition)
        output = self.quant(input)
        likelihood = self.entropy_model._likelihood(output)
        
        return output, likelihood
