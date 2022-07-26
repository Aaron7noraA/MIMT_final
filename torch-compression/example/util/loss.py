import torch
import torch.nn as nn
from .ssim import MS_SSIM, SSIM


class PSNR(nn.Module):
    """PSNR"""

    def __init__(self, reduction='none', data_range=1.):
        super(PSNR, self).__init__()
        self.reduction = reduction
        self.data_range = data_range

    def forward(self, input, target):
        mse = (input-target).pow(2).flatten(1).mean(-1)
        ret = 10 * (self.data_range ** 2 / (mse+1e-12)).log10()
        if self.reduction != 'none':
            ret = ret.mean() if self.reduction == 'mean' else ret.sum()
        return ret


class LossNorm(nn.modules.batchnorm._BatchNorm):
    """LossNorm"""

    def __init__(self, track_running_stats=True):
        super().__init__(1, affine=False, track_running_stats=track_running_stats)

    def forward(self, input):
        assert input.dim() > 0
        loss = input.unsqueeze(-1) if input.dim() == 1 else input

        mean, var = loss.mean(), loss.var()
        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / \
                        float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

            mean = self.running_mean * exponential_average_factor + \
                mean * (1-exponential_average_factor)
            var = self.running_var * exponential_average_factor + \
                var * (1-exponential_average_factor)

            with torch.no_grad():
                self.running_mean.data.copy_(mean)
                self.running_var.data.copy_(var)

        return loss.sub(mean).div(var)


def huber_loss(input, target, delta: float, reduction='batchmean'):
    abs_error = input.sub(target).abs()
    inlier_mask = abs_error.lt(delta)
    l2_loss = 0.5 * abs_error.masked_select(inlier_mask).pow(2)

    ret = torch.empty_like(input)
    ret.masked_scatter_(inlier_mask, l2_loss)

    outlier_mask = torch.logical_not(inlier_mask)
    l1_loss = delta*abs_error.masked_select(outlier_mask) - 0.5 * delta ** 2
    ret.masked_scatter_(outlier_mask, l1_loss)
    if reduction == 'mean':
        return ret.mean()
    elif reduction == "sum":
        return ret.sum()
    elif reduction == "batchmean":
        return ret.flatten(1).mean(1)

    return ret
