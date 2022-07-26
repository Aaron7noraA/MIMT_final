import torch
import torch.nn.functional as F

from util.sampler import warp3d


def gen_gaussian_kernel(kernel_size: int, std, device='cpu'):
    """Makes 2D gaussian Kernel for convolution."""
    d = torch.distributions.Normal(float(kernel_size//2), float(abs(std)))

    sample = torch.arange(kernel_size, dtype=torch.float32, device=device)
    vals = d.log_prob(sample).exp()

    gauss_kernel = torch.einsum('i,j->ij', vals, vals)

    return gauss_kernel / (gauss_kernel.sum() + 1e-12)


_gaussian_kernels = {}


def get_gaussian_kernel(kernel_size: int, std, device='cpu', channel=3):
    dis_key = (kernel_size, std, device)
    if dis_key not in _gaussian_kernels:
        _gaussian_kernels[dis_key] = gen_gaussian_kernel(*dis_key)

    kernel_key = dis_key+(channel,)
    if kernel_key not in _gaussian_kernels:
        kernel = _gaussian_kernels[dis_key]
        _gaussian_kernels[kernel_key] = kernel.repeat(channel, 1, 1, 1)
    return _gaussian_kernels[kernel_key]


def make_scale_space(input, vstack=True, sigmas=[0, 4, 8], kernel_sizes=[1, 3, 5]):
    """make scale_space volume"""
    volume = []
    C = input.size(1)
    for sigma, kernel_size in zip(sigmas, kernel_sizes):
        if sigma == 0:
            volume.append(input)
            continue
        kernel = get_gaussian_kernel(
            kernel_size, sigma, device=input.device, channel=C)
        padded = F.pad(input, pad=((kernel_size-1)//2, ) * 4,
                       mode='replicate')
        blured = F.conv2d(padded, kernel, groups=C)
        volume.append(blured)
        if vstack:
            volume.insert(0, blured)
    return torch.stack(volume, dim=2)


def make_inplace_pool(input, vstack=True, lv=3):
    """make inplace_pool volume"""
    volume = [input]
    shape = input.size()[-2:]
    for l in range(1, lv):
        pooled = F.avg_pool2d(volume[-1], kernel_size=2)
        volume.append(pooled)

    for l in range(-lv+1, 0):
        blured = F.interpolate(volume[l], shape, mode='nearest')
        volume[l] = blured
        if vstack:
            volume.insert(0, blured)

    return torch.stack(volume, dim=2)


def scale_space(input, scale, mode='blur', vstack=True, **kwargs):
    """scale_space"""
    if input.dim() == 4:
        if mode == 'blur':
            volume = make_scale_space(input, vstack, **kwargs)
        elif mode == 'pool':
            volume = make_inplace_pool(input, vstack, **kwargs)
    else:
        volume = input

    B, C, D, H, W = volume.size()
    if vstack:
        idx = (scale + 1) * ((D-1) / 2)
    else:
        idx = scale * (D-1)
    idx = idx.unsqueeze(2)
    lb = idx.detach().floor().clamp(0, D-1)
    ub = (lb + 1).clamp(0, D-1)
    alpha = idx - idx.floor()

    lv = volume.gather(2, lb.long().expand(B, C, -1, H, W))
    uv = volume.gather(2, ub.long().expand(B, C, -1, H, W))

    val = (1-alpha) * lv + alpha * uv
    return val.squeeze(2), (1 - scale.abs()) * 2


def scale_space_3d(input, flow, mode='blur', vstack=True, sample_mode='bilinear', padding_mode='border', align_corners=True, **kwargs):
    """scale_space"""
    if input.dim() == 4:
        if mode == 'blur':
            volume = make_scale_space(input, vstack, **kwargs)
        elif mode == 'pool':
            volume = make_inplace_pool(input, vstack, **kwargs)
    else:
        volume = input

    D = volume.size(2)
    if vstack:
        scale = flow[:, -1:] * (D-1)
    else:
        scale = (flow[:, -1:] * 2 - 1) * (D-1)

    flow = torch.cat([flow[:, :2], scale], dim=1)
    # print(volume.shape, flow.shape)

    if flow.dim() == 4:
        return warp3d(volume, flow.unsqueeze(2), sample_mode, padding_mode, align_corners).squeeze(2)
    return warp3d(volume, flow, sample_mode, padding_mode, align_corners)


class ScaleSpace3d(torch.nn.Module):
    """ScaleSpace3d

        mentioned in https://openaccess.thecvf.com/content_CVPR_2020/papers/Agustsson_Scale-Space_Flow_for_End-to-End_Optimized_Video_Compression_CVPR_2020_paper.pdf
    """

    def __init__(self, mode='blur', vstack=False, activation=torch.sigmoid, sample_mode='bilinear', padding_mode='border', align_corners=True, **kwargs):
        super(ScaleSpace3d, self).__init__()
        self.mode = mode
        self.vstack = vstack
        self.activation = activation
        self.sample_mode = sample_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.scale_space_kwargs = kwargs

    def forward(self, input, flow):
        if self.activation is not None:
            scale = self.activation(flow[:, -1:])
            flow = torch.cat([flow[:, :2], scale], dim=1)
        return scale_space_3d(input, flow, self.mode, self.vstack, self.sample_mode, self.padding_mode, self.align_corners, **self.scale_space_kwargs)
