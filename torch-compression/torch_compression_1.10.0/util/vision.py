import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from numpy.fft import fft2, fftshift
from torch_compression.modules.functional import depth_to_space, space_to_depth

_rgb2yuv_matrices = {}


def RGB2YUV(RGB):
    device = RGB.device
    if device not in _rgb2yuv_matrices:
        _rgb2yuv_matrices[device] = torch.FloatTensor([[0.299,   0.587,   0.114],
                                                       [-0.196, -0.332, 0.5],
                                                       [0.5,    -0.419, -0.0813]]).to(device)
    T = _rgb2yuv_matrices[device]
    YUV = T.expand(RGB.size(0), -1, -1).bmm(RGB.flatten(2)).view_as(RGB)
    YUV = torch.cat([YUV[:, 0:1], YUV[:, 1:]+0.5], dim=1)
    return YUV.clamp(min=0, max=1)


_yuv2rgb_matrices = {}


def YUV2RGB(YUV):
    YUV = torch.cat([YUV[:, 0:1], YUV[:, 1:]-0.5], dim=1)
    device = YUV.device
    if device not in _yuv2rgb_matrices:
        _yuv2rgb_matrices[device] = torch.FloatTensor([[1,        0,    1.402],
                                                       [1, -0.34414, -0.71414],
                                                       [1,   1.1772,        0]]).to(device)
    T = _yuv2rgb_matrices[device]
    RGB = T.expand(YUV.size(0), -1, -1).bmm(YUV.flatten(2)).view_as(YUV)
    return RGB.clamp(min=0, max=1)


def RGB2YUV420(RGB):
    YUV = RGB2YUV(RGB)
    return torch.cat([space_to_depth(YUV[:, 0:1], 2), YUV[:, 1:2, ::2, ::2], YUV[:, 2:3, 1::2, ::2]], dim=1)


def YUV4202RGB(YUV420):
    YUV = torch.cat([depth_to_space(YUV420[:, :4], 2), F.interpolate(
        YUV420[:, 4:], scale_factor=2, mode='nearest')], dim=1)
    return YUV2RGB(YUV)


def RGB2Gray(RGB):
    T = torch.FloatTensor([[0.299,   0.587,   0.114]]).to(RGB.device)
    Gray = RGB.mul(T.view((3,)+(1,)*(RGB.dim()-2))).sum(1, keepdim=True)
    return Gray.clamp(min=0, max=1)


def image_diff(img1, img2, threshold=10):
    d = RGB2Gray(img1.sub(img2).abs())
    # d = img1.sub(img2).abs().mean(1, keepdim=True)
    mark = torch.Tensor([249, 0, 255]).div_(255.).view(1, 3, 1, 1)
    return img1.where(d.lt(threshold/255.), mark.to(img1.device))


def gen_color(colors=None, n=10):
    def crange(c1, c2, insert_n=10):
        clist = [np.linspace(c1[i], c2[i], insert_n) for i in range(3)]
        return np.vstack(clist).transpose()

    # print(type(colors))
    if isinstance(colors, np.ndarray):
        pass
    elif colors == None or (isinstance(colors, str) and colors == "RAINBOW"):
        colors = np.array([[255, 0, 0],
                           [255, 127, 0],
                           [240, 255, 0],
                           [0, 255, 0],
                           [0, 30, 255],
                           [75, 0, 130],
                           [148, 0, 211]]) / 255.
    elif isinstance(colors, str) and colors == "RAINBOW2":
        colors = np.array([[255, 0, 0],
                           [255, 127, 0],
                           [240, 255, 0],
                           [0, 255, 0],
                           [0, 30, 255],
                           [75, 0, 130],
                           [148, 0, 211]]) / 255. * 0.5
    elif isinstance(colors, str) and colors == "K":
        colors = np.array([[0, 0, 0],
                           [0, 0, 0]]) / 255.
    elif isinstance(colors, str) and colors == "G":
        colors = np.array([[117, 249, 76],
                           [117, 249, 76]]) / 255.
    elif isinstance(colors, str) and colors == "U":
        colors = np.array([[0, 255,  0],
                           [0,   0, 255]]) / 255.
    elif isinstance(colors, str) and colors == "V":
        colors = np.array([[0, 255, 0],
                           [255, 0, 0]]) / 255.
    elif isinstance(colors, str) and colors == "RB":
        assert n % 2 == 0
        r = np.array([[255, 0, 0],
                      [255, 200, 200]]) / 255.
        b = np.array([[0, 0, 255],
                      [200, 200, 255]]) / 255.
        r_tensor = gen_color(r, n=n//2)
        b_tensor = gen_color(b, n=n//2)
        return torch.cat([r_tensor, b_tensor])

    c = len(colors)
    ln = (n*10-1)//(c-1)+1

    linear_color = []
    for i in range(c-1):
        li = crange(colors[i], colors[i+1], ln)
        linear_color.append(li[1:] if i else li)

    linear_color = np.concatenate(linear_color, axis=0)
    index = np.linspace(0, len(linear_color)-1, n).astype(int)
    return torch.from_numpy(linear_color[index])


class PlotHeatMap(torch.nn.Module):
    """heat map color encoding scheme

    Args:
        color (str): color map `'RAINBOW'` | `'RAINBOW'` | `'K'` | `'RB'`

    Shape:
        - Input: :math:`(N, 1, H, W)`
        - Output: :math:`(N, 3, H, W)`

    Returns:
        Heatmap
    """

    def __init__(self, color='RAINBOW'):
        super().__init__()
        color_map = gen_color(color, n=10).t().unsqueeze(1).float()
        self.register_buffer('color_map', color_map.repeat(1, 2, 1))

    def forward(self, input):
        assert input.size(1) == 1
        input = input.permute(0, 2, 3, 1) * 2 - 1  # ~(-1, 1) (B, H, W, 1)
        grid = torch.cat([input.neg(), torch.zeros_like(input)], dim=-1)
        if self.color_map.device != input.device:
            self.color_map = self.color_map.to(input.device)
        heatmap = F.grid_sample(self.color_map.repeat(grid.size(0), 1, 1, 1), grid,
                                mode='bilinear', padding_mode='border', align_corners=True)
        return heatmap


class PlotYUV(torch.nn.Module):
    def __init__(self, mode="YUV") -> None:
        super().__init__()
        self.U_plot = PlotHeatMap("U")
        self.V_plot = PlotHeatMap("V")

    def forward(self, input):
        Y = input[:, :-2]
        U = self.U_plot(input[:, -2:-1])
        V = self.V_plot(input[:, -1:])
        if input.size(1) == 6:
            Y = depth_to_space(Y, 2).repeat(1, 3, 1, 1)
            merged = torch.cat([Y, torch.cat([U, V], dim=2)], dim=3)
        else:
            merged = torch.cat([Y.repeat(1, 3, 1, 1), U, V], dim=3)

        return merged


plot_yuv = PlotYUV()


def plot_grad_flow(named_parameters, figname="./gradflow.png", full_name=False):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backward() as
    "plot_grad_flow(module.named_parameters())" to visualize the gradient flow'''
    avg_grads, max_grads, names = [], [], []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n) and p.grad is not None:
            texts = n.replace('.weight', '').split(".")
            if full_name:
                names.append("_".join(texts))
            else:
                names.append("_".join(text[:3] if i != len(
                    texts)-1 else text[:6] for i, text in enumerate(texts)))
            avg_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())
    leng = len(max_grads)
    graph = plt.figure(figsize=(4.8, 6.4/20*leng))
    plt.barh(np.arange(leng), max_grads[::-1], alpha=0.3, lw=1.5, color="c")
    plt.barh(np.arange(leng), avg_grads[::-1], alpha=0.5, lw=1.5, color="b")
    plt.vlines(0, -1, leng+1, lw=2, color="k")
    plt.yticks(range(leng), names[::-1])
    plt.ylim((-1, leng))
    # zoom in on the lower gradient regions
    top = np.max(max_grads)
    plt.xlim(left=-top*1e-6, right=top*1.2)
    plt.ylabel("Layers")
    plt.xlabel("gradient value(abs)")
    plt.title("Gradient Flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)],
               ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    plt.savefig(figname)
    plt.close(graph)


@torch.no_grad()
def plot_bits(likelihoods, input, figname, vmin=None, vmax=16, cmap='Blues'):
    H, W = input.size()[-2:]
    rates = 0
    for ll in likelihoods:
        rate = ll.clamp_min(1e-9).log2().sum(1, keepdim=True).neg()
        Hs, Ws = np.ceil(H/rate.size(-2)), np.ceil(W/rate.size(-1))
        aligned = torch.nn.functional.interpolate(
            rate, size=(H, W), mode='nearest')
        rates += aligned/(Hs*Ws)

    if rates.size(0) > 1:
        raise NotImplementedError()
    else:
        fig, ax = plt.subplots(figsize=(6.4*(W/H*.75), 4.8))
        sns.heatmap(rates[0, 0].cpu(), ax=ax, vmin=vmin, vmax=vmax, cmap=cmap,
                    linewidths=0, xticklabels=False, yticklabels=False)
        ax.set_title('Bits map (bpp: %.4f)' % rates[0, 0].mean())
        fig.savefig(figname)
        plt.close(fig)


@torch.no_grad()
def fft_visual(input, figname, zoom_in=True):
    A3 = input[0].mean(0).mul(255.).clamp(0, 255).round().cpu().numpy()
    H, W = np.shape(A3)
    F3 = fft2(A3)/(W*H)
    F3 = fftshift(F3)
    P3 = np.abs(F3)
    # print(P3.min(), P3.max())

    if zoom_in:
        hH, hW = H//2, W//2
        qH, qW = H//32, W//32
        P3 = P3[hH-qH:hH+qH, hW-qW:hW+qW]

    fig, ax = plt.subplots(figsize=(6.4*(W/H*.75), 4.8))
    sns.heatmap(np.log(1+P3), ax=ax, vmin=0, vmax=4, cmap="viridis",
                linewidths=0, xticklabels=False, yticklabels=False)
    fig.savefig(figname)
    plt.close(fig)


@torch.no_grad()
def channel_analysis(feature, figname):
    data = feature[0].flatten(1)
    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(data.mean(1).cpu().numpy())
    ax[0].set_title("mean")
    ax[1].plot(data.var(1).cpu().numpy())
    ax[1].set_title("var")
    ax[2].plot(data.abs().max(1)[0].cpu().numpy())
    ax[2].set_title("abs_max")

    plt.savefig(figname)
    plt.close(fig)


@torch.no_grad()
def channel_analysis_queue(features, figname):
    data = torch.stack(features, 1)[0].flatten(2)
    mean = data.mean(2).cpu()
    maxmean, minmean = mean.max(), mean.min()
    var = data.var(2).cpu()
    maxvar, minvar = var.max(), var.min()
    abs = data.max(2)[0].cpu()
    maxabs, minabs = abs.max(), abs.min()

    for l in range(data.size(0)):
        fig, ax = plt.subplots(3, sharex=True)
        ax[0].plot(mean[l].numpy())
        ax[0].set_title("mean")
        ax[0].set_ylim(minmean, maxmean)
        ax[1].plot(var[l].numpy())
        ax[1].set_title("var")
        ax[1].set_ylim(minvar, maxvar)
        ax[2].plot(abs[l].numpy())
        ax[2].set_title("abs_max")
        ax[2].set_ylim(minabs, maxabs)

        plt.savefig(figname+f"_FCA_{l}.png")
        plt.close(fig)


if __name__ == "__main__":
    t = torch.empty(1, 3, 256, 256)
    lls = (torch.randn(1, 128, 16, 16).add(10).mul(10000000).add(10).sigmoid(),
           torch.randn(1, 128, 4, 4).mul(100).add(10).sigmoid(),)
    plot_bits(lls, t, './bits2d.png')
