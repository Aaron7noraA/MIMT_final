import argparse
import math
import os
import random
import shutil
import warnings
from functools import partial
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from example.util.metric import PSNR_np
from torch import cat, linspace, nn, rand, rand_like, randn, randn_like, stack
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.utils import _pair
from torchvision.datasets.folder import pil_loader as imgloader
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from tqdm import tqdm, trange

import torch_compression as trc
import torch_compression.util.vision as V
from torch_compression.modules.conditional_module import ConditionalLayer
from torch_compression.util import *

__DEVICE__ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_image(path, device='cpu'):
    return to_tensor(imgloader(path)).unsqueeze(0).to(device)


def test_signalconv():
    t = rand(1, 3, 4, 4)
    conv = trc.SignalConv2d(3, 3, 3, stride=2, padding=1)
    deconv = trc.SignalConvTranspose2d(
        3, 3, 3, stride=2, padding=1, output_padding=1)

    t2 = conv(t)
    print(t2.shape)
    print(t2)
    t3 = deconv(t)
    print(t3.shape)
    print(t3)

    conv = trc.SignalConv2d(3, 3, 3, stride=1, padding=1)
    deconv = trc.SignalConvTranspose2d(3, 3, 3, stride=1, padding=1)

    t2 = conv(t)
    print(t2.shape)
    print(t2)
    t3 = deconv(t)
    print(t3.shape)
    print(t3)

    conv = trc.SignalConv2d(3, 3, 3, stride=2, padding=1, parameterizer=None)
    deconv = trc.SignalConvTranspose2d(
        3, 3, 3, stride=2, padding=1, parameterizer=None, output_padding=1)

    t2 = conv(t)
    print(t2.shape)
    print(t2)
    t3 = deconv(t)
    print(t3.shape)
    print(t3)

    conv = trc.SignalConv2d(3, 3, 3, stride=1, padding=1, parameterizer=None)
    deconv = trc.SignalConvTranspose2d(
        3, 3, 3, stride=1, padding=1, parameterizer=None)

    t2 = conv(t)
    print(t2.shape)
    print(t2)
    t3 = deconv(t)
    print(t3.shape)
    print(t3)


def test_gdn():
    t = rand(1, 3, 4, 4)
    gdn = trc.GeneralizedDivisiveNorm(3, simplify=False)
    print(gdn)

    t2 = gdn(t)
    print(t2.shape)
    print(t2)


def test_entropy_model():
    C = 4
    EB = trc.EntropyBottleneck(C)
    params, quants = trc.get_coder_params(EB)
    optim = torch.optim.Adam([dict(params=params, lr=1e-4),
                              dict(params=quants, lr=1e-3)])

    def print_param():
        print(EB.factorizer[0].weight.flatten())
        print(EB.factorizer[0].weight.grad.flatten())
        print(EB.quantiles.flatten())
        if EB.quantiles.grad is not None:
            print(EB.quantiles.grad.flatten())

    def train(epoch):
        optim.zero_grad()
        t = rand(4, C, 4, 4)
        t2, ll = EB(t)

        loss = F.mse_loss(t, t2) * 2048 + trc.estimate_bpp(ll, 16).mean()
        print(f'M{epoch}')
        print(loss.item())

        loss += EB.aux_loss()
        loss.backward()

        print_param()

        optim.step()

        print(f'R{epoch}E')
        print_param()

    for i in range(2):
        train(i+1)

    C = 64

    mse = nn.MSELoss()
    logits = randn(1, C, 4, 4).to(__DEVICE__).requires_grad_()

    m = trc.EntropyBottleneck(C)
    # m = trc.GaussianConditional(1, use_mean=True)
    # m = trc.BernoulliConditional()
    print(m)
    m = m.to(__DEVICE__)

    t2 = None
    if isinstance(m, trc.EntropyBottleneck):
        C2 = C
    else:
        C2 = C * m.condition_size

    m2 = nn.Sequential(
        nn.Conv2d(C, C*2, 1),
        nn.Conv2d(C*2, C*4, 1),
        nn.Conv2d(C*4, C*8, 1),
        nn.Conv2d(C*8, C*4, 1),
        nn.Conv2d(C*4, C2, 1),
    )
    m2 = m2.to(__DEVICE__)

    optim = torch.optim.Adam([{'params': m.parameters(), 'lr': 1e-3},
                              {'params': m2.parameters(), 'lr': 1e-3},
                              {'params': [logits], 'lr': 1e-3}])
    lmda = 0.1

    for i in range(10000):
        optim.zero_grad()
        t = logits
        # if isinstance(m, trc.BernoulliConditional):
        #     t = AF.hard_sigmoid(logits)
        #     # t = torch.sigmoid(logits)
        if not isinstance(m, trc.EntropyBottleneck):
            t2 = m2(t)
        t_tilde, t_likelihood = m.forward(t, t2)
        r = m.rate_estimate(t_likelihood).mean()
        d = mse(t, t_tilde)
        loss = d + lmda * r
        loss.backward()
        optim.step()

        if i % 500 == 0:
            print(loss.item(), d.item(), r.item(), d.item()/r.item())

    print('train RD')
    print(mse(t, t_tilde).item(), m.rate_estimate(t_likelihood).sum().item())

    m.eval()
    t_tilde, t_likelihood = m.forward(t, t2)
    print('test RD')
    print(mse(t, t_tilde).item(), m.rate_estimate(t_likelihood).sum().item())

    from torch_compression.util.bitstream import BitStreamIO
    file_name = _TMP+"compress_test.ifc"
    stream, t_hat = m.compress(t, t2, return_sym=True)
    stream, outbound_stream = stream
    stream_list, shape_list = [stream, outbound_stream], [t.size()]

    print('compress D')
    print(mse(t, t_hat).item())

    # stream_io = BitStreamIO(file_name, 'w')
    # stream_io.write(stream_list, shape_list)
    # stream_io.write_file()

    with BitStreamIO(file_name, 'w') as fp:
        fp.write(stream_list, shape_list)
        fp.write(stream_list, shape_list)

    with BitStreamIO(file_name, 'r') as fp:
        for streams, shapes in fp.split(1):
            # streams, shapes = fp.read()
            print(len(streams), len(shapes))
            shape = tuple(shapes[0])

            t_hat = m.decompress(*streams, shape, t2, device=__DEVICE__)
            print(t_hat.shape, shape)

            print('decode RD')
            print(mse(t, t_hat).item(), os.path.getsize(file_name)*8)

        # streams, shapes = fp.read()
        # print(len(streams), len(shapes))
        # shape = tuple(shapes[0])

        # t_hat = m.decompress(*streams, shape, t2, device=__DEVICE__)
        # print(t_hat.shape, shape)

        # print('decode RD')
        # print(mse(t, t_hat).item(), os.path.getsize(file_name)*8)

    # streams, shapes = BitStreamIO(file_name, 'r').read_file()
    # print(len(streams), len(shapes))
    # shape = tuple(shapes[0])

    # t_hat = m.decompress(*streams, shape, t2, device=__DEVICE__)
    # print(t_hat.shape, shape)

    # print('decode RD')
    # print(mse(t, t_hat).item(), os.path.getsize(file_name)*8)


def transform_weight(weight):
    weight_new = {}
    for k, w in weight.items():
        key = k.replace('module.', '')
        if key.endswith('quantiles') and w.dim() == 3:
            w = w.squeeze(1).t()

        weight_new[key] = w

    return weight_new


def test_hpcoder():
    try:
        log_dir = os.getenv('LOG')+'torch_compression/'
    except:
        log_dir = os.getenv('HOME')+'/Downloads/temp/'
    import sys

    load_model = '1017_0216' if len(sys.argv) == 1 else sys.argv[1]
    weight_dir = log_dir+load_model+'/'

    ckpt = torch.load(weight_dir+'model.ckpt', map_location='cpu')
    weight = ckpt['coder']

    from math import log10

    m = trc.GoogleHyperPriorCoder(
        192, 384, 256, use_mean=True, use_abs=True).to(__DEVICE__)
    try:
        m.load_state_dict(weight)
    except:
        m.load_state_dict(transform_weight(weight))
    a = trc.util.Alignment()
    mse = nn.MSELoss()
    print(m)
    print('load from', load_model)
    # print('device:', __DEVICE__)

    result_dir = './results/'
    transform = transforms.Compose([
        # transforms.CenterCrop((1024, 1920)),
        # transforms.CenterCrop((64, 64)),
        transforms.ToTensor()
    ])
    # img = transform(
    #     imgloader('../c387dbc4d3c23f45627fc6212666fa459f111938.png')).unsqueeze(0)
    # img = transform(imgloader(result_dir+'cute.png')).unsqueeze(0)
    Iframe_dir = os.getenv("LOG")+"OpenDVC/raw_video_1080/BasketballDrive/"
    mses, rates = [], []
    for f in os.listdir(Iframe_dir):
        if f[0] == '.':
            continue
        fid = int(f.split(".")[0].split("_")[-1])
        if fid % 10 != 1:
            continue

        img = transform(imgloader(Iframe_dir+f)).unsqueeze(0)
        print(img.shape)

        t = img.to(__DEVICE__)
        ta = a.align(t)
        t_tilde, rate = m(ta)
        print(t.shape, t_tilde.shape, rate.shape)
        t_tilde = a.resume(t_tilde, img.size())

        print('train RD')
        print(mse(t, t_tilde).item(), rate.item())

        m.eval()
        t_tilde, rate = m(ta)
        t_tilde = a.resume(t_tilde, img.size())
        print('test RD')
        print(mse(t, t_tilde).item(), rate.item())
        mses.append(mse(t.mul(255.).round(), t_tilde.mul(255.).round()).item())
        rates.append(rate.item())

    print(np.mean(mses), np.mean(rates))

    # file_name = result_dir+"compress_test.ifc"

    # with trc.util.BitStreamIO(file_name, 'w') as fp:
    #     t0 = perf_counter()
    #     stream_list, shape_list = m.compress(ta)
    #     print(len(stream_list[0]), len(stream_list[1]))
    #     print('encode time:{:.3f}(s)'.format(perf_counter() - t0))

    #     t0 = perf_counter()
    #     fp.write(stream_list, shape_list)

    # print('IO time:{:.3f}(s)'.format(perf_counter() - t0))

    # real_rate = os.path.getsize(file_name)*8/np.prod(t.size()[-2:])
    # print(real_rate)

    # with trc.util.BitStreamIO(file_name, 'r') as fp:
    #     # streams, shapes = fp.chunk(2)
    #     # print(len(streams), len(shapes))
    #     for streams, shapes in fp.split(2):
    #         # print(len(streams), len(shapes))
    #         print(len(streams[0]), len(streams[1]))

    #         t0 = perf_counter()
    #         t_hat = m.decompress(streams, shapes)
    #         print('decode time:{:.3f}(s)'.format(perf_counter() - t0))
    #         t_hat = a.resume(t_hat, img.size())

    #         print('decode RD')
    #         _mse = mse(t, t_hat).item()
    #         psnr = 20 * log10(1.) - 10 * log10(_mse)
    #         print(_mse, psnr, real_rate)

    #     save_image(t_hat, result_dir+'reconstruct.png')


def test_context():
    C, H, W = 2, 16, 16

    m = trc.ContextModel(C, C*4, trc.GaussianMixtureModelConditional(K=3))
    # m = ContextModel(C, C*4, trc.GaussianConditional(use_mean=True))
    m = m.to(__DEVICE__)

    f = randn(1, C, H, W).to(__DEVICE__)*50
    con = rand(1, C*4, H, W).to(__DEVICE__)

    # f2, ll = m(f, con)
    # print(f2.shape, ll.shape)

    t0 = perf_counter()
    s, sl = m.compress2(f, con)
    print('encode time:{:.3f}(s)'.format(perf_counter() - t0))
    print('package len', len(s))
    cc = 0
    for ss in sl:
        # print(len(ss[0]), len(ss[1]))
        cc += len(ss[0]) + len(ss[1])
    print('splited package len', cc)
    sl2 = [b'\x46\xE2\x84\x91'.join(ss) for ss in sl]
    print('elem conut', len(sl2))
    s2 = b'\x46\xE2\x84\x90'.join(sl2)
    print('reconcat len', len(s2))

    t0 = perf_counter()
    f2 = m.decompress(s, sl, f.size(), con)
    print('decode time:{:.3f}(s)'.format(perf_counter() - t0))
    print(f2.shape)

    # print(f.round().permute(0, 2, 3, 1), '\n', f2.permute(0, 2, 3, 1))
    print(F.mse_loss(f.round(), f2))


def test_range_coding():
    lim = 6

    def torchac1(pt):
        s = pt['s'][:lim]
        i = pt['i'][:lim]
        c = pt['c']
        l = pt['l']
        print(s, i)
        print(c[i.long()])
        print(l[i.long()])
        import torchac_backend_cpu as ac
        strings = ac.encode_cdf_index(c, l, i, s)
        string = b'\x46\xE2\x84\x91'.join(strings)
        print(string)
        print(len(string))

        rec = ac.decode_cdf_index(c, l, i, *strings)
        print(rec)

        print((s - rec).float().mean())

    torchac1(torch.load('./test_hp.pt'))
    torchac1(torch.load('./test_cb.pt'))

    def torchac2(pt):
        s = pt['s'][:lim]
        i = pt['i'][:lim]
        c = pt['c']
        l = pt['l']
        print(s, i)
        print(c[i.long()])
        print(l[i.long()])
        import torchac_backend_cpu2 as ac2
        string2 = ac2.unbounded_index_range_encode(s, i, c, l, 16, 4)
        print(string2)
        print(len(string2))

        rec2 = ac2.unbounded_index_range_decode(string2, i, c, l, 16, 4)
        print(rec2)

        print((s - rec2).float().mean())

    torchac2(torch.load('./test2_hp.pt'))
    torchac2(torch.load('./test2_cb.pt'))


def resavenpz(file_name):
    pt = torch.load(file_name)
    new_dict = {}
    for k, t in pt.items():
        new_dict[k] = t.numpy()
    np.savez_compressed(file_name.replace('./', '../tmp/'), **new_dict)


def cal_num_params(m: nn.Module):
    count = 0
    for p in m.parameters():
        count += p.numel()

    return count


class DepthwiseSeparableConv2d(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super(DepthwiseSeparableConv2d, self).__init__(
            trc.Conv2d(in_channels, in_channels, kernel_size,
                       stride, groups=in_channels, **kwargs),
            trc.Conv2d(in_channels, out_channels, kernel_size=1)
        )


class DepthwiseSeparableConvTranspose2d(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super(DepthwiseSeparableConvTranspose2d, self).__init__(
            trc.ConvTranspose2d(in_channels, in_channels, kernel_size,
                                stride, groups=in_channels, **kwargs),
            trc.Conv2d(in_channels, out_channels, kernel_size=1)
        )


class Invertible1x1Conv(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.weight = torch.Tensor(num_channels, num_channels)

        self.reset_parameters()

    def reset_parameters(self):
        # initialization done with rotation matrix
        w_init, _ = torch.qr(torch.randn_like(self.weight))
        self.weight.data.copy_(w_init)

    def jac(self):
        return torch.det(self.weight).abs().log()

    def forward(self, x, jac, rev=False):
        if not rev:
            weight = self.weight
            _jac = self.jac() * np.prod(x.size()[-2:])
        else:
            weight = self.weight.inverse()
            _jac = self.jac() * -np.prod(x.size()[-2:])

        return F.conv2d(x, weight.unsqueeze(-1).unsqueeze(-1)), jac + _jac


def switch_mode(coder, args, iter, mode):
    if iter and iter % args.alt_step == 0:
        coder.num_layers = mode+1
        if mode > 1:
            coder['analysis'+str(mode-1)].requires_grad_(False)
            coder['synthesis'+str(mode-1)].requires_grad_(False)
        else:
            coder.requires_grad_(True)
        mode = (mode + 1) % args.num_layers

    return mode


@torch.no_grad()
def plot_func(func, shape=None):
    if shape is None:
        x = torch.linspace(-10, 10, 1000)
    else:
        x = torch.linspace(-10, 10, int(np.prod(shape))).view(shape)

    fig = plt.figure()
    y = func(x)
    plt.plot(x.flatten(), y.flatten())
    plt.show()
    plt.close(fig)


class Flatten(nn.Module):
    r"""
    Flattens a contiguous range of dims into a tensor. For use with :class:`~nn.Sequential`.

    Shape:
        - Input: :math:`(N, *dims)`
        - Output: :math:`(N, \prod *dims)` (for the default case).

    Args:
        start_dim: first dim to flatten (default = 1).
        end_dim: last dim to flatten (default = -1).

    Examples::
        >>> input = torch.randn(32, 1, 5, 5)
        >>> m = nn.Sequential(
        >>>     nn.Conv2d(1, 32, 5, 1, 1),
        >>>     nn.Flatten()
        >>> )
        >>> output = m(input)
        >>> output.size()
        torch.Size([32, 288])
    """
    __constants__ = ['start_dim', 'end_dim']
    start_dim: int
    end_dim: int

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.flatten(self.start_dim, self.end_dim)

    def extra_repr(self) -> str:
        return 'start_dim={}, end_dim={}'.format(
            self.start_dim, self.end_dim
        )


class PredictionModel(nn.Module):
    """PredictionModel of C2F HPCoder"""

    def __init__(self, num_features, kernel_size=5):
        super().__init__()
        self.num_features = num_features
        self.kernel_size = _pair(kernel_size)
        self.padding = _pair((kernel_size-1) // 2)

        self.param = nn.Sequential(
            nn.Conv2d(self.num_features, self.num_features, 3, padding=1),
            nn.Conv2d(
                self.num_features, self.num_features, 3, padding=1, stride=2),
            nn.Conv2d(self.num_features,
                      self.num_features, 3, padding=1),
            Flatten(1),
            nn.Linear(self.num_features*((self.kernel_size[0]+1)//2)*(
                (self.kernel_size[1]+1)//2), self.num_features*2)
        )

    def forward(self, input):
        B, C, H, W = input.size()
        unfolded = F.unfold(input, self.kernel_size, padding=self.padding)

        permuted = unfolded.transpose(1, 2).reshape(-1, C, *self.kernel_size)

        condition = self.param(permuted).view(B, H, W, -1).permute(0, 3, 1, 2)

        return condition


def enc_optim():
    path = "/work/nctu0756640/log/torch_compression/ANFHyperPriorCoder_0402_1458/"
    # ckpt = torch.load(path, map_location="cpu")
    coder = trc.hub.AugmentedNormalizedFlowHyperPriorCoder(
        128, 320, 192, num_layers=2, use_code=False, use_DQ=True, use_context=True, condition="GaussianMixtureModel")
    print(coder)
    coder.requires_grad_(False)
    self = coder
    print(cal_num_params(coder))
    from example.util.loss import PSNR
    psnr = PSNR()

    img = rand(1, 3, 256, 256)
    with torch.no_grad():
        input, code, _ = coder.encode(img, None, None)
        hyper = hyper_code = coder.hyper_analysis(
            code.abs() if coder.use_abs else code)
    print(code.shape, hyper.shape)

    code = code.requires_grad_(True)
    init_input = input

    optim = torch.optim.Adam([code], lr=1e-4)

    for _ in range(3):
        optim.zero_grad()

        Y_error, y_tilde, z_tilde, y_likelihood, z_likelihood = self.entropy_model(
            input, code)

        input, code, hyper_code = torch.zeros_like(input), y_tilde, z_tilde
        # input, code, hyper_code = Y_error, y_tilde, z_tilde
        # input = Y_error
        debug(Y_error.shape, code.shape, hyper_code.shape)
        input, code, _ = self.decode(input, code)

        if self.DQ is not None:
            debug("DQ")
            input = self.DQ(input)

        rec = input
        distortion = (img - rec).pow(2).mean()
        rate = trc.estimate_bpp((y_likelihood, z_likelihood), input=img).mean()

        rd_loss = distortion * (255.**2) * 5e-2 + rate

        rd_loss.backward()
        print(rd_loss.item(), psnr(rec, img).mean().item(), rate.item())
        optim.step()


class BD_RATE_Loss(nn.Linear):

    def __init__(self, deg: int, precision=1e-3) -> None:
        super().__init__(deg, 1)
        self.precision = precision

    def forward(self, distortion, rate) -> torch.Tensor:
        PSNR = 10*torch.log10(1/distortion.add(1e-12))
        log_rate = rate.log()
        return super().forward(input)


def rd_sample():
    size = (0.5, 2)
    block = (385, 143)

    bases = [
        (1.5, 40),
        (1.0, 38),
        (0.5, 36),
        (0.5, 34),
        (0.0, 32),
        (0.0, 30),
    ]
    points = [
        (69, 25),
        (208, 64),
        (381, 94),
        (146, 82),
        (333, 58),
        (228, 78),
    ]

    size = (0.25, 2.5)
    block = (271, 148)

    bases = [
        (0.75, 42.5),
        (0.25, 40),
        (0.25, 37.5),
        (0.0, 35),
        (0.0, 32.5),
        (0.0, 30),
    ]
    points = [
        (91, 98),
        (253, 91),
        (33, 83),
        (166, 76),
        (91, 63),
        (51, 57),
    ]

    # bases = [
    #     (0.75, 42.5),
    #     (0.5, 40),
    #     (0.25, 37.5),
    #     (0.0, 35),
    #     (0.0, 32.5),
    #     (0.0, 30),
    # ]
    # points = [
    #     (156, 38),
    #     (160, 82),
    #     (223, 124),
    #     (236, 66),
    #     (137, 74),
    #     (77, 71),
    # ]
    # bases = [
    #     (0.75, 42.5),
    #     (0.5, 40),
    #     (0.25, 37.5),
    #     (0.25, 35),
    #     (0.0, 32.5),
    #     (0.0, 30),
    # ]
    # points = [
    #     (27, 31),
    #     (103, 96),
    #     (223, 157),
    #     (65, 195),
    #     (207, 198),
    #     (76, 113),
    # ]

    # size = (0.25, 2.5)
    # block = (265, 166)

    # bases = [
    #     (1.25, 40),
    #     (0.75, 37.5),
    #     (0.25, 35),
    #     (0.0, 32.5),
    #     (0.0, 30),
    #     (0.0, 27.5),
    # ]
    # points = [
    #     (177, 156),
    #     (35, 122),
    #     (207, 119),
    #     (251, 113),
    #     (123, 107),
    #     (53, 113),
    # ]

    # bases = [
    #     (1.25, 40),
    #     (0.75, 37.5),
    #     (0.5, 35),
    #     (0.25, 32.5),
    #     (0.0, 30),
    #     (0.0, 27.5),
    # ]
    # points = [
    #     (142, 97),
    #     (252, 139),
    #     (210, 183),
    #     (138, 164),
    #     (203, 153),
    #     (92, 148),
    # ]
    # bases = [
    #     (1.0, 40),
    #     (0.75, 37.5),
    #     (0.5, 35),
    #     (0.25, 35),
    #     (0.25, 32.5),
    #     (0.0, 30),
    #     (0.0, 27.5),
    # ]
    # points = [
    #     (211, 38),
    #     (180, 115),
    #     (211, 198),
    #     (243, 78),
    #     (60, 128),
    #     (190, 170),
    #     (83, 162),
    # ]

    # size = (0.25, 2.5)
    # block = (265, 151)

    # bases = [
    #     (1.25, 20),
    #     (0.75, 17.5),
    #     (0.25, 15),
    #     (0.0, 12.5),
    #     (0.0, 10),
    #     (0.0, 10),
    # ]
    # points = [
    #     (177, 174),
    #     (35, 147),
    #     (207, 155),
    #     (251, 163),
    #     (123, 172),
    #     (53, 37),
    # ]

    # bases = [
    #     (1.25, 20),
    #     (0.75, 17.5),
    #     (0.5, 17.5),
    #     (0.25, 15),
    #     (0.0, 12.5),
    #     (0.0, 10),
    # ]
    # points = [
    #     (142, 119),
    #     (252, 158),
    #     (210, 59),
    #     (138, 60),
    #     (203, 63),
    #     (92, 77),
    # ]
    # bases = [
    #     (1.0, 40),
    #     (0.75, 37.5),
    #     (0.5, 35),
    #     (0.25, 35),
    #     (0.25, 32.5),
    #     (0.0, 30),
    #     (0.0, 27.5),
    # ]
    # points = [
    #     (211, 38),
    #     (180, 115),
    #     (211, 198),
    #     (243, 78),
    #     (60, 128),
    #     (190, 170),
    #     (83, 162),
    # ]

    # size = (0.1, 1)
    # block = (291, 142)

    # bases = [
    #     (0.6, 38),
    #     (0.4, 36),
    #     (0.3, 35),
    #     (0.2, 34),
    #     (0.1, 32),
    #     (0.1, 31),
    #     (0.1, 31),
    # ]
    # points = [
    #     (272, 61),
    #     (383, 179),
    #     (226, 102),
    #     (232, 92),
    #     (242, 139),
    #     (51, 77),
    #     (-77, -136),
    # ]

    # bases = [
    #     (0.7, 37),
    #     (0.5, 35),
    #     (0.3, 34),
    #     (0.2, 32),
    #     (0.1, 30),
    #     (0.1, 30),
    # ]
    # points = [
    #     (178, 80),
    #     (276, 194),
    #     (345, 110),
    #     (250, 155),
    #     (192, 186),
    #     (-66, -132),
    # ]

    # bases = [
    #     (0.7, 37),
    #     (0.5, 35),
    #     (0.3, 34),
    #     (0.2, 32),
    #     (0.1, 30),
    #     (0.1, 30),
    # ]
    # points = [
    #     (138, 154),
    #     (261, 274),
    #     (343, 179),
    #     (251, 235),
    #     (215, 302),
    #     (37, 105),
    # ]

    size = (0.1, 1)
    block = (290, 80)

    # bases = [
    #     (0.6, 19),
    #     (0.4, 17),
    #     (0.3, 16),
    #     (0.2, 15),
    #     (0.1, 13),
    #     (0.1, 13),
    #     (0.1, 13),
    # ]
    # points = [
    #     (272, 39),
    #     (383, 111),
    #     (226, 80),
    #     (232, 77),
    #     (242, 115),
    #     (51, 3),
    #     (-77, -110),
    # ]

    # bases = [
    #     (0.7, 18),
    #     (0.5, 17),
    #     (0.3, 15),
    #     (0.2, 14),
    #     (0.1, 12),
    #     (0.1, 12),
    # ]
    # points = [
    #     (178, 50),
    #     (276, 41),
    #     (345, 93),
    #     (250, 49),
    #     (192, 77),
    #     (-66, -98),
    # ]

    bases = [
        (0.7, 21),
        (0.5, 19),
        (0.3, 17),
        (0.2, 16),
        (0.1, 15),
        (0.1, 14),
    ]
    points = [
        (229, 19),
        (260, 68),
        (273, 89),
        (161, 61),
        (269, 61),
        (80, 36),
    ]

    # size = (0.25, 5)
    # block = (271, 271)

    # bases = [
    #     (0.75, 20),
    #     (0.25, 20),
    #     (0.25, 15),
    #     (0.0, 15),
    #     (0.0, 10),
    #     (0.0, 10),
    # ]
    # points = [
    #     (94, 241),
    #     (253, 90),
    #     (33, 216),
    #     (164, 65),
    #     (86, 209),
    #     (46, 97),
    # ]

    # bases = [
    #     (0.75, 20),
    #     (0.5, 20),
    #     (0.25, 15),
    #     (0.0, 15),
    #     (0.0, 10),
    #     (0.0, 10),
    # ]
    # points = [
    #     (156, 186),
    #     (160, 86),
    #     (223, 257),
    #     (236, 69),
    #     (137, 219),
    #     (77, 117),
    # ]
    # bases = [
    #     (0.75, 42.5),
    #     (0.5, 40),
    #     (0.25, 37.5),
    #     (0.25, 35),
    #     (0.0, 32.5),
    #     (0.0, 30),
    # ]
    # points = [
    #     (27, 31),
    #     (103, 96),
    #     (223, 157),
    #     (65, 195),
    #     (207, 198),
    #     (76, 113),
    # ]

    # IWP

    # size = (0.5, 1)
    # block = (290, 73)

    # bases = [
    #     (1.1, 38),
    #     (0.6, 37),
    #     (0.6, 36),
    #     (0.6, 35),
    #     (0.6, 35),
    #     (0.6, 34),
    #     (0.6, 33),
    #     (0.6, 31),
    # ]
    # points = [
    #     (146, 95),
    #     (263, 63),
    #     (151, 47),
    #     (48, 20),
    #     (-38, -73),
    #     (-92, -74),
    #     (-171, -130),
    #     (-230, -119),
    # ]

    # CConv
    # size = (0.5, 2)
    # block = (421, 149)

    # bases = [
    #     (1.5, 38),
    #     (1.0, 38),
    #     (1.0, 36),
    #     (0.5, 34),
    #     (0.5, 32),
    #     (0.5, 32),
    # ]
    # points = [
    #     (134, 190),
    #     (336, 108),
    #     (38, 122),
    #     (130, 69),
    #     (-169, -48),
    #     (-296, -234),
    # ]
    for p, b in zip(points, bases):
        print(p[0]/block[0]*size[0]+b[0],
              p[1]/block[1]*size[1]+b[1])


def eval_vtm():
    from example.util.loss import MS_SSIM, PSNR
    ssim_m = MS_SSIM()
    psnr_m = PSNR(data_range=255)
    vtm_dir = os.getenv("HOME")+"/Downloads/vtm_tecnick/"
    GT_dir = os.getenv("HOME")+"/Downloads/Tecnick/"
    fp = open(vtm_dir+"eval_VTM.txt", "w")
    for idx in range(1, 41):
        file_name = vtm_dir+"Tecnick%02d" % idx
        # print(file_name)

        GT = transforms.ToTensor()(imgloader(GT_dir+"Tecnick%02d" % idx+".png"))
        # print(img.shape)
        pixels = GT.size(-1) * GT.size(-2)

        bpp = os.path.getsize(file_name+".bin")*8/pixels
        # print(bpp)

        img = transforms.ToTensor()(imgloader(file_name+".png"))
        # print(img.shape)

        img = img.mul(255).clamp(0, 255).round().unsqueeze(0)
        GT = GT.mul(255).clamp(0, 255).round().unsqueeze(0)

        psnr = psnr_m(img, GT).item()
        ssim = ssim_m(img, GT).item()
        msg = "Tecnick {:02d}, rate: {:.3f}, psnr: {:.2f}, msssim: {:.3f}".format(
            idx, bpp, psnr, ssim)
        print(msg)
        fp.write(msg+"\n")
    fp.close()


def eval_bpg():
    from example.util.loss import MS_SSIM, PSNR
    ssim_m = MS_SSIM()
    psnr_m = PSNR(data_range=255)
    bpg_dir = os.getenv("HOME")+"/Downloads/bpg_tecnick/"
    os.makedirs(bpg_dir, exist_ok=True)
    GT_dir = os.getenv("HOME")+"/Downloads/Tecnick/"
    fp = open(bpg_dir+"eval_BPG.txt", "w")
    for idx in range(1, 41):
        file_name = bpg_dir+"Tecnick%02d" % idx
        # print(file_name)

        GT = transforms.ToTensor()(imgloader(GT_dir+"Tecnick%02d" % idx+".png"))
        # print(img.shape)
        pixels = GT.size(-1) * GT.size(-2)

        bpgenc = "bpgenc -q 33 -f 444 -o " + file_name + \
            ".bin " + GT_dir+"Tecnick%02d" % idx+".png"
        # print(bpgenc)
        os.system(bpgenc)

        bpp = os.path.getsize(file_name+".bin")*8/pixels
        while bpp < 0.0001:
            os.system(bpgenc)
            bpp = os.path.getsize(file_name+".bin")*8/pixels
        # print(bpp)

        bpgdec = "bpgdec -o " + file_name+".png "+file_name+".bin"
        # print(bpgdec)
        os.system(bpgdec)

        img = transforms.ToTensor()(imgloader(file_name+".png"))
        # print(img.shape)

        img = img.mul(255).clamp(0, 255).round().unsqueeze(0)
        GT = GT.mul(255).clamp(0, 255).round().unsqueeze(0)

        psnr = psnr_m(img, GT).item()
        ssim = ssim_m(img, GT).item()
        msg = "Tecnick {:02d}, rate: {:.3f}, psnr: {:.2f}, msssim: {:.3f}".format(
            idx, bpp, psnr, ssim)
        print(msg)
        fp.write(msg+"\n")
    fp.close()


def rename_tecnick():
    GT_dir = os.getenv("HOME")+"/Downloads/Tecnick/"

    for idx, f in enumerate(sorted(os.listdir(GT_dir))):
        print(f)
        os.rename(GT_dir+f, GT_dir+"Tecnick%02d.png" % (idx+1))

    IWP_dir = os.getenv("HOME")+"/Downloads/iWave++/"

    texts = []
    name_set = set()
    with open(IWP_dir+"tecnick_rates.txt", 'r') as fp:
        for line in fp.read().splitlines():
            # print(line.split(" ")[4][:-1])
            name_set.add(line.split(" ")[4])
            texts.append(line.split(" "))

    name_map = {}
    for idx, name in enumerate(sorted(list(name_set))):
        print(idx, name)
        name_map[name] = "Tecnick%02d.png" % (idx+1)
    print(name_map)

    for text in texts:
        text[4] = name_map[text[4]]

    print(texts)

    texts = sorted(texts, key=lambda t: (int(t[2][:-1]), t[4]))
    print(texts)

    with open(IWP_dir+"tecnick_rates2.txt", 'w') as fp:
        for text in texts:
            fp.write(" ".join(text)+"\n")

    for rate_dir in os.listdir(IWP_dir+"Tecnick/"):
        print(rate_dir)
        rate_dir = IWP_dir+"Tecnick/"+rate_dir+"/"
        for idx, f in enumerate(sorted(os.listdir(rate_dir))):
            print(f)
            os.rename(rate_dir+f, rate_dir+"Tecnick%02d.png" % (idx+1))


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


class CheckerBoardMaskedConv2d(nn.Conv2d):

    def __init__(self, in_channels: int, out_channels: int, kernel_size, **kwargs) -> None:
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.register_buffer("_mask", torch.zeros(*self.kernel_size))
        for loc, val in enumerate(self._mask.flatten()):
            if loc % 2:
                val.data.fill_(1)

    def forward(self, input):
        return self.conv2d_forward(input, self.weight*self._mask)


if __name__ == "__main__":
    print('Torch Compression version', trc.__version__)
    _TMP = './tmp/'
    os.makedirs(_TMP, exist_ok=True)
    # torchseed(666)
    # print(trc.__CODER_TYPES__, trc.__CONDITIONS__)
    # test_signalconv()
    # test_gdn()
    # test_entropy_model()
    # test_hpcoder()
    # test_context()Ｆ
    # test_range_coding()
    # resavenpz('./test_hp.pt')
    # resavenpz('./test_cb.pt')
    # test_variance()
    # eval_vtm()
    # eval_bpg()
    # rd_sample(), normalized=True

    # m = CheckerBoardMaskedConv2d(3, 3, 3)

    # o = torch.optim.Adam(nn.Linear(1,1).parameters(),lr=1)
    # s = torch.optim.lr_scheduler.StepLR(o, 10)
    # print(s.state_dict())
    # m = trc.modules.StochasticGumbelAnnealing(iter=0)
    # print(m.state_dict())

    # tau = []
    # for i in range(3000):
    #     if i % 100 == 0:
    #         t = randn(10).mul(100).requires_grad_()
    #         print(t)
    #         t2 = m(t)
    #         print(t2)
    #         print(t.round().sub(t2).abs().div(t).mean())
    #         print(m)

    #     tau.append(m.tau.item())
    #     m.step()

    # plt.figure()
    # plt.plot(tau)
    # plt.show()
    # m2 = trc.modules.StochasticGumbelAnnealing(iter=0)
    # print(m2.state_dict())
    # m2.load_state_dict(m.state_dict())
    # print(m2)

    # input = os.getenv("HOME")+"/Downloads/Kodak/kodim24.png"

    # I = Image.open(input)
    # I = I.convert('L')                     # 'L' for gray scale mode
    # A3 = np.asarray(I, dtype=np.float32)
    # print(A3.min(), A3.max())

    # V.fft_visual(load_image(input), _TMP+"freq.png")

    # crop_dir = _TMP+"crop/"
    # os.makedirs(crop_dir, exist_ok=True)
    # for f in os.listdir(crop_dir):
    #     print(crop_dir+f)
    #     img = load_image(crop_dir+f)
    #     save_image(F.upsample_nearest(img[..., 58:427, 90:536], (512, 768)), crop_dir+f)

    # GT_dir = os.getenv("HOME")+"/Downloads/Tecnick/"

    # GT_imgs = []
    # for idx, f in enumerate(sorted(os.listdir(GT_dir))):
    #     GT_imgs.append(load_image(GT_dir+f))

    # VTM_dir = os.getenv("HOME")+"/Downloads/vtm_tecnick/"

    # VTM_imgs = []
    # for idx, f in enumerate(sorted(os.listdir(VTM_dir))):
    #     if f.endswith("png"):
    #         VTM_imgs.append(load_image(VTM_dir+f))

    # for idx, (GT, vtm) in enumerate(zip(GT_imgs, VTM_imgs)):
    #     save_image(V.image_diff(vtm, GT, 8), _TMP +
    #                "Tecnick_VTM_diff_%02d.png" % (idx+1))

    # IWP_dir = os.getenv("HOME")+"/Downloads/iWave++/Tecnick/128/"

    # IWP_imgs = []
    # for idx, f in enumerate(sorted(os.listdir(IWP_dir))):
    #     IWP_imgs.append(load_image(IWP_dir+f))

    # for idx, (GT, iwp) in enumerate(zip(GT_imgs, IWP_imgs)):
    #     save_image(V.image_diff(iwp, GT, 8), _TMP +
    #                "Tecnick_IWP_diff_%02d.png" % (idx+1))

    # ANF_dir = os.getenv("HOME")+"/Downloads/torch-compression/tmp/ANF_1738/"

    # ANF_imgs = []
    # for idx, f in enumerate(sorted(os.listdir(ANF_dir))):
    #     if f.endswith("png"):
    #         ANF_imgs.append(load_image(ANF_dir+f))

    # for idx, (GT, anf) in enumerate(zip(GT_imgs, ANF_imgs)):
    #     save_image(V.image_diff(anf, GT, 8), _TMP +
    #                "Tecnick_ANF_diff_%02d.png" % (idx+1))

    # ds = os.getenv("DATASET")+"CLIC_train/images"

    # for path in os.listdir(ds):
    #     img = transforms.ToTensor()(imgloader(os.path.join(ds, path)))
    #     if img.shape[-1] < 256 or img.shape[-2] < 256:
    #         print(path, img.shape)
    # weight = (0.401960784313725, 0.598039215686275)

    # rd1 = (39.9087, 0.989236, 0.9728)
    # rd2 = (40.2082, 0.9832, 0.9079)

    # rd1 = (39.7559, 0.989236, 0.9271)
    # rd2 = (40.1416, 0.9832, 0.8727)

    # rd1 = (36.2025, 0.9832, 0.4355)
    # rd2 = (36.4730, 0.9832, 0.4062)

    # rd1 = (34.6759, 0.9742, 0.3072)
    # rd2 = (35.0122, 0.9832, 0.2908)

    # rd1 = (33.1141, 0.9603, 0.2100)
    # rd2 = (33.5250, 0.9832, 0.2031)

    # rd1 = (30.9617, 0.9603, 0.1195)
    # rd2 = (31.4233, 0.9832, 0.1193)

    # rd1 = (29.5231, 0.9002, 0.0779)
    # rd2 = (29.9320, 0.9832, 0.0789)

    # rd1 = (39.9087, 0.995365, 1.0528)
    # rd2 = (38.2781, 0.996720, 1.0433)

    # rd1 = (38.2322, 0.993103, 0.7499)
    # rd2 = (40.1416, 0.994980, 0.7575)

    # rd1 = (36.2025, 0.988613, 0.4684)
    # rd2 = (36.4730, 0.991343, 0.4835)

    # rd1 = (34.6759, 0.983875, 0.3296)
    # rd2 = (35.0122, 0.987224, 0.3392)

    # rd1 = (34.6759, 0.977991, 0.2306)
    # rd2 = (33.5250, 0.981284, 0.2331)

    # # rd1 = (30.9617, 0.9603, 0.1195)
    # # rd2 = (31.4233, 0.9832, 0.1193)

    # # rd1 = (29.5231, 0.9002, 0.0779)
    # # rd2 = (29.9320, 0.9832, 0.0789)

    # for a, b in zip(rd1, rd2):
    #     print(a*weight[0]+b*weight[1])

    # # import example.util.msssim as SSIM
    # from scipy.signal import fftconvolve
    # k1 = SSIM._FSpecialGauss(3, 1.5)
    # print(k1)
    # import example.util.ssim as SSIM2
    # k2 = SSIM2.create_window(3, 1.5, 1)
    # print(k2.view(3))

    # t = rand(1, 1, 6, 6)
    # print(fftconvolve(t.permute(0, 2, 3, 1).numpy(),
    #                   np.reshape(k1, (1, 3, 3, 1)), mode="valid"))
    # print(SSIM2.gaussian_blur(t, k2, use_padding=False))

    # t = rand(2, 3, 768, 512)
    # t2 = rand_like(t)*0.12+t
    # t = t.mul(255).round()
    # t2 = t2.mul(255).round()

    # def ssim2db(ssim):
    #     return -10*np.log10(1-ssim)

    # from example.util.loss import MS_SSIM
    # # from pytorch_msssim import MS_SSIM as MS_SSIM0
    # from example.util.metric import MultiScaleSSIM as MS_SSIM0

    # print(ssim2db(MS_SSIM(data_range=255.)(t, t2).mean().item()))
    # # print(MS_SSIM0(data_range=255.)(t, t2))
    # print(ssim2db(MS_SSIM0(t.permute(0, 2, 3, 1).numpy(),
    #                        t2.permute(0, 2, 3, 1).numpy(), max_val=255.)))

    # for i in range(24):
    #     t = rand(1, 3, 512, 768)
    #     image = _TMP+"noise%d" % i
    #     save_image(t, image+".png")
    #     os.system("bpgenc -f 444 -q 37 -o "+image+".bpg "+image+".png")
    #     bytes = os.path.getsize(image+".bpg")
    #     print(bytes*8/512/768)

    # os.system("bpgdec -f 420 -o "+image+".bpg "+image+".png")

    # m = trc.hub.RGBNormailize()
    # t = rand(4, 3, 8, 8)
    # m.set_mean_var(t)
    # t2 = m(t)
    # print(t2.mean(dim=(2, 3)))

    # t3 = m(t2, rev=True)
    # print(F.mse_loss(t3, t))

    # m = trc.hub.AugmentedNormalizedFlowHyperPriorCoder(
    #     128, 320, 192, num_layers=2, use_DQ=True, use_code=False,
    #     use_context=True, condition='GaussianMixtureModel', quant_mode="RUN")
    # print(m)
    # from torch_compression.modules.conditional_module import (
    #     conditional_warping, gen_condition, gen_discrete_condition,
    #     gen_random_condition, set_condition)

    # d = dict(zip([1e-1, 5e-2, 2e-2, 1e-2, 5e-3], [1, 2, 3, 4, 5]))
    # lmdas = gen_random_condition([1e-1, 5e-2, 2e-2, 1e-2, 5e-3], 40)
    # print(lmdas)
    # lmdas = [1e-1, 5e-2, 2e-2, 1e-2, 5e-3]
    # lmdas = lmdas + np.exp(np.random.uniform(np.log(np.min(lmdas)), np.log(np.max(
    #     lmdas)), len(lmdas)*2)).tolist()
    # lmdas = sorted(lmdas)
    # for l in lmdas:
    #     if l in d:
    #         print(l, d[l])
    #     else:
    #         print(l)
    # conds, lmdas = gen_discrete_condition([0.05, 0.1, 0.5, 1], 8)
    # print(conds)
    # print(torch.nn.functional.normalize(
    #             lmdas.reciprocal().pow(0), 1., dim=0))
    # conditional_warping(m, conditions=1, ver=2)
    # set_condition(m, conds)
    # print(m)
    # for n, mm in m.named_modules():
    #     if isinstance(mm, ConditionalLayer):
    #         mm.requires_grad_(True)
    #         print(n, "C")
    #     else:
    #         mm.requires_grad_(False)
    #         print(n)
    # # # print(gen_condition([5e-3, 1e-2, 2e-2, 5e-2], 8, shuffle=True))
    # # lmdas = gen_condition([2e-3, 5e-3, 1e-2, 2e-2, 5e-2], 5)
    # # print(lmdas, lmdas.shape)
    # # print(F.normalize(lmdas.reciprocal().sqrt(), p=1., dim=0).flatten())
    # # print(F.normalize(lmdas.reciprocal(), p=1., dim=0).flatten())
    # # print(F.normalize(lmdas.reciprocal().pow(2), p=1., dim=0).flatten())
    # # set_condition(m, lmdas)
    # for n, p in m.named_parameters():
    #     if p.requires_grad:
    #         print(n, p.shape)
    # t = rand(8, 3, 256, 256)
    # # m.eval()
    # # trc.util.toolbox._debug = True
    # t2, ll, _, _, _, _ = m(t, IDQ=False, visual=False, figname=_TMP+"test")
    # # print(len(ll))
    # r = trc.estimate_bpp(ll, num_pixels=256*256)
    # # print(r)
    # # print(F.mse_loss(t2, t))
    # d = (t2 - t).pow(2).flatten(1).mean(1)
    # # print(d.shape, r.shape)
    # rd = d * lmdas.flatten() + r
    # # print(rd.shape)
    # print(cal_num_params(m))
    # rd.mean().add(m.aux_loss()).backward()
    # plot_grad_flow(m.named_parameters(), _TMP+"CC_grad.png")
    # m = trc.hub.AugmentedNormalizedFlowHyperPriorCoder(
    #     96, 320, 192, num_layers=2, use_DQ=False, use_code=True, dec_add=False,
    #     use_context=True, condition='GaussianMixtureModel', use_mean=True)

#     m2 = trc.hub.AugmentedNormalizedFlowHyperPriorCoder(
    # 96, 320, 192, num_layers=2, use_DQ=True, use_code=False, dec_add=False, use_context=True, condition='GaussianMixtureModel', use_mean=True)
#     path = "/work/nctu0756640/log/torch_compression/ANFHyperPriorCoder_0402_1458/"
#     ckpt = torch.load(
#         path+f"model_700.ckpt", map_location="cpu")
#     m2.load_state_dict(ckpt['coder'])
#     m2 = m2.to(__DEVICE__)

#     # trc.util.toolbox._debug = True
#     from example.util.datasets import Kodak, Vimeo90K
#     # ds = Kodak(os.getenv("DATASET")+"Kodak/", transforms.ToTensor())
#     data_transforms = transforms.Compose([
#         transforms.RandomCrop(256),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor()
#     ])
#     # import fnmatch
#     # def find(pattern, path):
#     #     result = []
#     #     for root, dirs, files in os.walk(path):
#     #         for name in files:
#     #             if fnmatch.fnmatch(name, pattern):
#     #                 result.append(root)
#     #     return result

#     # folder = find('1.png', os.getenv("DATASET")+'/vimeo_septuplet/img/')
#     # print(len(folder))
#     # np.save('./folder.npy', folder)

#     # root = os.getenv("DATASET")+'/vimeo_septuplet'
#     # npy = np.load(os.getenv("DATASET")+'/vimeo_septuplet/folder_twcc.npy')
#     # print(npy.shape)
#     # new_npy = []
#     # for d in npy:
#     #     print(d)
#     #     new_npy.append(d.replace('/work/nctu0756640/DataSets/vimeo_septuplet/', ""))

#     # np.save("./folder.npy", new_npy)

#     path = "/work/nctu0756640/log/torch_compression/ANFHyperPriorCoder_0412_2120/"

#     from torch.utils.data import DataLoader
#     ds = Vimeo90K(os.getenv("DATASET") + "vimeo_septuplet/", data_transforms)
#     m = m.to(__DEVICE__)
#     dl = DataLoader(ds, 32, num_workers=16)
#     # values = {}


#     # df = DataFrame(columns=["path", "mse", "rate", "rdloss"])
#     # from torch.nn.parallel import data_parallel
#     optim = torch.optim.Adam(m.parameters(), lr=1e-4)
#     trc.util.toolbox._check_grad = True

# # with torch.no_grad():
#     for idx in range(2626, 2634):
#         # print(path)
#         print(idx)
#         imgs = []
#         with open(path+"path_"+str(idx)+".txt", 'r') as fp:
#             for line in fp.read().splitlines():
#                 random.seed(1)
#                 t = random.randint(0, 6)

#                 img = data_transforms(imgloader(line))
#                 imgs.append(img)
#         img = torch.stack(imgs)
#         # print(img.shape)
#         img = img.to(__DEVICE__)

#         optim.zero_grad()
#         ckpt = torch.load(
#             path+f"model_{idx%100}.ckpt", map_location=__DEVICE__)
#         m.load_state_dict(ckpt['coder'])
#         rec, ll, Yerr, _, _ = m(img, visual=idx, figname=_TMP+f"rerun_{idx}")
#         print(check_range(rec, "rec"))
#         rec = trc.util.bound(rec, 0, 1)
#         print(check_range(rec, "rec"))
#         mse = (rec - img).pow(2).flatten(1).mean(1)
#         rate = trc.estimate_bpp(ll, input=img)
#         rd_loss = mse * (255**2 * 5e-2) + rate
#         rd_loss.mean().backward()

#         trc.AugmentedNormalizedFlows.dump_grad(_TMP+f"grad_log_{idx}.txt")
#         save_image(torch.cat([img, rec]), _TMP +
#                    f"rerun_{idx}.png", nrow=img.size(0))
#         print(rd_loss, rd_loss.mean(), "\n")

#         with torch.no_grad():
#             rec, ll, Yerr, _, _ = m2(img)
#             print(check_range(rec, "rec"))
#             mse = (rec - img).pow(2).flatten(1).mean(1)
#             rate = trc.estimate_bpp(ll, input=img)
#             rd_loss = mse * (255**2 * 5e-2) + rate
#             print(rd_loss, rd_loss.mean(), "\n")
#         # print(mse.shape, rate.shape)
#         # for b in range(img.size(0)):
#         #     # values[path[b]] = (mse[b], rate[b], rd_loss[b])
#         #     df.loc[len(df)] = [path[b], mse[b].item(),
#         #                         rate[b].item(), rd_loss[b].item()]

#         # if idx == 5:
#         #     break

    # print(df)
    # df.to_csv("./data.csv")

    # df = pd.read_csv("./data.csv", index_col=0)
    # plt.hist(df["rate"], bins=np.arange(0, 10, 0.001).tolist())
    # plt.xlabel("Rate (bpp)")
    # plt.ylabel("Counting")
    # plt.title("Vimeo_data_analisys")
    # plt.show()
    # print(df)
    # df = df.sort_values(by=["rdloss"])
    # print(df)

    # # fig = plt.figure()
    # batch_size = 128
    # count = 0
    # mean, var = [], []
    # ara = list(range(len(df)))
    # bins = np.arange(0, 11, 0.1)
    # for _ in range(10000):
    #     idx = sorted(random.sample(ara, batch_size))
    #     # print(idx)
    #     # df_sample = df.loc[idx].sort_values(by=["rdloss"])
    #     df_sample = df.loc[idx]
    #     # print(df)
    #     rd_loss = df_sample["rdloss"]
    #     # plt.hist(rd_loss, bins=bins)
    #     m, v = np.mean(rd_loss), np.var(rd_loss)
    #     # print(m, v)
    #     mean.append(m)
    #     var.append(v)
    #     count += batch_size

    #     if count > len(df):
    #         break

    # # plt.savefig(_TMP+"random_rd_%d.png" % batch_size)
    # # plt.close(fig)

    # fig = plt.figure()
    # plt.plot(np.arange(len(mean)), mean, label="mean_%.3f" % np.var(mean))
    # plt.plot(np.arange(len(mean)), var, label="var")
    # plt.legend()
    # plt.savefig(_TMP+"random_mv_%d.png" % batch_size)
    # plt.close(fig)

    # q = Queue(10)
    # print(q.mean())
    # for i in range(20):
    #     q.put(i)
    #     print(q.mean())

    # for e in range(4):
    #     for idx, img in enumerate(ds):
    #         print(img.shape)
    #         _ = m(img.unsqueeze(0), visual=e if e else -1, figname=_TMP+"test"+str(idx))
    #         if idx == 1:
    #             break

    # m = nn.Conv2d(2, 2, 1, groups=2, bias=False)
    # print(m.weight)
    # print(m.weight.shape)
    # t = torch.ones(1, 2, 1, 1)
    # t2 = m(t)
    # print(t2)
    # print(t2.shape)

    # dim = 5
    # mask = torch.ones(dim, dim).tril_(diagonal=-1)
    # print(mask)
    # mask = mask[:, :, None, None]

    # t = rand(1, dim, 4, 4)
    # # print(t)
    # t2 = F.unfold(t, 3, padding=1).view(1, dim, -1, 4, 4)
    # # print(t2, t2.shape)
    # print(t2.shape)

    # que = rand_like(t2)

    # score = torch.einsum("bckhw, bdkhw->bcdhw", t2, que)
    # print(score.shape)
    # score = score.mul(mask).softmax(2).mul(mask)
    # # print(score)

    # dim = 192
    # kernel = 5
    # padding = (kernel - 1) // 2
    # t = rand(2, dim, 32, 48)
    # # print(t)
    # # m = trc.modules.attention_module.NonLocalBlock(dim, block_size=1)
    # m = trc.hub.AttentionBlock(dim, non_local=False)
    # # m = trc.hub.AttentionBlock(dim, non_local=True)
    # # m = trc.hub.TriAttentionBlock(dim)
    # # m = trc.modules.attention_module.TripleAttentionModule()
    # # m = trc.modules.attention_module.Involution2d(dim, dim, 7, padding=3, reduce_ratio=4, sigma_mapping=nn.LeakyReLU(inplace=True))
    # print(m)
    # print(cal_num_params(m))
    # t2 = m(t)
    # print(t2.shape)

    # flt = torch.zeros(kernel, kernel, dim, dim*kernel*kernel)
    # for i in range(0, kernel):
    #     for j in range(0, kernel):
    #         for k in range(0, dim):
    #             s = k*kernel*kernel + i * kernel + j
    #             flt[i, j, k, s] = 1

    # print(flt.shape)
    # flt = flt.permute(3, 2, 1, 0)
    # print(flt.shape)
    # # t2 = F.conv2d(t, flt, padding=(padding, padding))
    # # t2 = t2.view(-1, dim, kernel, kernel, 4, 4).permute(0,
    # #                                                     3, 2, 1, 4, 5).flatten(1, 3)
    # # print(t2)
    # # print(t2.shape)

    # t3 = F.unfold(t, kernel, 1, padding, 1).view(-1, dim, kernel, kernel, 4, 4)
    # print(t3)
    # print(t3.shape)
    # # print(t2.equal(t3))

    # t4 = t3.permute(0, 4, 5, 1, 2, 3).flatten(0, 2)
    # print(t4.shape)

    # m = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1),
    #                   nn.Conv2d(dim, dim, 3, padding=1, stride=2),
    #                   nn.Conv2d(dim, dim, 3, padding=1),
    #                   Flatten(1),
    #                   nn.Linear(dim*((kernel+1)//2)**2, dim*2))

    # t5 = m(t4)
    # print(t5.shape)

    # t6 = t5.view(1, 4, 4, -1).permute(0, 3, 1, 2)
    # print(t6.shape)

    # m = PredictionModel(dim, kernel_size=5)
    # print(m)
    # print(cal_num_params(m))
    # condition = m(t)
    # print(condition.shape)

    # mean = torch.Tensor([.406, .456, .485]).view(-1, 1, 1).to(__DEVICE__)
    # var = torch.Tensor([.225, .224, .229]).view(-1, 1, 1).to(__DEVICE__)
    # t = randn(10).mul(57).add(128).div(255)
    # print(t)
    # t2 = randn_like(t).mul(57).add(128).div(255)
    # l1 = F.l1_loss(t * 255, t2 * 255, reduction="none")
    # print(l1, l1.mean())
    # l2 = F.mse_loss(t * 255, t2 * 255, reduction="none")
    # print(l2, l2.mean())
    # from example.util.loss import huber_loss
    # hl = huber_loss(t * 255, t2 * 255, 50, reduction="none")
    # print(hl, hl.mean())
    # hl = huber_loss(t * 255, t2 * 255, 25, reduction="none")
    # print(hl, hl.mean())

    # t = randn(2, 4)
    # print(t)
    # m = trc.Swish(num_parameters=1, init=100)
    # print(m)
    # t2 = m(t)
    # print(t2)
    # plot_func(m, shape=(1, 4, 1000))

    # C_in, C_out = 128, 192
    # t = rand(4, C_in, 256, 256).to(__DEVICE__)
    # m = trc.modules.involution.Involution2d(C_in, C_out, kernel_size=5, padding=2, stride=2, reduce_ratio=2,
    #                                         sigma_mapping=nn.LeakyReLU(inplace=True)).to(__DEVICE__)
    # print(m)
    # t2 = m(t)
    # print(t2.shape)
    # print(cal_num_params(m))
    # for _ in trange(1000):
    #     t2 = m(t)

    # m = nn.Conv2d(C_in, C_out, kernel_size=5,
    #               padding=2, stride=2).to(__DEVICE__)
    # print(m)
    # t2 = m(t)
    # print(t2.shape)
    # print(cal_num_params(m))
    # for _ in trange(1000):
    #     t2 = m(t)

    # m = trc.GoogleHyperPriorCoder(4, 4, 4)
    # optim = torch.optim.Adam(m.parameters(), lr=1e-4)
    # print(optim)
    # sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, verbose=True)
    # sched.step(1e-3)
    # sched.step(1e-3)
    # sched.step(1e-3)
    # sched.step(1e-4)
    # sched.step(1e-3)
    # sched.step(1e-3)
    # sched.step(1e-3)
    # sched.step(1e-3)
    # sched.step(1e-3)
    # sched.step(1e-3)
    # sched.step(1e-3)
    # sched.step(1e-3)
    # sched.step(1e-3)
    # sched.step(1e-3)
    # sched.step(1e-3)
    # sched.step(1e-3)
    # sched.step(1e-3)
    # print(sched.state_dict())
    # print("patience={patience}, best={best}, num_bad_epochs={num_bad_epochs}, cooldown_counter={cooldown_counter}".format(
    #     **sched.state_dict()))

    # loss = rand(32)
    # print(loss)
    # print(loss.var())
    # m = nn.BatchNorm1d(1, affine=False)
    # l2 = m(loss.view(-1, 1)).view(-1)
    # print(l2)
    # print(l2.var())

    # coder = trc.GoogleHyperPriorCoder(192, 320, 192).to(__DEVICE__)
    # optim = torch.optim.Adam([dict(params=coder.main_parameters(), lr=1e-4),
    #                           dict(params=coder.aux_parameters(), lr=1e-3)])

    # from example.util.loss import LossNorm
    # lossnorm = LossNorm().to(__DEVICE__)
    # mean = torch.Tensor([.406, .456, .485]).view(-1, 1, 1).to(__DEVICE__)
    # var = torch.Tensor([.225, .224, .229]).view(-1, 1, 1).to(__DEVICE__)

    # pbar = tqdm(np.arange(10000))
    # for _ in pbar:
    #     t = randn(32, 3, 256, 256).to(__DEVICE__)
    #     # t = t.mul(var).add(mean)

    #     optim.zero_grad()
    #     t2, ll = coder(t)
    #     rdLoss = torch.sub(t, t2).pow(2).flatten(1).mean(
    #         1)*(255 ** 2 * 5e-2) + trc.estimate_bpp(ll, input=t)
    #     aux_loss = coder.aux_loss()
    #     var = rdLoss.var()
    #     norm_rdLoss = lossnorm(rdLoss)
    #     var2 = norm_rdLoss.var()
    #     totalLoss = rdLoss.mean().add(aux_loss)
    #     totalLoss.backward()

    #     optim.step()
    #     pbar.set_description_str(
    #         "  {:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}".format(rdLoss.mean(), aux_loss, var, norm_rdLoss.mean(), var2))

    # t = rand(2, 4, 2, 2)
    # m = nn.AlphaDropout(0.2)
    # print(t)
    # print(m(t))
    # m2 = nn.Dropout2d(0.2)
    # print(m2(t))

    # plot_func(trc.modules.activation.swish)
    # trc.util.toolbox._check_grad = True

    # m = trc.modules.context_model.MaskedConv2d(192, 384, 5)
    # print(cal_num_params(m))
    # print(m._mask)
    # t = rand(1, 192, 16, 16)
    # t2 = m(t)
    # print(t2.shape)

    # t = randn(10).requires_grad_(True)
    # print(t)
    # # tt = t.mul(10)
    # # print(tt)
    # # t2 = torch.nn.functional.leaky_relu(tt, 0.01-1, inplace=True)
    # # print(t2)
    # # # print(tt)
    # # # print(t)

    # tt = t.mul(10)
    # t2 = trc.mq_relu(tt, inplace=True)
    # print(t2)
    # print(t)

    # t2.sub(randn_like(t2)).mul(randn_like(t2)).mean().backward()

    # print(t.grad)

    # m = nn.LeakyReLU(0.02, inplace=True)
    # m2 = trc.MQReLU(inplace=True)

    # t = rand(32, 192, 8, 8).to(__DEVICE__)
    # for _ in trange(100000):
    #     m(t)
    #     t.normal_()

    # for _ in trange(100000):
    #     m(t)
    #     t.normal_()

    # for _ in trange(100000):
    #     m2(t)
    #     t.normal_()

    # args = argparse.Namespace()
    # args.num_layers = 2
    # args.alt_step = 1

    # m = trc.hub.AugmentedNormalizedFlowHyperPriorCoder(
    #     128, 320, 192, num_layers=3, use_DQ=True,
    #     use_code=False, share_wei=False, use_QE=True,
    #     use_context=True, condition='GaussianMixtureModel', use_mean=True, quant_mode="SGA",
    # )
    # m = trc.hub.AugmentedNormalizedIntegerDiscreteFlowHyperPriorCoder(
    #     128, 320, 192, num_layers=3, use_DQ=True,
    #     use_code=False, share_wei=False, use_QE=False,
    #     use_context=True, condition='GaussianMixtureModel', use_mean=True, quant_mode="round",
    # )
    # m = trc.GoogleHyperPriorCoder(192, 320, 192, use_mean=False)
    # m = trc.hub.GoogleHyperPriorCoder2(192, 320, 192, use_mean=True)
    # trc.set_default_conv(deconv_type=trc.RSubPixelConv2d)
    # m = trc.CSTKContextCoder(192, 192, 192)
    # m = trc.hub.AugmentedNormalizedFlowCSTKCoder(
    #     128, 320, 192, num_layers=2, use_DQ=True,
    #     use_code=False,
    #     use_context=True, condition='GaussianMixtureModel', use_mean=True)
    # # optim = torch.optim.Adam(m.parameters(), lr=1e-4)
    m = trc.hub.GoogleContextCoder(192, 320, 192, condition="Gaussian")

    # trc.util.toolbox._debug = True
    print(m)
    print(cal_num_params(m))

    # 13380579 Google
    # 21651340 L2 mul
    # 20107497 L2 add CGMM
    # 22636268 L3 add
    # 27082668 CSTK

    # m.eval()
    # for _ in range(2000):
    #     for m_ in m.modules():
    #         if m_.__class__.__name__ == "StochasticGumbelAnnealing":
    #             # print(m_)
    #             m_.step()
    # print(m)
    # t = rand(1, 3, 256, 256).mul(255).round()
    # m(t, visual=True, figname=_TMP+"testQ")
    # # optim.zero_grad()
    # trc.util.toolbox._check_grad = True
    # t2, ll, YE, jac, Y_err = m(t)
    # r1 = trc.estimate_bpp(ll, input=t)
    # r1.add(F.mse_loss(t, trc.util.bound(t2, 0, 1))).backward()
    # grad = trc.util.toolbox.dump_grad()
    # for k, v in grad.items():
    #     print(k, v)
    # for n, p in m.named_parameters():
    #     if p.grad is not None:
    #         print(n, p.grad.mean())
    #         # break

    # print(r1, r2)
    # mode = 0

    # for i in range(3):
    #     mode = switch_mode(m, args, i, mode)
    #     print(mode, m.num_layers, "\n\n")
    #     t2, ll, YE, jac, Y_err = m(t)
    #     check_shape(t2)
