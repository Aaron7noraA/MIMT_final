import argparse
import math
import os
import random
import shutil
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import cat, nn, rand, rand_like, randn, randn_like, stack
from tqdm import tqdm, trange

import dataloader as D
import flownets as ME
import models as M
import scale_module
import torch_compression as trc
import util.flow_utils as FU
import util.functional as FE
import util.sampler as S
import util.vision as V
from util.vision import load_image, save_image

__testseq__ = './BasketballPass/'
__DEVICE__ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def test_flownet(net='spy'):
    m = ME.SPyNet() if net == 'spy' else ME.PWCNet()
    transform = transforms.Compose([
        V.CTUCrop(64),
        transforms.ToTensor()
    ])

    f0 = load_image(__testseq__+'f059.png', transform=transform).to(__DEVICE__)
    f1 = load_image(__testseq__+'f060.png', transform=transform).to(__DEVICE__)
    print(m)
    print(f0.shape, f1.shape)
    m = m.to(__DEVICE__)

    t0 = perf_counter()
    for _ in range(10000):
        flow = m(f0, f1)
    print(perf_counter() - t0)
    # print(flow.shape)
    # plot_flow = FU.PlotFlow().to(__DEVICE__)
    # flowmap = plot_flow(flow)
    # print(flowmap.shape)
    # save_image(flowmap, _TMP+'flow.png')


def test_OpenDVC():
    from OpenDVC_train_PSNR import Pframe
    m = Pframe().to(__DEVICE__)

    transform = transforms.Compose([
        V.CTUCrop(16),
        transforms.ToTensor()
    ])

    f0 = load_image(__testseq__+'f059.png', transform=transform).to(__DEVICE__)
    f1 = load_image(__testseq__+'f060.png', transform=transform).to(__DEVICE__)
    print(m)
    print(f0.shape, f1.shape)

    rec, rate = m(f0, f1)[:2]
    print(rec.shape, rate.shape)


def test_dataloader(I_QP=22):

    transformer = transforms.Compose([
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    dataset = D.VideoDataIframe(os.getenv(
        'DATASET')+'vimeo_septuplet/', 'BPG_QP'+str(I_QP), frames=2, transform=transformer)
    traindata = D.DataLoader(dataset, 4, shuffle=True,
                             drop_last=True, num_workers=16)

    for (idx, gop) in enumerate(traindata):
        print(gop.shape)
        save_image(V.compare_img(gop), _TMP+'gop{}.png'.format(idx), nrow=4)

        if idx == 2:
            break


def test_scale_space():
    t = load_image(__testseq__+'f059.png')

    v = scale_module.make_inplace_pool(t)
    print(v.shape)
    save_image(v.transpose(1, 2).flatten(0, 1), _TMP+'pool.png')
    scale = torch.empty_like(t)
    scalel, scaler = scale.chunk(2, dim=-1)
    scalel.fill_(0)
    scaler.fill_(1)
    print(scale)

    t2, scale = scale_module.scale_space(t, scale, vstack=False)
    print(t2.shape, scale.shape)
    save_image(t2, _TMP+"blured.png")


def ICNR_(weight, bias, out_channels, scale_factor, mode='nearest'):
    C = out_channels // scale_factor ** 2
    k1 = weight[:C]
    print(k1.shape)

    k2 = F.interpolate(k1.transpose(
        0, 1), scale_factor=scale_factor, mode=mode)
    print(k2.shape)

    k3 = FE.space_to_depth(k2, scale).transpose(0, 1)
    print(k3.shape)

    weight.data.copy_(k3)

    k1 = bias[:C]

    k2 = F.interpolate(k1.transpose(
        0, 1), scale_factor=scale_factor, mode=mode)
    print(k2.shape)

    k3 = FE.space_to_depth(k2, scale).transpose(0, 1)
    print(k3.shape)

    bias.data.copy_(k3)


def bicubic_grid_sample(input, grid, times, padding_mode: str = 'zeros', align_corners: bool = False):
    """bicubic_grid_sample"""
    kernel_size = 4
    t0 = perf_counter()
    if not align_corners:
        grid = grid * FE.getWH(input) / FE.getWH(input).sub_(1)
    center = FE.center_of(input)
    abs_loc = ((grid + 1) * center).unsqueeze(-1)
    torch.cuda.synchronize()
    times[0] += (perf_counter() - t0)

    # print(abs_loc.shape)
    offset = torch.tensor([-1, 0, 1, 2], device=grid.device)
    locs = abs_loc.floor() + offset

    t0 = perf_counter()
    # loc_grid = torch.meshgrid(offset, offset)
    # loc_grid = torch.stack([loc_grid[1], loc_grid[1]], dim=0)
    # loc_grid = abs_loc.floor().unsqueeze(-1) + loc_grid
    # loc_grid = loc_grid.flatten(0, 2).permute(0, 2, 3, 1)
    # print(loc_grid)
    # print(loc_grid.shape)
    loc_w, loc_h = locs.detach().flatten(0, 2).unbind(dim=-2)
    loc_w = loc_w.reshape(-1, 1, kernel_size).expand(-1, kernel_size, -1)
    loc_h = loc_h.reshape(-1, kernel_size, 1).expand(-1, -1, kernel_size)
    loc_grid = torch.stack([loc_w, loc_h], dim=-1)
    # print(loc_grid)
    # print(loc_grid.shape)
    loc_grid = loc_grid.reshape(grid.size(0), -1, 1, 2)/center - 1
    torch.cuda.synchronize()
    times[1] += (perf_counter() - t0)

    t0 = perf_counter()
    selected = F.grid_sample(input, loc_grid.detach(), mode='nearest',
                             padding_mode=padding_mode, align_corners=True)
    patch = selected.view(input.size()[:2]+grid.size()[1:3]+(kernel_size,)*2)
    torch.cuda.synchronize()
    times[2] += (perf_counter() - t0)

    t0 = perf_counter()
    mat_r, mat_l = S.u(torch.abs(abs_loc - locs.detach())).unbind(dim=-2)
    torch.cuda.synchronize()
    times[3] += (perf_counter() - t0)
    t0 = perf_counter()
    output = torch.einsum('bhwl,bchwlr,bhwr->bchw', mat_l, patch, mat_r)
    # output = mat_l.unsqueeze(
    #     1).unsqueeze(-2) @ patch[:, 0] @ mat_r.unsqueeze(1).unsqueeze(-1)
    torch.cuda.synchronize()
    times[4] += (perf_counter() - t0)
    return output


def test_scale_space3d():

    t = load_image(__testseq__+"f080.png", (256, 256)).to(__DEVICE__)
    t2 = load_image(__testseq__+"f081.png", (256, 256)).to(__DEVICE__)
    print(t.shape)
    m = ME.SPyNet().to(__DEVICE__)
    flow2d = m(t, t2)
    print(flow2d.shape)
    scale = randn_like(flow2d[:, :1]).requires_grad_(True)
    flow = torch.cat([flow2d, scale], dim=1)
    print(flow.shape)

    t4 = S.warp(t, flow2d)
    print(F.mse_loss(t4, t2).item())
    # print(t4.shape)
    # save_image(t4, _TMP+'F1_warp2d.png')[0, 0.5, 1, 2, 4, 8]

    ms = scale_module.ScaleSpace3d(
        sigmas=[0, 1, 2, 4, 8, 16], kernel_sizes=[1, 3, 3, 3, 5, 5])

    optim = torch.optim.Adam([scale], lr=1e-3)
    losses = []
    pbar = tqdm(range(100000))
    for _ in pbar:
        optim.zero_grad()
        flow = torch.cat([flow2d, scale], dim=1)
        t3 = ms(t, flow)
        loss = F.mse_loss(t3, t2)
        loss.add(ms.activation(scale).abs().mean()*0.1).backward()
        optim.step()
        pbar.set_description_str(f"{loss.item():.3e}")
        losses.append(loss.item())

    print(t3.shape)
    # save_image(t, _TMP+'F0.png')
    # save_image(t2, _TMP+'F1.png')
    save_image(t3, _TMP+'F1_warp2.png')
    # flowmap = V.PlotFlow().to(__DEVICE__)(flow2d)
    # save_image(flowmap, _TMP+"F01.png")

    fig = plt.figure()
    plt.plot(losses)
    plt.savefig(_TMP+"scale_optim2.png")
    plt.close(fig)

    scalemap = ms.activation(scale).abs()
    save_image(scalemap, _TMP+"scale_map2.png")


if __name__ == "__main__":
    _TMP = './tmp/'
    os.makedirs(_TMP, exist_ok=True)
    print(trc.__version__)
    # FE.torchseed(666)
    # test_flownet('spy')
    # test_OpenDVC()
    # test_dataloader()
    # test_scale_space()

    a = ()
    b = (1, 2)
    print(a+b)

    # t = load_image(__testseq__+"f080.png", (256, 256)).to(__DEVICE__)
    # t2 = load_image(__testseq__+"f081.png", (256, 256)).to(__DEVICE__)
    # print(t.shape)

    # m = ME.PWCNet()
    # print(m)

    # m2 = ME.PWCNet3d()
    # print(m)

    # flow2d = m(t, t2)
    # flow3d = m2(t, t2)
    # print(flow2d.shape, flow3d.shape)
    # print(F.mse_loss(flow2d, flow3d[:, :2]))
    # print(flow3d[:, -1].mean())

    # m = M.MC_Net()
    # print(m)
    # count = 0
    # for p in m.parameters():
    #     count += p.numel()
    # print(count)
    # t = rand(1, 3, 6, 6)
    # g = randn(1, 6, 6, 2).tanh()

    # t2 = S.bicubic_grid_sample(t, g)
    # print(t2.shape)

    # C, scale = 3, 2
    # k = rand(C*scale**2, 1, 3, 3)
    # b = rand(C*scale**2, 1, 1)
    # ICNR_(k, b.unsqueeze(1), k.size(0), scale)
    # print(k)
    # print(b.shape)

    # t = rand(2, 3, 8, 6)
    # f = rand(2, 8, 6, 2)

    # assert S.warp2d(t, f).equal(S.warp(t, f))

    # t = rand(2, 3, 4, 8, 6)
    # f = rand(2, 4, 8, 6, 3)

    # assert S.warp3d(t, f).equal(S.warp(t, f))

    # t = rand(1, 12, 100, 100)
    # # print(t)
    # t0 = perf_counter()

    # for _ in range(100):
    #     t2 = F.pixel_shuffle(t, 2)
    #     # print(t2.shape)

    # print(perf_counter() - t0)

    # # t0 = perf_counter()

    # # for _ in range(100):
    # #     t2 = FE.depth_to_space(t, 2)
    # #     # print(t2.shape)

    # # print(perf_counter() - t0)

    # t3 = FE.space_to_depth(t2, 2)
    # # print(t3)

    # # print(t.equal(t3))

    # print(t2)
    # parser = argparse.ArgumentParser(
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # trc.add_coder_args(parser)
    # args = parser.parse_args()
    # print(args.architecture)
    # coder = trc.get_coder_from_args(args)()
    # print(coder)

    # from HPDVC_train_PSNR import Pframe
    # M.use_signalconv = False
    # m = Pframe()
    # print(m)
    # times = [0, 0, 0, 0, 0]

    # B, C, H, W = 4, 3, 1024, 1920
    # # B, C, H, W = 1, 3, 6, 6

    # t = rand(B, C, H, W).to(__DEVICE__)
    # f = randn(B, H, W, 2).to(__DEVICE__).mul(10).tanh()
    # mode = 'bicubic'
    # print(f.max())
    # iter = 1000
    # # t0 = perf_counter()

    # # for _ in range(iter):
    # #     t3 = S.warp_cuda(t, f, mode)

    # # print((perf_counter() - t0)/iter)

    # t0 = perf_counter()

    # for _ in range(iter):
    #     t2 = bicubic_grid_sample(t, f, times)

    # print((perf_counter() - t0)/iter)
    # print(times)

    # print(t2.sub(t3).mean())

    # transform = transforms.Compose([
    #     V.CTUCrop(3),
    #     transforms.ToTensor()
    # ])
    # from functools import partial
    # load_image = partial(load_image, transform=transform, device=__DEVICE__)

    # t = load_image(__testseq__+'f059.png')
    # print(t.shape)
    # frame, GOP = 602, 12
    # for gop_number in range(np.int(np.ceil(frame/GOP))):

    #     f_start, f_end = gop_number * GOP, np.min([gop_number * GOP + GOP, frame])
    #     for f in range(f_start, f_end):
    #         print(f, end=' ')
    #     print()
