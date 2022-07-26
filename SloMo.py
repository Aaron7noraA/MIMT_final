import os
import random
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torchvision.datasets.folder import pil_loader as imgloader
from torchvision.utils import save_image
from tqdm import tqdm, trange

from dataloader import DataLoader, VideoData
from flownets import PWCNet

try:
    from util.sampler import warp
    from util.flow_utils import PlotFlow
except:
    from sampler import warp
    from flow_utils import PlotFlow


_coeffs = {}


def getCoeff(idx, max_idx, device='cpu'):
    """
    Gets flow coefficients used for calculating intermediate optical
    flows from optical flows between I0 and I1: F_0_1 and F_1_0.

    F_t_0 = C00 x F_0_1 + C01 x F_1_0
    F_t_1 = C10 x F_0_1 + C11 x F_1_0

    where,
    C00 = -(1 - t) x t
    C01 = t x t
    C10 = (1 - t) x (1 - t)
    C11 = -t x (1 - t)

    It_gen = (C0 x V_t_0 x g_I_0_F_t_0 + C1 x V_t_1 x g_I_1_F_t_1) / (C0 x V_t_0 + C1 x V_t_1)

    where,
    C0 = 1 - t
    C1 = t

    V_t_0, V_t_1 --> visibility maps
    g_I_0_F_t_0, g_I_1_F_t_1 --> backwarped intermediate frames


    Parameters
    ----------
        indices : tensor
            indices corresponding to the intermediate frame positions
            of all samples in the batch.
        device : device
                computation device (cpu/cuda).

    Returns
    -------
        tensor
            coefficients [[C00, C01, C0], [C10, C11, C1]].
    """
    key = (max_idx, device)
    if key not in _coeffs:
        t = torch.linspace(0, 1, max_idx+1, device=device)[1:-1]
        C0 = 1 - t
        C1 = t
        C11 = C00 = -C0 * C1
        C01 = C1 * C1
        C10 = C0 * C0
        coeff = torch.stack([C00, C01, C0, C10, C11, C1], 1).view(-1, 2, 3)
        _coeffs[key] = coeff

    return _coeffs[key][idx-1]


def bidirectional_warp(input, flow):
    # print(input.shape, flow.shape)
    warped = warp(input, flow)
    return warped.reshape(-1, 2, *warped.size()[1:])


class DownSampleConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__(
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=(kernel_size-1)//2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size,
                      padding=(kernel_size-1)//2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )


class UpSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                   nn.Conv2d(
                                       in_channels, out_channels, 3, padding=1),
                                   nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels*2, out_channels, 3, padding=1),
                                   nn.LeakyReLU(negative_slope=0.1, inplace=True))

    def forward(self, input, skip):
        conv1 = self.conv1(input)
        conv2 = self.conv2(torch.cat([conv1, skip], dim=1))
        return conv2


class UNet(nn.Module):
    """
    A class for creating UNet like architecture as specified by the
    Super SloMo paper.

    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """

    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, padding=3),
            nn.LeakyReLU(
                negative_slope=0.1, inplace=True),
            nn.Conv2d(32, 32, 7, padding=3),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.conv2 = DownSampleConv(32, 64, 5)
        self.conv3 = DownSampleConv(64, 128, 3)
        self.conv4 = DownSampleConv(128, 128, 3)
        self.conv5 = DownSampleConv(128, 128, 3)
        self.conv6 = DownSampleConv(128, 256, 3)
        self.deconv6 = UpSampleConv(256, 128)
        self.deconv5 = UpSampleConv(128, 128)
        self.deconv4 = UpSampleConv(128, 128)
        self.deconv3 = UpSampleConv(128, 64)
        self.deconv2 = UpSampleConv(64, 32)
        self.deconv1 = nn.Conv2d(32, out_channels, 3, padding=1)

    def forward(self, *input):
        conv1 = self.conv1(torch.cat(input, dim=1))
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)

        deconv6 = self.deconv6(conv6, conv5)
        deconv5 = self.deconv5(deconv6, conv4)
        deconv4 = self.deconv4(deconv5, conv3)
        deconv3 = self.deconv3(deconv4, conv2)
        deconv2 = self.deconv2(deconv3, conv1)
        deconv1 = self.deconv1(deconv2)

        return deconv1


class SloMoNet(nn.Module):
    "SloMoNet"

    def __init__(self, version=0):
        super(SloMoNet, self).__init__()
        self.BiFlowNet = UNet(6, 4) if version == 0 else PWCNet()
        self.ArbitraryTimeFlowNet = UNet(20, 5)
        self.version = version

    def getFlowCoeff(self, target_idx, time_range):
        device = next(self.parameters()).device
        return getCoeff(target_idx, time_range, device)[:, :2]

    @ staticmethod
    def getInterpCoeff(target_idx, time_range, device='cpu'):
        return getCoeff(target_idx, time_range, device)[:, -1].view(-1, 1, 1, 1)

    def predict(self, I0, I1, time_range, target_idx=None):
        cated_I = torch.stack((I0, I1), dim=1).flatten(0, 1)
        T = time_range-1

        if self.version == 0:
            flowOut = self.BiFlowNet(I0, I1)
        else:
            flowOut = torch.cat(
                [self.BiFlowNet(I0, I1), self.BiFlowNet(I1, I0)], dim=1)
        F_bi = flowOut.reshape(-1, 2, 2, *flowOut.size()[2:])

        IntrpInput = torch.cat([I0, I1, flowOut], dim=1)

        returns = [F_bi]
        # Generate intermediate frames
        for intermediateIndex in [target_idx] if target_idx is not None else range(1, T):
            coef = self.getFlowCoeff(intermediateIndex, T)

            F_t = torch.einsum('nj,bjfhw->bnfhw', coef, F_bi)

            g_I_F_t = bidirectional_warp(cated_I, F_t.flatten(0, 1))

            intrpOut = self.ArbitraryTimeFlowNet(
                IntrpInput, F_t.flatten(1, 2), g_I_F_t.flatten(1, 2))

            F_t_f = F_t + intrpOut[:, :4].reshape_as(F_t)
            gamma = intrpOut[:, 4:5]
            returns.append(torch.cat([F_t_f.flatten(1, 2), gamma]))

        return returns

    @ staticmethod
    def interpolate(cls, input, Flow, time_range, target_idx):
        """interpolate I_t from I0 and I1 at target_idx/time_range"""
        if isinstance(input, list):
            input = torch.stack(input, dim=1).flatten(0, 1)
        F_t = Flow[:, :4].reshape(-1, 2, 2, *Flow.size()[2:])
        gamma = Flow[:, 4:5]

        g_I_t_f = bidirectional_warp(input, F_t.flatten(0, 1))

        coef = cls.getInterpCoeff(target_idx, time_range-1, F_t.device)
        V_t = torch.sigmoid(gamma)
        weight = coef * torch.stack([V_t, 1 - V_t], dim=1)
        I_t = torch.einsum('bnkhw,bnchw->bchw', weight,
                           g_I_t_f) / weight.sum(1)

        return I_t

    def forward(self, I0, I1, time_range, target_idx=None, timer=None):
        cated_I = torch.stack((I0, I1), dim=1).flatten(0, 1)
        T = time_range-1

        # timer[0].restart()
        if self.version == 0:
            flowOut = self.BiFlowNet(I0, I1)
        else:
            flowOut = torch.cat(
                [self.BiFlowNet(I0, I1), self.BiFlowNet(I1, I0)], dim=1)
        F_bi = flowOut.reshape(-1, 2, 2, *flowOut.size()[2:])
        # timer[0].tap()

        IntrpInput = torch.cat([I0, I1, flowOut], dim=1)

        frames, Flows, warpeds = [], [F_bi], []
        # Generate intermediate frames
        for intermediateIndex in [target_idx] if target_idx is not None else range(1, T):
            # timer[1].restart()
            coef = getCoeff(intermediateIndex, T, I0.device)

            F_t = torch.einsum('nj,bjfhw->bnfhw', coef[:, :2], F_bi)

            g_I_F_t = bidirectional_warp(cated_I, F_t.flatten(0, 1))
            # timer[1].tap()

            # timer[2].restart()
            intrpOut = self.ArbitraryTimeFlowNet(
                IntrpInput, F_t.flatten(1, 2), g_I_F_t.flatten(1, 2))
            # timer[2].tap()

            # timer[3].restart()
            F_t_f = intrpOut[:, :4].reshape_as(F_t)
            V_t = torch.sigmoid(intrpOut[:, 4:5])
            weight = coef[:, -1].view(-1, 1, 1, 1) * \
                torch.stack([V_t, 1 - V_t], dim=1)
            # timer[3].tap()

            # timer[4].restart()
            g_I_F_t_f = bidirectional_warp(cated_I, F_t_f.flatten(0, 1))

            I_t = torch.einsum('bnkhw,bnchw->bchw', weight,
                               g_I_F_t_f) / weight.sum(1)
            frames.append(I_t)
            Flows.append(F_t_f)
            warpeds.append(g_I_F_t_f)
            # timer[4].tap()

        return torch.stack([I0]+frames+[I1], dim=1), Flows, warpeds


class Clock():
    def __init__(self, name):
        self.name = name
        self.step = 0
        self.count = 0

    def restart(self):
        if CIA:
            torch.cuda.synchronize()
        self.t0 = perf_counter()

    def tap(self):
        if CIA:
            torch.cuda.synchronize()
        self.count += perf_counter()-self.t0
        self.step += 1

    def __repr__(self):
        return "Clock:{} spend:{}".format(self.name, self.count/self.step)


def smoothness_loss(Flow):
    return torch.mean(torch.abs(Flow[..., :, :-1] - Flow[..., :, 1:])) + torch.mean(torch.abs(Flow[..., :-1, :] - Flow[..., 1:, :]))


def train(model, args):  # SloMoNet, gop(B, T, C, H, W)
    transformer = transforms.Compose([
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    traindataset = VideoData(
        os.getenv("DATASET", './')+"vimeo_septuplet/", 7, transformer)
    traindata = DataLoader(traindataset, 4, shuffle=True,
                           num_workers=8, drop_last=True)
    model.BiFlowNet.requires_grad_(True)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    t0 = perf_counter()
    clocks = [Clock("BiFlow"), Clock("G_F"), Clock(
        "Arb"), Clock("Refine"), Clock("WeiSum")]
    for epoch in range(10000 if CIA else 1):
        pbar = tqdm(traindata, total=len(traindata))
        for idx, gop in enumerate(pbar):
            time_range = random.randint(7//2, 7)
            gop = gop[:, :time_range].to(__DEVICE__)
            I0 = gop[:, 0]
            I1 = gop[:, -1]

            optim.zero_grad()
            frames, Flows, warpeds = model(I0, I1, time_range, timer=clocks)

            mse = F.mse_loss(frames[:, 1:-1], gop[:, 1:-1])

            W_t01 = bidirectional_warp(torch.stack(
                (I0, I1), dim=1).flatten(0, 1), Flows[0].flatten(0, 1))
            warpLoss = F.l1_loss(W_t01, torch.stack((I1, I0), dim=1))
            for i in range(len(warpeds)):
                warpLoss += F.l1_loss(warpeds[i],
                                      gop[:, i+1:i+2].repeat(1, 2, 1, 1, 1))
            warpLoss /= i+2
            smoothnessLoss = smoothness_loss(Flows[0])

            loss = 204 * mse + 102 * warpLoss + smoothnessLoss
            loss.backward()
            optim.step()

            pbar.set_description_str(
                f" len:{time_range}, mse:{mse.item():.3e}, warp:{warpLoss:.3e}, smooth:{smoothnessLoss:.3e}")

            if (perf_counter() - t0) > 10:
                image = torch.stack([frames, gop], 2).transpose(0, 1).flatten(0, 2)
                save_image(image, "./tmp/SloMo4.png", nrow=gop.size(0)*2)
                flowmap = plot_flow(torch.stack(Flows).flatten(0, 2))
                save_image(flowmap, "./tmp/SloMo_F2.png", nrow=gop.size(0)*2)
                t0 = perf_counter()

            # if idx % 180 == 0:
            #     print(clocks)

        # torch.save(model.state_dict(), "./models/SloMoNet.model")


if __name__ == "__main__":
    torch.random.manual_seed(666)
    torch.cuda.manual_seed(666)
    CIA = torch.cuda.is_available()
    __DEVICE__ = torch.device("cuda:0" if CIA else "cpu")
    plot_flow = PlotFlow().to(__DEVICE__)

    # Initialize model
    m = SloMoNet(version=1)
    print(m)
    count = 0
    for p in m.parameters():
        count += p.numel()
    print(count)
    m = m.to(__DEVICE__)
    # t = torch.rand(2, 3, 64, 64).to(__DEVICE__)
    # t2 = m(t, torch.rand_like(t), time_range=7)
    # print(t2.shape)

    train(m, args=1)
