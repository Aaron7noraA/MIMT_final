#! /Users/gcwhiteshadow/anaconda3/bin/python3
# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

matplotlib.rcParams['text.usetex'] = False


class RD_Curve:
    def __init__(self, data=None, index=['PSNR', 'MS-SSIM', 'bpp']):
        assert len(index) != 0
        assert any('PSNR' in s for s in index)
        assert any('MS-SSIM' in s for s in index)
        assert any('bpp' in s for s in index)

        self.index = index
        self.points = np.empty((0, len(self.index)))

        '''
        PSNR | MS-SSIM | bpp | ext....
        ===========================...
             |         |     |     ...
             |         |     |     ...
             |         |     |     ...
             |         |     |     ...
        '''

        if data is not None:
            self.add_points(data)

    def add_points(self, points: list):
        points_np = np.array(points)
        assert ((len(points_np.shape) == 1 and points_np.shape[0] == len(self.index)) or
                (len(points_np.shape) == 2 and points_np.shape[1] == len(self.index)))

        if len(points_np.shape) == 1:
            points_np = np.expand_dims(points_np, 0)

        '''
        [   [psnr_1, ms_ssim_1, bpp_1, ext.....],
            .
            .
            .
        ]
        '''

        self.points = np.concatenate([self.points, points_np], axis=0)

    def sorted(self):
        order = sorted(enumerate(self.bpp), key=lambda x: x[1])
        self.points = self.points[list(idx for idx, _ in order)]

    def get_series(self, index_name: str):
        assert any(index_name in s for s in self.index)
        dict_name = {self.index[i]: i for i in range(0, len(self.index))}

        return self.points[:, dict_name[index_name]].astype(np.float32)

    @property
    def PSNR(self):
        return self.get_series('PSNR')

    @property
    def MS_SSIM(self):
        return self.get_series('MS-SSIM')

    @property
    def MS_SSIM_dB(self):
        MS_SSIM = self.MS_SSIM
        if MS_SSIM[0] <= 1:
            return -10*np.log10(1-MS_SSIM)
        return MS_SSIM

    @property
    def bpp(self):
        return self.get_series('bpp')


def BD_RATE(R1, PSNR1, R2, PSNR2, piecewise=0, lower_bound=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # integration interval
    min_int = max(min(PSNR1), min(PSNR2), lower_bound)
    max_int = min(max(PSNR1), max(PSNR2))

    # find integral
    if piecewise == 0:
        # rate method
        p1 = np.polyfit(PSNR1, lR1, 3)
        p2 = np.polyfit(PSNR2, lR2, 3)

        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
        
        # find avg diff
        avg_exp_diff = (int2-int1)/(max_int-min_int)
        avg_diff = (np.exp(avg_exp_diff)-1)*100
        return avg_diff, p1, p2
    else:
        samples, interval = np.linspace(
            min_int, max_int, num=100, retstep=True)

        v1 = scipy.interpolate.pchip_interpolate(
            np.sort(PSNR1), lR1[np.argsort(PSNR1)], samples)
        v2 = scipy.interpolate.pchip_interpolate(
            np.sort(PSNR2), lR2[np.argsort(PSNR2)], samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

        # find avg diff
        avg_exp_diff = (int2-int1)/(max_int-min_int)
        avg_diff = (np.exp(avg_exp_diff)-1)*100
        return avg_diff

class RD_Plot:
    def __init__(self, name, metric='PSNR', anchor=None, xlim=None, ylim=None, fontsize=16, interp='linear', 
                 colors='default', styles='default', print_style='poly', figsize=(12, 8)):
        assert metric in ['PSNR', 'MS_SSIM', 'MS_SSIM_dB']
        self.metric = metric
        
        assert print_style in ['poly', 'linear']
        self.print_style = print_style
        
        self.anchor = anchor
        self.kind = interp
        matplotlib.rcParams.update({'font.size': fontsize})
        self.buffer = plt.figure(figsize=figsize, num="buf")
        self.bx = self.buffer.add_subplot(111)
        self.fig = plt.figure(figsize=figsize, num="R_D")

        self.curves = []
        self.colors = colors if colors != "default" else plt.rcParams['axes.prop_cycle'].by_key()[
            'color']
        self.color_count = 0
        self.styles = styles if styles != "default" else ["-"]
        self.style_count = 0
        self.ax = self.fig.add_subplot(111)
        
        if xlim is not None:
            self.ax.set_xlim(*xlim)
        if ylim is not None:
            self.ax.set_ylim(*ylim)
#         major_ticks = np.arange(0, xlim[1], 0.05)
#         minor_ticks = np.arange(0, xlim[1], 0.01)
#         major_ticks = np.arange(xlim[0], xlim[1], 0.1)
#         minor_ticks = np.arange(xlim[0], xlim[1], 0.01)
#         self.ax.set_xticks(major_ticks)
#         self.ax.set_xticks(minor_ticks, minor=True)
        
#         major_ticks = np.arange(33, ylim[1], 1)
#         minor_ticks = np.arange(33, ylim[1], 0.2)
#         self.ax.set_yticks(major_ticks)
#         self.ax.set_yticks(minor_ticks, minor=True)
        
#         self.ax.grid(which='both')
#         self.ax.grid(which="major",alpha=0.2)
#         self.ax.grid(which="minor",alpha=1)

        plt.title('{} Dataset'.format(name))
        plt.xlabel("Bit-rate (bpp)")
        y_label = metric.replace("_dB", "")+"-RGB" + (" (dB)" if metric in ["PSNR", "MS_SSIM_dB"] else "")
        y_label = y_label.replace("MS_SSIM", "MS-SSIM")
        plt.ylabel(y_label)
        plt.grid()


    def add_curve(self, curve: RD_Curve, label='str',
                  color=None, style=None, width=None, marker=None, 
                  fillstyle='none', markersize=10, piecewise=0):
#                   color=None, style=None, width=2, marker=None):
        curve.sorted()
        self.curves.append(curve)
        bpp = curve.bpp
        value = getattr(curve, self.metric)
        assert self.anchor is not None
        if self.metric == 'PSNR':
            lb = 29.5
        elif self.metric == 'MS_SSIM':
#             lb = 0.964900
            lb = 0
        else:
            lb = 0
        # bd_rate, _, poly_coef = BD_RATE(self.anchor.bpp, 
        #                          getattr(self.anchor, self.metric),
        #                          bpp, 
        #                          value, 
        #                          piecewise=piecewise,
        #                          lower_bound=lb)
        # if curve != self.anchor:
        #     bd_rate = " (%.1f)" % bd_rate  
        # else:
        #     bd_rate = " (anchor)"
        # label += bd_rate
            
        if color is None:
            color = self.colors[self.color_count]
            self.color_count = (self.color_count+1) % len(self.colors)
        if style is None:
            style = self.styles[self.style_count]
            self.style_count = (self.style_count+1) % len(self.styles)
        
        if self.print_style == 'poly':
#             x = np.linspace(min(bpp), max(bpp), 100)
#             model = scipy.interpolate.interp1d(bpp, value, self.kind)
#             plt.figure('buf')
#             plt.plot(x, model(x), ls=style, c=color, marker=marker, label=label,
#                      lw=width, markevery=5, markersize=8)
#             plt.figure('R_D')
#             plt.plot(x, np.polyval(poly_coef, x), ls=style, c=color, marker=None,
#                      lw=width, zorder=1)
            y = np.linspace(min(value), max(value), 100)
            model = np.poly1d(poly_coef)
            plt.figure('buf')
            plt.plot(np.exp(model(y)), y, ls=style, c=color, marker=marker, label=label,
                     lw=width, markevery=5, markersize=8)
            plt.figure('R_D')
            plt.plot(np.exp(model(y)), y, ls=style, c=color, marker=None,
                     lw=width, zorder=1)
        elif self.print_style == 'linear':
            plt.figure('buf')
            plt.plot(bpp, value, ls=style, c=color, marker=marker, label=label,
                     lw=width, markevery=5, markersize=8)
            plt.figure('R_D')
            plt.plot(bpp, value, ls=style, c=color, marker=None,
                     lw=width, zorder=1)
            # plt.plot(bpp, value, ls=style, c=color, marker=marker, label=label,
            #          lw=width, markevery=5, markersize=8)
        else:
            raise ValueError
        if marker is not None:
            plt.scatter(bpp, value, c=color,
                        marker=marker, zorder=2, s=100)

    def plot_curves(self):
        pass

    def add_legend(self):
        handles, labels = self.bx.get_legend_handles_labels()
        self.ax.legend(handles[::-1], labels[::-1],
                       loc='lower right', prop={'size': 18})

    def plot(self):
        for c in self.curves:
            pass
        self.add_legend()
        plt.show()

    def save_figure(self, filename: str):
        self.add_legend()
        plt.savefig(filename)
        plt.close(self.fig)
        plt.close(self.buffer)


x265_veryslow = RD_Curve()
x265_veryslow.add_points([
    [38.91692, 0.98434, 1.14416],
    [37.06323, 0.97784, 0.61154],
    [34.72736, 0.96675, 0.18405],
    [32.80642, 0.95331, 0.06781],
    [30.88697, 0.93183, 0.03160],
])

GS_nought = RD_Curve()
GS_nought.add_points([
    [35.1477, None, 0.1707],
    [34.0749, None, 0.1107],
    [33.0888, None, 0.0793],
    [31.5979, None, 0.0617]
])

nought_16_channels_without_trainMC = RD_Curve()
nought_16_channels_without_trainMC.add_points([
    [34.9010, None, 0.2282]
])

Reproduce_GS_nought = RD_Curve()
Reproduce_GS_nought.add_points([
    [35.1393, None, 0.1699]
])

nought_64_channels_without_trainMC = RD_Curve()
nought_64_channels_without_trainMC.add_points([
    [34.9603, None, 0.2125]
])

nought_16_channels = RD_Curve()
nought_16_channels.add_points([
    [35.0936, None, 0.2258]
])

nought_3_channels_without_trainMC = RD_Curve()
nought_3_channels_without_trainMC.add_points([
    [34.8146, None, 0.2386]
])

mc_16_channels = RD_Curve()
mc_16_channels.add_points([
    [34.8460, None, 0.2052]
])

nought_mc_decode_cond = RD_Curve()
nought_mc_decode_cond.add_points([
    [34.9256, None, 0.2480]
])

mc_mc_decode_cond = RD_Curve()
mc_mc_decode_cond.add_points([
    [34.6722, None, 0.2315]
])

mimic_DCVC = RD_Curve()
mimic_DCVC.add_points([
    [35.1373, None, 0.2087],
])

tgtIn_PredPrior_ANFIC = RD_Curve()
tgtIn_PredPrior_ANFIC.add_points([
    [35.2517, None, 0.1704],
    [34.1702, None, 0.1022],
    [32.9246, None, 0.0626],
    [31.3260, None, 0.0379],
])

DCVC = RD_Curve()
DCVC.add_points([
    [34.417480, None, 0.135718],
    [33.438904, None, 0.083026],
    [32.213693, None, 0.053624],
    [30.968304, None, 0.035117]
])

DVC_condInter_1L_ANFIC = RD_Curve()
DVC_condInter_1L_ANFIC.add_points([
    [35.2975, None, 0.1898],
    [34.3159, None, 0.1148],
    [33.1887, None, 0.0717],
    [31.7229, None, 0.0434],
])

def main_plot(mode="PSNR"):
    if mode == "PSNR":
        fig_plot = RD_Plot("HEVC_B", metric="PSNR",
                           anchor=x265_veryslow, xlim=(0.13, 0.32), ylim=(33, 36), fontsize=22,
                           interp='quadratic', styles=['-', '--', '-.', ":"],
                           print_style='linear')

        # fig_plot.add_curve(x265_veryslow, 
        #                    color='black', label='x265 (veryslow)', marker='o', fillstyle='full')
        fig_plot.add_curve(GS_nought,
                           marker="s", label='3 ch: w pretrain', style='-', color='#00fff7')
        # fig_plot.add_curve(Reproduce_GS_nought,
        #                    marker='s', label='reproduce', style='-', color='#444444')
        # fig_plot.add_curve(nought_16_channels_without_trainMC,
        #                    marker='s', label='16 ch: w/o pretrain', style='-', color='#f12c11')
        # fig_plot.add_curve(nought_64_channels_without_trainMC,
        #                    marker='+', label='64 ch: w/o pretrain', style='-.', color='#4DA2E7')
        fig_plot.add_curve(nought_16_channels,
                           marker='s', label='16 ch : w pretrain', style='-.', color='#f12c11')
        # fig_plot.add_curve(mc_16_channels,
        #                    marker=">", label='16 ch (mc): w pretrain', style='-', color='#9118de')
        # fig_plot.add_curve(nought_mc_decode_cond,
        #                    marker=">", label='(f)', style='-', color='#f009ee')
        # fig_plot.add_curve(mc_mc_decode_cond,
        #                    marker=">", label='(g)', style='-', color='#c8e30d')
        fig_plot.add_curve(mimic_DCVC,
                           marker='+', label='64 ch: w pretrain', style='-', color='#4DA2E7')
        # fig_plot.add_curve(nought_3_channels_without_trainMC,
        #                    marker='>', label='3 ch: w/o pretrain', style='-', color='#c8e30d')
        fig_plot.add_curve(DCVC, marker='h', 
                            color='#f98b24', label='DCVC (NIPS\'21)', style=(0,(3,1,1,1)))        
        # fig_plot.add_curve(tgtIn_PredPrior_ANFIC,
        #                    marker='>', label='CANF-VC$^-$(DVC)', style='-', color='#598745')
        # fig_plot.add_curve(DVC_condInter_1L_ANFIC,
        #                    marker='>', label='1L CANF-VC(DVC)', style='-', color='#00ff9d')
        fig_plot.save_figure(fig_dir+"RD_HEVC_B_PSNR_CANFVC_GRU_GOP32_w_pretrain.png")



""
if __name__ == "__main__":
    import os
    fig_dir = os.getenv("HOME")+"/RD_plot/"
    main_plot("PSNR")
    main_plot("SSIM")
