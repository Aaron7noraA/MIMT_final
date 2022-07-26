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
        bd_rate, _, poly_coef = BD_RATE(self.anchor.bpp, 
                                 getattr(self.anchor, self.metric),
                                 bpp, 
                                 value, 
                                 piecewise=piecewise,
                                 lower_bound=lb)
        if curve != self.anchor:
            bd_rate = " (%.1f)" % bd_rate  
        else:
            bd_rate = " (anchor)"
        label += bd_rate
            
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
                        marker=marker, zorder=2)

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

# x265
x265_veryslow = RD_Curve()
x265_veryslow.add_points([
    [43.605155, 0.994995, 0.266973],
    [41.947545, 0.993015, 0.183448],
    [39.056156, 0.988178, 0.096584],
    [36.117413, 0.980034, 0.049100],
    [33.236031, 0.964942, 0.024097],
])

# HM
HM = RD_Curve()
HM.add_points([
    [38.394741975952634, None, 0.0850201338836215],
    [37.09568948685745, None, 0.05404833499097868],
    [34.35080985401225, None, 0.026377464411635416],
    [31.757460325217775, None, 0.012964702112369062],
])

# Ours
motionCond_L_tgtIn_PredPrior_ANFIC = RD_Curve()
motionCond_L_tgtIn_PredPrior_ANFIC.add_points([
    [39.3138, None, 0.1111],
    [37.7418, None, 0.0725],
    [35.9543, None, 0.0469],
    [33.6042, None, 0.0273]
])
motionCond_L_tgtIn_PredPrior_ANFIC_ssim = RD_Curve()
motionCond_L_tgtIn_PredPrior_ANFIC_ssim.add_points([
    [None, 0.9951, 0.1483],
    [None, 0.9921, 0.0980],
    [None, 0.9882, 0.0663],
    [None, 0.9831, 0.0459],
    [None, 0.9770, 0.0307],
])

# Ours_Lite
motionCond_tgtIn_PredPrior_ANFIC_Lite = RD_Curve()
motionCond_tgtIn_PredPrior_ANFIC_Lite.add_points([
    [39.3671, None, 0.1215],
    [37.7600, None, 0.0805],
    [36.0183, None, 0.0506],
    [33.7493, None, 0.0310],
])
motionCond_tgtIn_PredPrior_ANFIC_Lite_ssim = RD_Curve()
motionCond_tgtIn_PredPrior_ANFIC_Lite_ssim.add_points([
    [None, 0.9923, 0.1062],
    [None, 0.9885, 0.0686],
    [None, 0.9813, 0.0410],
    [None, 0.9738, 0.0271],
])

# DCVC
DCVC = RD_Curve()
DCVC.add_points([
    [37.952253, None, 0.099996],
    [36.459898, None, 0.065876],
    [34.681865, None, 0.045525],
    [32.963988, None, 0.029638],
])

DCVC_MSSSIM = RD_Curve()
DCVC_MSSSIM.add_points([
    [None, 0.994311, 0.138198],
    [None, 0.991296, 0.094991],
    [None, 0.985097, 0.058756],
    [None, 0.977028, 0.034470],
])

DCVC_ANFIC = RD_Curve()
DCVC_ANFIC.add_points([
    [38.286049, None, 0.104761],
    [36.825698, None, 0.069200],
    [35.027115, None, 0.047287],
    [32.969937, None, 0.030475],
])
DCVC_ANFIC_MSSSIM = RD_Curve()
DCVC_ANFIC_MSSSIM.add_points([
    [None, 0.994122, 0.137291],
    [None, 0.991271, 0.093663],
    [None, 0.984835, 0.057344],
    [None, 0.978660, 0.032327],
])


def main_plot(mode="PSNR"):
    if mode == "PSNR":
        fig_plot = RD_Plot("CLIC", metric="PSNR",
                           anchor=x265_veryslow, xlim=(0.02, 0.13), ylim=(33, 39.5), fontsize=22,
                        # anchor=DCVC_ANFIC, xlim=(0, 0.16), ylim=(33, 39.5), fontsize=22,
#                            anchor=tgtIn_PredPrior_ANFIC, xlim=(0, 0.16), ylim=(31, 39), fontsize=22,
                           interp='quadratic', styles=['-', '--', '-.', ":"],
                           print_style='linear')

        fig_plot.add_curve(x265_veryslow, color='black', label='x265 (veryslow)', marker='o', fillstyle='full')
        fig_plot.add_curve(HM, marker='x', color='#099ac8', label='HM',style='-')
        fig_plot.add_curve(DCVC, 
                           marker='x', color='#f74e0a', label='DCVC',style='-')
        fig_plot.add_curve(DCVC_ANFIC, 
                           marker='H', label='DCVC(ANFIC)', style='-.', color='#349063')
        fig_plot.add_curve(motionCond_tgtIn_PredPrior_ANFIC_Lite,
                           marker='<', label='CANF-VC Lite', style='-', color='#FF75D6')
        fig_plot.add_curve(motionCond_L_tgtIn_PredPrior_ANFIC, 
                           marker='s', label='CANF-VC', style='-', color='#E33A3A')
        fig_plot.save_figure(fig_dir+"RD_CLIC_PSNR_GOP32.png")
        
        
        
    else:
        fig_plot = RD_Plot("CLIC", metric="MS_SSIM",
                           anchor=x265_veryslow, xlim=(0.01, 0.16), ylim=(0.97, 0.997), fontsize=22,
                        #    anchor=DCVC_ANFIC_MSSSIM, xlim=(0.01, 0.17), ylim=(0.955, 0.995), fontsize=22,
                           interp='quadratic', styles=['-', '--', '-.', ":"], print_style='linear')
        
        fig_plot.add_curve(x265_veryslow, color='black', label='x265 (veryslow)', marker='o', fillstyle='full')
        fig_plot.add_curve(DCVC_MSSSIM, 
                           marker='x', color='#f74e0a', label='DCVC',style='-')
        fig_plot.add_curve(DCVC_ANFIC_MSSSIM,
                           marker='H', label='DCVC(ANFIC)', style='-.', color='#349063')
        fig_plot.add_curve(motionCond_tgtIn_PredPrior_ANFIC_Lite_ssim,
                           marker='<', label='CANF-VC Lite', style='-', color='#FF75D6')
        fig_plot.add_curve(motionCond_L_tgtIn_PredPrior_ANFIC_ssim,
                           marker='s', label='CANF-VC', style='-', color='#E33A3A')
        fig_plot.save_figure(fig_dir+"RD_CLIC_MSSSIM_GOP32.png")




""
if __name__ == "__main__":
    import os
    fig_dir = os.getenv("HOME")+"/RD_plot/"
    main_plot("PSNR")
    main_plot("SSIM")
