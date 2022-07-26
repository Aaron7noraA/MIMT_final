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


x265_veryslow = RD_Curve()
x265_veryslow.add_points([
#crf
#     [42.310640, 0.992304, 1.390427],
#     [39.947100, 0.985990, 0.654263],
#     [38.163498, 0.977760, 0.272295],
#     [36.659374, 0.967215, 0.113330],
#     [34.8625, 0.95534, 0.0521],
#     [33.45, 0.94274, 0.0293],
#     [31.85, 0.92287, 0.0159],
#qp
#     [39.741841, 0.983616, 0.492292],
    [38.456771, 0.977060, 0.259318],
#     [36.623059, 0.964421, 0.094087],
#     [34.803380, 0.952785, 0.045210],
#     [32.798723, 0.933458, 0.021915],
    [36.125, 0.9645, 0.0940],
    [34.8625, 0.95534, 0.0521],
    [33.45, 0.94274, 0.0293],
    [31.85, 0.92287, 0.0159],
])
x265_veryslow_TCM = RD_Curve()
x265_veryslow_TCM.add_points([
    [36.125, 0.9645, 0.0940],
    [34.8625, 0.95534, 0.0521],
    [33.45, 0.94274, 0.0293],
    [31.85, 0.92287, 0.0159],
])
HM_LDP = RD_Curve()
HM_LDP.add_points([ 
    [39.0983, 0.97970, 0.29714], # 17
    [37.4141, 0.96870, 0.10312], # 22
    [36.6565, 0.96459, 0.06867], # 24
    [35.5011, 0.95749, 0.04034], # 27
    [33.6242, 0.94145, 0.01794], # 32
    # [31.7240, None, 0.00817], # 37
])



# +

# Ours
motionCond_L_tgtIn_PredPrior_ANFIC = RD_Curve()
motionCond_L_tgtIn_PredPrior_ANFIC.add_points([
    [37.5971, None, 0.1014],
    [36.4293, None, 0.0600],
    [35.1831, None, 0.0373],
    [33.5179, None, 0.0216],
])
motionCond_L_tgtIn_PredPrior_ANFIC_RNN = RD_Curve()
motionCond_L_tgtIn_PredPrior_ANFIC_RNN.add_points([
    [37.9511, None, 0.1182],
    [36.8064, None, 0.0694],
    [35.6110, None, 0.0423],
    [33.9548, None, 0.0256],
])
motionCond_L_tgtIn_PredPrior_ANFIC_ssim_GOP32 = RD_Curve()
motionCond_L_tgtIn_PredPrior_ANFIC_ssim_GOP32.add_points([
    [None, 0.9814, 0.1903],
    [None, 0.9764, 0.1208],
    [None, 0.9705, 0.0799],
    [None, 0.9629, 0.0511],
])
# Ours*
motionCond_L_tgtIn_PredPrior = RD_Curve()
motionCond_L_tgtIn_PredPrior.add_points([
    [37.5762, None, 0.1155],
    [36.2780, None, 0.0634],
    [34.7552, None, 0.0374],
    [32.9538, None, 0.0216],
])
motionCond_L_tgtIn_PredPrior_ssim_GOP32 = RD_Curve()
motionCond_L_tgtIn_PredPrior_ssim_GOP32.add_points([
    [None, 0.9842, 0.2340],
    [None, 0.9769, 0.1246],
    [None, 0.9696, 0.0776],
    [None, 0.9575, 0.0446],
])
# Ours-
motionANF_tgtIn_PredPrior_ANFIC_GOP32 = RD_Curve()
motionANF_tgtIn_PredPrior_ANFIC_GOP32.add_points([
    [37.5655, None, 0.1074],
    [36.5041, None, 0.0656],
    [35.2373, None, 0.0396],
    [33.7977, None, 0.0303],
])
motionANF_tgtIn_PredPrior_ANFIC_ssim_GOP32 = RD_Curve()
motionANF_tgtIn_PredPrior_ANFIC_ssim_GOP32.add_points([
    [None, 0.9812, 0.1908],
    [None, 0.9756, 0.1163],
    [None, 0.9667, 0.0675],
    [None, 0.9558, 0.0363],
])
# Ours_Lite
motionCond_tgtIn_PredPrior_ANFIC_Lite = RD_Curve()
motionCond_tgtIn_PredPrior_ANFIC_Lite.add_points([
    [37.6426, None, 0.1075],
    [36.3618, None, 0.0642],
    [35.2693, None, 0.0384],
    [33.6328, None, 0.0226],
#     [35.1831, None, 0.0373],
#     [33.5179, None, 0.0216],
])
motionCond_light_tgtIn_PredPrior_ANFIC_ssim_GOP32 = RD_Curve()
motionCond_light_tgtIn_PredPrior_ANFIC_ssim_GOP32.add_points([
    [None, 0.9821, 0.2105],
    [None, 0.9760, 0.1217],
    [None, 0.9671, 0.0669],
    [None, 0.9578, 0.0409],
])
# Ours_Lite*
motionCond_tgtIn_PredPrior_Lite = RD_Curve()
motionCond_tgtIn_PredPrior_Lite.add_points([
    [37.6061, None, 0.1244],
    [36.4065, None, 0.0696],
    [34.9353, None, 0.0406],
    [33.2755, None, 0.0246],
])
# MLVC
MLVC_ANFIC = RD_Curve()
MLVC_ANFIC.add_points([
    [38.2213, 0.9768, 0.1606],
    [37.2382, 0.9705, 0.1134],
    [36.1996, 0.9631, 0.0759],
    [35.1658, 0.9570, 0.0599],
])
# MLVC*
MLVC = RD_Curve()
MLVC.add_points([
    [38.3520, 0.9770, 0.1828],
    [37.4877, 0.9711, 0.1288],
    [36.5483, 0.9640, 0.0886],
    [35.7963, 0.9603, 0.0688],
])
# DVC
Ours_DVC_ANFIC = RD_Curve()
Ours_DVC_ANFIC.add_points([
    [36.8124, 0.9679, 0.1195],
    [35.3134, 0.9595, 0.0753],
    [33.9115, 0.9481, 0.0450],
    [32.3413, 0.9337, 0.0260],
])
# DVC*
Ours_DVC = RD_Curve()
Ours_DVC.add_points([
    [36.9331, 0.9690, 0.1344],
    [35.4874, 0.9602, 0.0786],
    [34.0200, 0.9485, 0.0456],
    [32.4618, 0.9311, 0.0264],
])

# +
# DCVC#
DCVC = RD_Curve()
DCVC.add_points([
    [36.934245, None, 0.100493],
    [35.774857, None, 0.061831],
    [34.550156, None, 0.039548],
    [33.308935, None, 0.026136]
])
DCVC_MSSSIM = RD_Curve()
DCVC_MSSSIM.add_points([
    [None, 0.987043, 0.326329],
    [None, 0.981604, 0.209080],
    [None, 0.973659, 0.113407],
    [None, 0.963319, 0.060842]

])
# DCVC
DCVC_ANFIC = RD_Curve()
DCVC_ANFIC.add_points([
    [37.162171, None, 0.107141],
    [35.968660, None, 0.065564],
    [34.761939, None, 0.041037],
    [33.262352, None, 0.026856],
])
DCVC_ANFIC_MSSSIM = RD_Curve()
DCVC_ANFIC_MSSSIM.add_points([
    [None, 0.986878, 0.321114],
    [None, 0.981774, 0.207662],
    [None, 0.973763, 0.111851],
    [None, 0.964900, 0.059977]

])
# DCVC*
DCVC_BPG = RD_Curve()
DCVC_BPG.add_points([
    [37.247326, None, 0.122876],
    [36.100168, None, 0.069483],
    [34.751588, None, 0.042116],
    [33.300669, None, 0.027137],
])
DCVC_BPG_MSSSIM = RD_Curve()
DCVC_BPG_MSSSIM.add_points([
    [None, 0.987326, 0.340905],
    [None, 0.981939, 0.211298],
    [None, 0.973711, 0.112411],
    [None, 0.962698, 0.059613]

])
# -


condInter_PredPrior = RD_Curve()
condInter_PredPrior.add_points([
    [37.9511, None, 0.1182],
    [36.8064, None, 0.0694],
    [35.6110, None, 0.0423],
    [33.9548, None, 0.0256],
])

condInter_ResBlock_ANFIC = RD_Curve()
condInter_ResBlock_ANFIC.add_points([
    [38.7391, None, 0.1692],
    [37.1272, None, 0.1085],
    [36.1344, None, 0.0695],
    [34.7898, None, 0.0513],
])

tgtIn_PredPrior_ANFIC = RD_Curve()
tgtIn_PredPrior_ANFIC.add_points([
    [37.6638, None, 0.1088],
    [36.3215, None, 0.0675],
    [34.9232, None, 0.0427],
    [33.2301, None, 0.0265],
])

# +
tgtIn_PredPrior_ANFIC_Lite = RD_Curve()
tgtIn_PredPrior_ANFIC_Lite.add_points([ 
    [37.6525, None, 0.1372],
    [36.5681, None, 0.0918],
    [35.4141, None, 0.0643],
    [34.1985, None, 0.0471],
])


tgtIn_PredPrior_ANFIC_chooseRef = RD_Curve()
tgtIn_PredPrior_ANFIC_chooseRef.add_points([
    [37.7522, None, 0.1078],
    [36.5383, None, 0.0663],
    [35.2542, None, 0.0421],
    [33.5167, None, 0.0261],
])

tgtIn_PredPrior_ANFIC_HoneyBee = RD_Curve()
tgtIn_PredPrior_ANFIC_HoneyBee.add_points([
    [37.6262, None, 0.0369],
    [36.0737, None, 0.0268],
    [34.7825, None, 0.0175],
    [33.0625, None, 0.0123],
])

tgtIn_PredPrior_ANFIC_chooseRef_HoneyBee = RD_Curve()
tgtIn_PredPrior_ANFIC_chooseRef_HoneyBee.add_points([
    [38.0540, None, 0.0321],
    [37.2217, None, 0.0205],
    [36.3951, None, 0.0147],
    [34.6405, None, 0.0099],
])


condMo_condInter_3FramesLinear_ANFIC = RD_Curve()
condMo_condInter_3FramesLinear_ANFIC.add_points([
    [37.8087, None, 0.1173],
    [36.6379, None, 0.0709],
    [35.4047, None, 0.0431],
    [33.9131, None, 0.0268],
])

condMo_condInter_3FramesFused_ANFIC = RD_Curve()
condMo_condInter_3FramesFused_ANFIC.add_points([
    [38.0033, None, 0.1167],
    [36.8363, None, 0.0698],
    [35.5135, None, 0.0409],
    [34.0351, None, 0.0251],
])
condMo_condInter_3FramesFused_ANFIC_fakeRNN = RD_Curve()
condMo_condInter_3FramesFused_ANFIC_fakeRNN.add_points([
    [37.5852, None, 0.0982],
    [36.3419, None, 0.0584],
    [34.9299, None, 0.0361],
    [33.2398, None, 0.0212],
])
condMo_condInter_3FramesFused_ANFIC2 = RD_Curve()
condMo_condInter_3FramesFused_ANFIC2.add_points([
    [38.1021, None, 0.1209],
#     [37.5465, None, 0.1058],
#     [36.1936, None, 0.0627],
#     [34.8350, None, 0.0388],
#     [33.1210, None, 0.0242],
])
condMo_condInter_3FramesFused_ANFIC_fakeRNN2 = RD_Curve()
condMo_condInter_3FramesFused_ANFIC_fakeRNN2.add_points([
    [37.5915, None, 0.1030],
#     [37.4831, None, 0.1108],
#     [36.2049, None, 0.0656],
#     [34.9133, None, 0.0413],
#     [33.2598, None, 0.0248],
])


condInter_condLSTM_ANFIC_Lite_fakeRNN = RD_Curve()
condInter_condLSTM_ANFIC_Lite_fakeRNN.add_points([ 
#     [37.6729, None, 0.1161],
    [37.6800, None, 0.1142],
    [36.4410, None, 0.0719],
#     [35.1364, None, 0.0447],
#     [33.5051, None, 0.0279],
    [35.1460, None, 0.0437],
    [33.5283, None, 0.0268],
])
condInter_condLSTM_ANFIC_Lite = RD_Curve()
condInter_condLSTM_ANFIC_Lite.add_points([ 
#     [38.3980, None, 0.1509],
    [38.3616, None, 0.1425],
#     [37.3363, None, 0.0916],
    [37.1152, None, 0.0877],
    [36.1724, None, 0.0599],
#     [35.8757, None, 0.0545],
    [34.6532, None, 0.0389],
#     [34.0359, None, 0.0331],
])

CANFVC_ANFIC_Lite_firstP = RD_Curve()
CANFVC_ANFIC_Lite_firstP.add_points([ 
    [37.6381, None, 0.1171],
    [36.3639, None, 0.0726],
    [35.2208, None, 0.0456],
    [33.6329, None, 0.0275],
])


# +

def main_plot(mode="PSNR"):
    if mode == "PSNR":
        fig_plot = RD_Plot("UVG", metric="PSNR",
                           anchor=x265_veryslow_TCM, xlim=(0, 0.16), ylim=(33, 38.5), fontsize=22,
#                            anchor=tgtIn_PredPrior_ANFIC, xlim=(0, 0.16), ylim=(31, 39), fontsize=22,
#                            anchor=tgtIn_PredPrior_ANFIC_HoneyBee, xlim=(0, 0.04), ylim=(31, 39), fontsize=22,
                           interp='quadratic', styles=['-', '--', '-.', ":"],
                           print_style='linear')
#         fig_plot.add_curve(x265_veryslow, color='black', label='x265 (veryslow)', marker='o', fillstyle='full')
        fig_plot.add_curve(x265_veryslow_TCM, color='black', label='x265 (veryslow)', marker='o', fillstyle='full')
        fig_plot.add_curve(HM_LDP, color='gray', label='HM (LDP)', marker='o', fillstyle='full')

#         fig_plot.add_curve(Ours_DVC,
#                            marker='x', color='#f7342c', label='DVC_Pro*',style='-')
#         fig_plot.add_curve(Ours_DVC_ANFIC,
#                            marker='x', color='#939b93', label='DVC_Pro',style='-')
#         fig_plot.add_curve(MLVC,
#                            marker='x', color='#f70ade', label='MLVC (CVPR\'20)',style='-')
        fig_plot.add_curve(DCVC, marker='h', color='#f98b24', label='DCVC (NIPS\'21)', style=(0,(3,1,1,1)))
#         fig_plot.add_curve(DCVC_BPG,
#                            marker='x', color='#13dc2e', label='DCVC*',style='-')
#         fig_plot.add_curve(motionCond_tgtIn_PredPrior_Lite,
#                            marker='x', color='#0a92f7', label='Ours_Lite*',style='-')
#         fig_plot.add_curve(MLVC_ANFIC,
#                            marker='v', label='M-LVC', style='--', color='#A6450C')
#         fig_plot.add_curve(DCVC_ANFIC, 
#                            marker='H', label='DCVC', style='-.', color='#349063')
#         fig_plot.add_curve(motionCond_tgtIn_PredPrior_ANFIC_Lite,
#                            marker='<', label='CANF-VC Lite', style='-', color='#FF75D6')
#         fig_plot.add_curve(motionANF_tgtIn_PredPrior_ANFIC_GOP32,
#                            marker='>', label='CANF-VC$^-$', style='-', color='#4DA2E7')
#         fig_plot.add_curve(motionCond_L_tgtIn_PredPrior,
#                            marker='x', color='#8e44ad', label='CANFVC*',style='-')


        fig_plot.add_curve(motionCond_L_tgtIn_PredPrior_ANFIC, 
                           marker='s', label='CANF-VC', style='-', color='#E33A3A')
#         fig_plot.add_curve(tgtIn_PredPrior_ANFIC_Lite,
#                            marker='v', label='Baseline: condInter_Lite', style='--', color='#A6450C')
        fig_plot.add_curve(CANFVC_ANFIC_Lite_firstP,
                           marker='v', label='CANF-VC Lite$^-$', style='--', color='#A6450C')
        fig_plot.add_curve(condInter_condLSTM_ANFIC_Lite_fakeRNN,
                           marker='>', color='#0a92f7', label='proposed',style='-')
        fig_plot.add_curve(condInter_condLSTM_ANFIC_Lite,
                           marker='<', color='#2ca02c', label='proposed (RNN-based training)',style='-')
        
        
#         fig_plot.add_curve(motionCond_L_tgtIn_PredPrior_ANFIC_RNN, marker='D', color='#E33A3A', label='CANFVC (RNN)',style='-.')
#         fig_plot.add_curve(condMo_condInter_3FramesLinear_ANFIC, marker='o', label='multi-ref (linear)', style='--', color='#FF75D6')
#         fig_plot.add_curve(condMo_condInter_3FramesFused_ANFIC, marker='X', label='multi-ref (fusion)', style='-.', color='#4DA2E7')



#         fig_plot.add_curve(motionCond_L_tgtIn_PredPrior_ANFIC, 
#                            marker='H', label='single', style='-', color='#4DA2E7')
#         fig_plot.add_curve(motionCond_L_tgtIn_PredPrior_ANFIC_RNN,
#                            marker='<', label='single ; RNN', style='-', color='#E33A3A')
#         fig_plot.add_curve(condMo_condInter_3FramesFused_ANFIC_fakeRNN,
#                            marker='>', label='multi', style='-', color='#4DA2E7')
#         fig_plot.add_curve(condMo_condInter_3FramesFused_ANFIC, 
#                            marker='s', label='multi ; RNN', style='-', color='#E33A3A')
#         fig_plot.add_curve(condMo_condInter_3FramesFused_ANFIC_fakeRNN2, 
#                            marker='H', label='new', style='-.', color='#349063')
#         fig_plot.add_curve(condMo_condInter_3FramesFused_ANFIC2,
#                            marker='<', label='new ; RNN', style='-', color='#FF75D6')
        fig_plot.save_figure(fig_dir+"RD_UVG_PSNR_GOP32.png")
        
        
        
    else:
        fig_plot = RD_Plot("UVG", metric="MS_SSIM",
                           anchor=x265_veryslow, xlim=(0.01, 0.25), ylim=(0.955, 0.985), fontsize=22,
                           interp='quadratic', styles=['-', '--', '-.', ":"], print_style='linear')
        
        fig_plot.add_curve(x265_veryslow, color='black', label='x265 (veryslow)', marker='o', fillstyle='full')
#         fig_plot.add_curve(x265_veryslow_TCM, color='black', label='TCM: x265 (veryslow)', marker='o', fillstyle='full')

        fig_plot.add_curve(HM_LDP, color='gray', label='HM (LDP)', marker='o', fillstyle='full')


#         fig_plot.add_curve(MLVC,
#                            marker='x', color='#f70ade', label='MLVC*',style='-')
#         fig_plot.add_curve(DCVC_MSSSIM,
#                            marker='x', color='#f74e0a', label='DCVC#',style='-')
#         fig_plot.add_curve(DCVC_BPG_MSSSIM,
#                            marker='x', color='#13dc2e', label='DCVC*',style='-')
        fig_plot.add_curve(MLVC_ANFIC,
                           marker='v', label='M-LVC', style='--', color='#A6450C')
        fig_plot.add_curve(DCVC_ANFIC_MSSSIM,
                           marker='H', label='DCVC', style='-.', color='#349063')
        fig_plot.add_curve(motionCond_light_tgtIn_PredPrior_ANFIC_ssim_GOP32,
                           marker='<', label='CANF-VC Lite', style='-', color='#FF75D6')
        fig_plot.add_curve(motionANF_tgtIn_PredPrior_ANFIC_ssim_GOP32,
                           marker='>', label='CANF-VC$^-$', style='-', color='#4DA2E7')
        fig_plot.add_curve(motionCond_L_tgtIn_PredPrior_ANFIC_ssim_GOP32,
                           marker='s', label='CANF-VC', style='-', color='#E33A3A')
#         fig_plot.add_curve(motionCond_L_tgtIn_PredPrior_ssim_GOP32,
#                            marker='x', color='#8e44ad', label='Ours*',style='-')
    
    
    # fig_plot.plot() 
        fig_plot.save_figure(fig_dir+"RD_UVG_MSSSIM_GOP32.png")
# -



""
if __name__ == "__main__":
    import os
    fig_dir = os.getenv("HOME")+"/RD_plot/"
    main_plot("PSNR")
    main_plot("SSIM")
