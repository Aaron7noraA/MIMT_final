import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import record
import pandas as pd
import scipy.interpolate
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import os.path as osp
import json

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
        assert metric in ['PSNR', 'MS-SSIM', 'MS-SSIM-dB']
        self.metric = metric.replace('-', '_')
        
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

        plt.title('{} Dataset'.format(name[:5]))
        plt.xlabel("Bit-rate (bpp)")
        y_label = metric.replace("_dB", "")+"-RGB" + (" (dB)" if metric in ["PSNR", "MS_SSIM_dB"] else "")
        y_label = y_label.replace("MS_SSIM", "MS-SSIM")
        plt.ylabel(y_label)
        plt.grid()


    def add_curve(self, curve: RD_Curve, label='str',
                  color=None, style=None, width=2, marker=None, 
                  fillstyle='none', markersize=20, piecewise=0):
#                   color=None, style=None, width=2, marker=None):
        curve.sorted()
        self.curves.append(curve)
        bpp = curve.bpp
        value = getattr(curve, self.metric)
        assert self.anchor is not None
        if self.metric == 'PSNR':
            lb = 29.5
        elif self.metric == 'MS_SSIM':
            # lb = 0.964900
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
                        marker=marker, s=100, zorder=2)

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

def get_metric_from_csv(filename, metric='PSNR'):
    df = pd.read_csv(filename, sep=',', header=0)
    assert metric in df.columns, "Invalid metric"
    return np.mean(df[metric].fillna(df['PSNR']).values.tolist())

def get_metric_from_json(filename, seq_name, QPs, mode):
    assert mode in ['PSNR', 'MS-SSIM']
    with open(filename, 'r') as json_file:
        record = json.load(json_file)

    result = []
    for qp in QPs:
        if mode == 'PSNR':
            result.append([record[seq_name][str(qp)]['PSNR'], None, record[seq_name][str(qp)]['BPP']])
        elif mode == 'MS-SSIM':
            result.append([None, record[seq_name][str(qp)]['MS-SSIM'], record[seq_name][str(qp)]['BPP']])
        
    return result

def DCVC_get_metric_from_json(filename, seq_name, dataset, metric='PSNR'):
    assert metric == 'PSNR' or metric == 'MS-SSIM', f'I haven\'t implement metric {metric}.'
    with open(filename, 'r') as json_file:
        record = json.load(json_file)
    
    result = []
    if metric == 'PSNR':
        for i in range(4):
            result.append([record[dataset][seq_name][f'model_dcvc_quality_{i}_psnr.pth']['ave_all_frame_quality'], None, record[dataset][seq_name][f'model_dcvc_quality_{i}_psnr.pth']['ave_all_frame_bpp']])
    else:
        for i in range(4):
            result.append([None, record[dataset][seq_name][f'model_dcvc_quality_{i}_msssim.pth']['ave_all_frame_quality'], record[dataset][seq_name][f'model_dcvc_quality_{i}_msssim.pth']['ave_all_frame_bpp']])

    return result

def main_plot(models, anchor='x265 (veryslow)', mode="PSNR"):
    assert mode == 'PSNR' or mode == 'MS-SSIM'
    if mode == "PSNR":
        ANCHOR = models[anchor]
        for seq in seq_name:
            fig_plot = RD_Plot(seq, metric="PSNR",
                               anchor=ANCHOR['seq_rd_PSNR'][seq], fontsize=22,
                               interp='quadratic', styles=['-', '--', '-.', ":"], print_style='linear', figsize=(12, 8))
            for model in models.keys():
                fig_plot.add_curve(
                    models[model]['seq_rd_PSNR'][seq], 
                    label=model,
                    marker=models[model]['marker'] if 'marker' in models[model].keys() else None,
                    style=models[model]['style'] if 'style' in models[model].keys() else None,
                    color=models[model]['color'] if 'color' in models[model].keys() else None,
                    width=models[model]['width'] if 'width' in models[model].keys() else 2,
                )
            fig_plot.save_figure(fig_dir + '/PSNR/' + "RD_" + seq[:5] + "_PSNR.png")
    else:
        
        ANCHOR = models[anchor]
        for seq in seq_name:
            fig_plot = RD_Plot(seq, metric="MS-SSIM",
                               anchor=ANCHOR['seq_rd_MS-SSIM'][seq], fontsize=22,
                               interp='quadratic', styles=['-', '--', '-.', ":"], print_style='linear', figsize=(12, 8))
            for model in models.keys():
                fig_plot.add_curve(
                    models[model]['seq_rd_MS-SSIM'][seq], 
                    label=model,
                    marker=models[model]['marker'] if 'marker' in models[model].keys() else None,
                    style=models[model]['style'] if 'style' in models[model].keys() else None,
                    color=models[model]['color'] if 'color' in models[model].keys() else None,
                    width=models[model]['width'] if 'width' in models[model].keys() else 2,
                )

            fig_plot.save_figure(fig_dir + '/MS-SSIM/' + "RD_" + seq[:5] + "_MS-SSIM.png")
    # fig_plot.plot() 



""
if __name__ == "__main__":
    import os
    fig_dir = os.getenv("HOME")+"/RD_plot/"
    os.makedirs(f'{fig_dir}/PSNR', exist_ok=True)
    os.makedirs(f'{fig_dir}/MS-SSIM', exist_ok=True)
    
    # ------------------------------------------- #    
    seq_name = [ 'a06845dd7d1d808e4f4743b7f08f2bf75a9a72264d4fb16505caf6e334611003',
                 '57cb54c2cde2789359ecf11b9b9b8207c6a79b7aa27f15a69d7e9a1c2caad912',
                 'fae057c83b04868424da3bb7139e29b3f328d5a93aaa9e617e825b93422d92c5',
                 'af31d741db80475c531bb7182ad0536df9dc88a6876fa38386dd5db850d86051',
                 'd0a99fb6b64e60d7754265586481ec43968e8fd97e7e4437332bb182d7548cb3',
                 '97d6ac9d81b64bf909bf4898072bb20492522ae182918e763a86b56745890add',
                 'd73059fe0ed42169f7e98ff7401d00479a7969753eb80af9846176a42543ccb0',
                 '23e266612abe7b8767587d6e77a5eb3c6b8a71c6bf4c4ff2b1c11cc478cc7244',
                 '9a6379abea3fc820ca60afb9a60092d41b3a772ff348cfec92c062f6187f85e2',
                 '7c7d58e4f82772f627d5cbe3df3b08573d5bd7a58639387b865449d5a550bbda',
                 '29aabdd9d3065802c21e2d828561c205d563e79d39d1e10a18f961b5b5bf0cad',
                 '7b0eaacc48c9b5ea0edf5dcf352d913fd0cf3f79ae149e94ada89ba1e772e711',
                 '0442d8bdf9902226bfb38fbe039840d4f8ebe5270eda39d7dba56c2c3ae5becc',
                 'b7ee0264612a6ca6bf2bfa03df68acf4af9bb5cac34f7ad43fe30fa4b7bc4824',
                 '8db183688ce3e59461355e2c7cc97b3aee9f514a2e28260ead5a3ccf2000b079',
                 '8cbafab285e74614f10d3a8bf9ee94434eacae6332f5f10fe1e50bfe5de9ec33',
                 '318c694f5c83b78367da7e6584a95872510db8544f815120a86923aff00f5ff9',
                 '04ca8d2ac3af26ad4c5b14cf214e0d7c317c953e804810829d41645fdce1ad88',
                 '1e3224380c76fb4cad0a8d3a7c74a8d5bf0688d13df15f23acd2512de4374cb4',
                 '04a1274a93ec6a36ad2c1cb5eb83c3bdf2cf05bbe01c70a8ca846a7f9fa4b550',
                 '0d49152a92ce3b843968bf2e131ea5bc5e409ab056196e8c373f9bd2d31b303d',
                 '5d8f03cf5c6a469004a0ca73948ad64fa6d222b3b807f155a66684387f5d208a',
                 '0e1474478f33373566b4fbd6b357cf6b65015a6f4aa646754e065bf4a1b43c15',
                 '0659b03fb82cae130fef6a931755bbaae6e7bd88f58873df1ae98d2145dba9ce',
                 'a89f641b8dd2192f6f8b0ae75e3a24388b96023b21c63ff67bb359628f5df6de',
                 '209921b14cef20d62002e2b0c21ad692226135b52fee7eead315039ca51c470c',
                 '917d1b33f0e20d2d81471c3a0ff7adbef9e1fb7ee184b604880b280161ffdd56',
                 '9ce4af9a3b304b4b5387f27bca137ce1f0f35c12837c753fc17ea9bb49eb8ec5',
                 '393608bbbf2ac4d141ce6a3616a2364a2071539acb1969032012348c5817ef3c',
                 '9299df423938da4fd7f51736070420d2bb39d33972729b46a16180d07262df12']

    # seq_name = ['Beauty', 'Bosphorus', 'HoneyBee', 'Jockey', 'ReadySteadyGo', 'ShakeNDry', 'YachtRide']

    # ------------------------------------------- #
    models = {}
    models['x265 (veryslow)'] = \
    {
        'path': "/home/u1481110/compress_265_gop32/report/x265_gop32_record.json",
        'metrics': ['PSNR', 'MS-SSIM'],
        'QPs': [19, 22, 27, 32, 37],
        'marker': 'o',
        'style': '-',
        'color': 'black'
    }

    models['HM (LDP)'] = \
    {
        'path': "/home/u1481110/LDP/report/HM_gop32_record.json",
        'metrics': ['PSNR'],
        'QPs': [24, 27, 32, 37],
        'marker': 'o',
        'color': 'gray'
    }

    models['CANFVC'] = \
    {
        'PSNR path': osp.join('/work/u1481110/models/torchDVC/ANF-based-resCoder-for-DVC/',
                         'CLIC_Test_CANFVC-GOP32'),
        'MS-SSIM path': osp.join('/work/u1481110/models/torchDVC/CANFVC_Plus/',
                         'CLIC_Test_CANFVC-GOP32-SSIM'),
        'metrics': ['PSNR', 'MS-SSIM'],
        'PSNR lambdas': [256, 512, 1024, 2048],
        'MS-SSIM lambdas': [256, 512, 1024, 2048, 4096],
        'marker': 's',
        'style': '-',
        'color': '#E33A3A'
    }

    models['CANFVC_Lite'] = \
    {
        'PSNR path': osp.join('/work/u1481110/models/torchDVC/ANF-based-resCoder-for-DVC/',
                         'CLIC_Test_CANFVC-Lite-GOP32'),
        'MS-SSIM path': osp.join('/work/u1481110/models/torchDVC/CANFVC_Plus/',
                         'CLIC_Test_CANFVC-Lite-GOP32-SSIM'),
        'metrics': ['PSNR', 'MS-SSIM'],
        'PSNR lambdas': [256, 512, 1024, 2048],
        'MS-SSIM lambdas': [256, 512, 1024, 2048],
        'marker': '+',
        'style': '-.',
        'color': '#4DA2E7'
    }

    models['DCVC(ANFIC)'] = \
    {
        'PSNR path': "/home/u1481110/DCVC/DCVC_result_psnr_ANFIC_CLIC_gop32.json",
        'MS-SSIM path': "/home/u1481110/DCVC/DCVC_result_msssim_ANFIC_CLIC_gop32.json",
        'metrics': ['PSNR', 'MS-SSIM'],
        'marker': 'H',
        'style': '-.',
        'color': '#349063',
    }

    # ------------------------------------------- #
    for model in models.keys():
        for metric in models[model]['metrics']:
            assert metric == 'PSNR' or metric == 'MS-SSIM'
            if metric == 'PSNR':
                models[model]['seq_rd_PSNR'] = {}
            else:
                models[model]['seq_rd_MS-SSIM'] = {}

            for seq in seq_name:
                if model == 'x265 (veryslow)' or model == 'HM (LDP)': 
                    models[model][f'seq_rd_{metric}'][seq] = RD_Curve()
                    models[model][f'seq_rd_{metric}'][seq].add_points(
                        get_metric_from_json(models[model]['path'], seq, models[model]['QPs'], metric)
                        )
                elif model == 'DCVC(ANFIC)':
                    models[model][f'seq_rd_{metric}'][seq] = RD_Curve()
                    if metric == 'PSNR':
                        models[model]['seq_rd_PSNR'][seq].add_points(
                            DCVC_get_metric_from_json(models[model]['PSNR path'], seq, 'CLIC_2022', 'PSNR')
                            )
                    else:
                        models[model]['seq_rd_MS-SSIM'][seq].add_points(
                            DCVC_get_metric_from_json(models[model]['MS-SSIM path'], seq, 'CLIC_2022', 'MS-SSIM')
                            )
                else:
                    models[model][f'seq_rd_{metric}'][seq] = RD_Curve()
                    for lmda in models[model][f'{metric} lambdas']:
                        filename = osp.join(models[model][f'{metric} path'] + '-' + str(lmda), 'report', seq+'.csv')
                        models[model][f'seq_rd_{metric}'][seq].add_points([
                            get_metric_from_csv(filename, 'PSNR') if metric == 'PSNR' else None,
                            # Because the report only has PSNR column (but it is MS-SSIM column)
                            get_metric_from_csv(filename, 'PSNR') if metric == 'MS-SSIM' else None,
                            get_metric_from_csv(filename, 'Rate')
                        ])
                
# ----------------------------------------------------------------- #

    main_plot(models, anchor='x265 (veryslow)', mode="PSNR")
    # main_plot(models, anchor='x265 (veryslow)', mode="MS-SSIM")
