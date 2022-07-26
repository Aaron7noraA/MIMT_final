#! /Users/gcwhiteshadow/anaconda3/bin/python3
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


class RD_Curve:
    def __init__(self, index=['PSNR', 'MS-SSIM', 'bpp']):
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

    def get_series(self, index_name: str):
        assert any(index_name in s for s in self.index)
        dict_name = {self.index[i]: i for i in range(0, len(self.index))}

        return self.points[:, dict_name[index_name]]

    @property
    def PSNR(self):
        return self.get_series('PSNR')

    @property
    def MS_SSIM(self):
        return self.get_series('MS-SSIM')

    @property
    def bpp(self):
        return self.get_series('bpp')


class RD_Plot:
    def __init__(self, metric='PSNR'):
        assert metric == 'PSNR' or metric == 'MS-SSIM'
        plt.figure(figsize=(16, 12))

        self.curves = []
        self.ax = plt.subplot(111)

        plt.title('{} on Kodak'.format(metric))
        plt.xlabel("bit-per-pixel")
        plt.ylabel(metric)

        xmajor = MultipleLocator(0.5)
        ymajor = MultipleLocator(5)
        # xmajor = MultipleLocator(0.5)
        # ymajor = MultipleLocator(0.01)

        self.ax.xaxis.set_major_locator(xmajor)
        self.ax.yaxis.set_major_locator(ymajor)

        xminor = MultipleLocator(0.1)
        yminor = MultipleLocator(1)
        # xminor = MultipleLocator(0.1)
        # yminor = MultipleLocator(0.005)

        self.ax.xaxis.set_minor_locator(xminor)
        self.ax.yaxis.set_minor_locator(yminor)

        plt.grid(b=True, which='major', color='black')
        plt.grid(b=True, which='minor', color='gray', linestyle='--')

    @staticmethod
    def add_curve(series_A: np.ndarray, series_B: np.ndarray, label='str',
                  color=None, style=None, width=None, marker=None):
        plt.plot(series_A, series_B, label=label, c=color, ls=style, lw=width, marker=marker, fillstyle='none',
                 markersize=10)
        plt.legend(loc='lower right')

    def plot(self):
        plt.show()

    def save_figure(self, filename: str):
        plt.savefig(filename)


BPG = RD_Curve()
BPG.add_points([
    # [23.787652, 0.783939, 0.023857],
    # [24.169736, 0.800621, 0.02854],
    # [24.583862, 0.815880, 0.034282],
    # [25.005823, 0.830978, 0.041183],
    # [25.441511, 0.845919, 0.049301],
    # [25.892665, 0.859465, 0.058861],
    [26.341951, 0.871920, 0.070444],
    [26.815071, 0.884832, 0.084175],
    [27.298468, 0.896600, 0.100255],
    [27.803390, 0.908020, 0.119211],
    [28.299163, 0.917528, 0.140076],
    [28.836366, 0.926932, 0.165529],
    [29.370955, 0.935368, 0.193939],
    [29.927924, 0.943184, 0.226882],
    [30.519758, 0.950550, 0.264902],
    [31.114314, 0.956814, 0.308004],
    [31.707365, 0.962287, 0.353821],
    [32.333445, 0.967124, 0.406663],
    [32.953625, 0.971309, 0.465174],
    [33.589638, 0.975060, 0.528513],
    [34.260077, 0.978421, 0.602615],
    [34.921675, 0.981216, 0.681227],
    [35.560822, 0.983577, 0.76315],
    [36.239184, 0.985761, 0.855122],
    [36.892823, 0.987516, 0.954195],
    [37.539114, 0.989023, 1.058729],
    [38.225035, 0.990455, 1.178031],
    [38.892220, 0.991701, 1.302895],
    [39.508856, 0.992658, 1.427853],
    # [40.150529, 0.993543, 1.570484],
    # [40.780304, 0.994321, 1.72431],
    # [41.391531, 0.994994, 1.890798],
    # [42.054650, 0.995657, 2.08568],
    # [42.720115, 0.996264, 2.296926],
    # [43.337300, 0.996777, 2.513738],
    # [43.993691, 0.997260, 2.762745],
    # [44.653016, 0.997676, 3.021654],
    # [45.301270, 0.998029, 3.311613],
    # [45.915667, 0.998321, 3.616902],
    # [46.518668, 0.998584, 3.94321],
])

Google_inference = RD_Curve()
Google_inference.add_points([
    [40.8280, None, 1.7386],
    [36.8613, None, 0.9648],
    [32.7323, None, 0.4814],
    [29.0628, None, 0.2064],
    [27.2412, None, 0.1275]
])

Google_reRun = RD_Curve()
Google_reRun.add_points([
    [38.7326, 0.9929, 1.8113],  # 0122_1019
    [36.6219, 0.9893, 1.2611],  # 0122_1020
    [36.0186, 0.9893, 1.1364],  # 0121_0324
    [31.9183, 0.9700, 0.4729],  # 0121_0023
    [29.8161, 0.9515, 0.2972],  # 0121_0325
    [25.9059, 0.8856, 0.0902]   # 0121_0024
])

Google_inference = RD_Curve()
Google_inference.add_points([
    [40.8280, None, 1.7386],
    [36.8613, None, 0.9648],
    [32.7323, None, 0.4814],
    [29.0628, None, 0.2064],
    [27.2412, None, 0.1275]
])

# 2FM lambda=30, 100, 300, 1000
FM2_wPCA_wRE = RD_Curve()
FM2_wPCA_wRE.add_points([
    [22.8404, 0.7885, 0.03349],
    [24.9357, 0.8249, 0.05771],
    [26.1662, 0.9051, 0.08631],
    [26.4656, None, 0.10367]
])

# 4FM lambda=30, 100, 200, 300, 3000
FM4_wPCA_wRE = RD_Curve()
FM4_wPCA_wRE.add_points([
    [23.5611, 0, 0.0556],
    [26.3808, 0, 0.1023],
    [27.0044, 0, 0.118],
    [27.6546, 0, 0.1446],
    [27.9303, 0, 0.1729]
])

# 8FM lambda=30, 100, 200, 300, 3000
FM8_wPCA_wRE = RD_Curve()
FM8_wPCA_wRE.add_points([
    [25.1125, 0.8733, 0.1054],
    [27.8471, 0.9307, 0.18655],
    [28.3093, 0.9383, 0.2068],
    [28.885, 0.9488, 0.2434],
    [29.3374, 0.9563, 0.289]
])

ETHZ_CVPR18_MSSSIM = RD_Curve()
ETHZ_CVPR18_MSSSIM.add_points([
    [None, 0.9289356, 0.1265306],
    [None, 0.9417454, 0.1530612],
    [None, 0.9497924, 0.1795918],
    [None, 0.9553684, 0.2061224],
    [None, 0.9598574, 0.2326531],
    [None, 0.9636625, 0.2591837],
    [None, 0.9668663, 0.2857143],
    [None, 0.9695684, 0.3122449],
    [None, 0.9718446, 0.3387755],
    [None, 0.9738012, 0.3653061],
    [None, 0.9755308, 0.3918367],
    [None, 0.9770696, 0.4183673],
    [None, 0.9784622, 0.4448980],
    [None, 0.9797252, 0.4714286],
    [None, 0.9808753, 0.4979592],
    [None, 0.9819255, 0.5244898],
    [None, 0.9828875, 0.5510204],
    [None, 0.9837722, 0.5775510],
    [None, 0.9845877, 0.6040816],
    [None, 0.9853407, 0.6306122],
    [None, 0.9860362, 0.6571429],
    [None, 0.9866768, 0.6836735],
    [None, 0.9872690, 0.7102041],
    [None, 0.9878184, 0.7367347],
    [None, 0.9883268, 0.7632653],
    [None, 0.9887977, 0.7897959],
    [None, 0.9892346, 0.8163265],
    [None, 0.9896379, 0.8428571]
])

# Google_pytorch = RD_Curve()
# Google_pytorch.add_points([
#     [36.2842, 0.9896, 1.1643],  # 0415_1337
#     [32.0858, 0.9694, 0.5137],  # 0415_1339
#     [30.1185, 0.9500, 0.3266],  # 0415_1340
#     [26.2197, 0.8810, 0.1015],  # 0415_1341
# ])

Google_pytorch = RD_Curve()
Google_pytorch.add_points([
    [36.4098, 0.9896, 1.0923],  # 0810_1622
    [32.1168, 0.9687, 0.4691],  # 0810_1623
    [30.1392, 0.9492, 0.2959],  # 0810_1624
    [26.2420, 0.8793, 0.0981],  # 0810_1625
])

Google_pytorch_3 = RD_Curve()
Google_pytorch_3.add_points([
    [38.61, None, 1.45],
    [36.8, None, 1.11],
    [32.3, None, 0.5],
    [30.1, None, 0.3]
])

Google_OctConv = RD_Curve()
Google_OctConv.add_points([
    [33.3737, 0.9869, 0.9080],
    [31.2465, 0.9691, 0.4327],
    [29.5892, 0.9527, 0.2910],
    [25.5575, 0.8833, 0.0977]
])

Google_OctConv_high = RD_Curve()
Google_OctConv_high.add_points([
    [38.3478, 0.9933, 1.5984],
    [36.5839, 0.9898, 1.1463],
])

CConv = RD_Curve()  # 0416_2324
CConv.add_points([
    [37.0171, 0.9919, 1.4643],
    [36.0083, 0.9890, 1.1856],
    [32.2798, 0.9717, 0.5965],
    [30.2597, 0.9550, 0.4059],
    [26.1214, 0.8950, 0.1372],
])

CConv_bypass = RD_Curve()
CConv_bypass.add_points([
    [36.8796, 0.9915, 1.4738],  # 0421_0622
    [35.8592, 0.9884, 1.1612],
    [32.2691, 0.9698, 0.5704],
    [30.3811, 0.9528, 0.3710],
    [26.2928, 0.8786, 0.1211],
])

CConv_bypass_high = RD_Curve()
CConv_bypass_high.add_points([
    [38.5634, 0.9925, 1.6371],  # 0520_2117
    [36.7143, 0.9887, 1.2293],
    [31.8376, 0.9690, 0.5462],
    [29.8565, 0.9519, 0.3570],
    [26.1155, 0.8871, 0.1247],
])

CConv_masked = RD_Curve()
CConv_masked.add_points([
    [36.9117, 0.9917, 1.4488],
    [35.9630, 0.9886, 1.1336],
    [32.3266, 0.9703, 0.5292],
    [30.4708, 0.9537, 0.3406],
    [26.4348, 0.8825, 0.1151],
])

CConv_high = RD_Curve()
CConv_high.add_points([
    [38.4323, 0.9922, 1.6303],  # 0519_2205
    [36.8512, 0.9885, 1.2606],
    [32.2268, 0.9661, 0.5615],
    [30.2850, 0.9466, 0.3687],
    [26.0127, 0.8551, 0.1299],
])

CConv_masked_high = RD_Curve()
CConv_masked_high.add_points([
    [38.4207, 0.9924, 1.5732],  # 0517_2059
    [36.6578, 0.9885, 1.1823],
    [32.0028, 0.9677, 0.5036],
    [30.1523, 0.9507, 0.3249],
    [26.3112, 0.8794, 0.1080],
])

CConv_masked_high_share1 = RD_Curve()
CConv_masked_high_share1.add_points([
    [38.4220, 0.9923, 1.5515],
    [36.7117, 0.9886, 1.1562],
    [32.2746, 0.9695, 0.5091],
    [30.3469, 0.9529, 0.3349],
    [25.9086, 0.8728, 0.1143],
])

CConv_masked_high_share2 = RD_Curve()
CConv_masked_high_share2.add_points([
    [38.7385, 0.9926, 1.5900],
    [37.0211, 0.9893, 1.2059],
    [32.7119, 0.9730, 0.5416],
    [30.8440, 0.9586, 0.3622],
    [26.6769, 0.8964, 0.1280],
])

CConv_masked_high_share3 = RD_Curve()
CConv_masked_high_share3.add_points([
    [38.4995, 0.9922, 1.4982],
    [35.9858, 0.9879, 1.1543],
    [31.8576, 0.9711, 0.5221],
    [29.9341, 0.9564, 0.3553],
    [25.6051, 0.8883, 0.1319],
])

CConv_masked_high_imp = RD_Curve()
CConv_masked_high_imp.add_points([
    [37.6259, 0.9851, 1.4555],  # 0518_2332
    [35.7534, 0.9813, 1.0876],
    [31.2694, 0.9623, 0.4555],
    [29.3696, 0.9424, 0.2914],
    [25.7359, 0.8673, 0.0953],
])

CConv_masked_high_fixed_imp = RD_Curve()
CConv_masked_high_fixed_imp.add_points([
    [38.3839, 0.9924, 1.3386],  # 0519_0944
    [36.7544, 0.9889, 0.9923],
    [32.2949, 0.9715, 0.3884],
    [30.5572, 0.9587, 0.2332],
    [26.7027, 0.9018, 0.0613],
])

CConv_bypass_high_imp = RD_Curve()
CConv_bypass_high_imp.add_points([
    [37.5466, 0.9897, 1.4690],  # 0521_0257
    [35.6379, 0.9843, 1.1019],
    [30.8429, 0.9566, 0.4715],
    [28.9626, 0.9339, 0.3033],
    [25.2385, 0.8506, 0.0989],
])

CConv_bypass_high_imp_end_to_end = RD_Curve()
CConv_bypass_high_imp_end_to_end.add_points([
    [37.7689, 0.9884, 2.0298],  # 0525_1221
    [35.8900, 0.9822, 1.5947],
    [30.1256, 0.9437, 0.6514],
    [27.8031, 0.9108, 0.3735],
    [24.5158, 0.8320, 0.0965],
])

CConv_bypass_high_fixed_imp = RD_Curve()
CConv_bypass_high_fixed_imp.add_points([
    [38.2401, 0.9907, 1.5028],  # 0521_0357
    [36.3817, 0.9861, 1.1308],
    [31.8286, 0.9632, 0.5075],
    [29.9731, 0.9447, 0.3308],
    [26.0990, 0.8707, 0.1066],
])

CConv_bypass_high_ld_1e_1 = RD_Curve()
CConv_bypass_high_ld_1e_1.add_points([
    # [39.7801, 0.9945, 2.2497],
    # [39.0015, 0.9933, 1.8033],
    # [38.0918, 0.9917, 1.5100],
    # [37.0678, 0.9894, 1.3201],
    # [35.9584, 0.9865, 1.1958],
    # [34.7925, 0.9825, 1.1146]
    [39.6108, 0.9940, 1.7362],
    [38.8654, 0.9930, 1.5021],
    [38.0194, 0.9916, 1.3463],
    [37.0871, 0.9897, 1.2426],
    [36.0843, 0.9873, 1.1741],
    [35.0395, 0.9841, 1.1288]
])

CConv_bypass_high_ld_5e_2 = RD_Curve()
CConv_bypass_high_ld_5e_2.add_points([
    # [38.0062, 0.9916, 1.7352],
    # [37.1801, 0.9899, 1.3661],
    # [36.2207, 0.9874, 1.1238],
    # [35.1553, 0.9841, 0.9653],
    # [34.0173, 0.9797, 0.8620],
    # [32.8519, 0.9738, 0.7951]
    [37.4891, 0.9907, 1.2798],
    [36.6956, 0.9890, 1.0997],
    [35.8239, 0.9869, 0.9800],
    [34.8918, 0.9840, 0.9005],
    [33.9157, 0.9804, 0.8481],
    [32.9318, 0.9757, 0.8138]
])

CConv_bypass_high_ld_1e_2 = RD_Curve()
CConv_bypass_high_ld_1e_2.add_points([
    # [32.9659, 0.9759, 0.8390],
    # [32.2401, 0.9717, 0.6239],
    # [31.4125, 0.9659, 0.4860],
    # [30.5129, 0.9583, 0.3984],
    # [29.5684, 0.9483, 0.3440],
    # [28.6288, 0.9357, 0.3109]
    [32.5407, 0.9738, 0.5731],
    [31.8778, 0.9695, 0.4793],
    [31.1553, 0.9640, 0.4173],
    [30.3977, 0.9571, 0.3766],
    [29.6014, 0.9484, 0.3502],
    [28.8030, 0.9379, 0.3332]
])

CConv_bypass_high_ld_5e_3 = RD_Curve()
CConv_bypass_high_ld_5e_3.add_points([
    # [30.8369, 0.9617, 0.5636],
    # [30.2025, 0.9557, 0.4113],
    # [29.4788, 0.9476, 0.3149],
    # [28.7021, 0.9372, 0.2556],
    # [27.9054, 0.9243, 0.2199],
    # [27.1077, 0.9083, 0.1988]
    [30.6536, 0.9597, 0.3815],
    [30.0520, 0.9536, 0.3153],
    [29.3979, 0.9459, 0.2720],
    [28.6921, 0.9363, 0.2435],
    [27.9775, 0.9248, 0.2254],
    [27.2378, 0.9104, 0.2140]
])

CConv_bypass_high_ld_1e_3 = RD_Curve()
CConv_bypass_high_ld_1e_3.add_points([
    # [26.7310, 0.9052, 0.1896],
    # [26.3336, 0.8939, 0.1416],
    # [25.8914, 0.8798, 0.1122],
    # [25.3967, 0.8622, 0.0946],
    # [24.8780, 0.8419, 0.0842],
    # [24.3036, 0.8171, 0.0781]
    [26.8894, 0.9027, 0.1344],
    [26.4236, 0.8914, 0.1108],
    [25.9056, 0.8776, 0.0957],
    [25.3409, 0.8602, 0.0861],
    [24.6823, 0.8396, 0.0801],
    [23.9529, 0.8136, 0.0764]
])

CConv_bypass_high_3 = RD_Curve()
CConv_bypass_high_3.add_points([
    [38.4623, 0.9924, 1.3794],  # 0524_2343
    [36.4283, 0.9884, 1.0190],
    [31.7308, 0.9679, 0.4408],
    [29.8731, 0.9512, 0.2923],
    [26.2567, 0.8863, 0.1057],
])

Google_blur_5 = RD_Curve()
Google_blur_5.add_points([
    [38.4715, 0.9941, 2.2207],
    [35.8581, 0.9859, 1.3933],
    [33.3514, 0.9754, 0.7853],
    [29.3601, 0.9401, 0.3525]
])

Google_blur_3 = RD_Curve()
Google_blur_3.add_points([
    [38.4801, 0.9941, 2.2157],
    [35.8740, 0.9860, 1.3933],
    [33.3384, 0.9753, 0.7863],
    [29.1343, 0.9379, 0.3419]
])

Google_blur_1 = RD_Curve()
Google_blur_1.add_points([
    [38.4780, 0.9941, 2.2163],
    [35.8616, 0.9859, 1.3920],
    [33.3340, 0.9752, 0.7853],
    [29.1119, 0.9376, 0.3411]
])

Google_blur_0 = RD_Curve()
Google_blur_0.add_points([
    [38.4772, 0.9941, 2.2155],
    [35.8613, 0.9859, 1.3922],
    [33.3336, 0.9752, 0.7853],
    [29.0475, 0.9368, 0.3391]
])

Blur_0_pretrain = RD_Curve()
Blur_0_pretrain.add_points([
    [39.2055, 0.9960, 2.2031],
    [37.5121, 0.9922, 1.3141],
    [36.1966, 0.9887, 1.0018],
    [31.7508, 0.9670, 0.4716]
])

Blur_4_pretrain = RD_Curve()
Blur_4_pretrain.add_points([
    [39.1510, 0.9933, 1.4969],
    [37.0556, 0.9893, 1.0842],
    [35.4381, 0.9849, 0.8398],
    [31.7381, 0.9672, 0.4663],
    [30.2342, 0.9492, 0.2874],
    [26.4695, 0.8754, 0.0987],
    [24.9158, 0.8361, 0.0600]
])

Blur_0_from_scratch = RD_Curve()
Blur_0_from_scratch.add_points([
    [38.8269, 0.9955, 2.3755],
    [37.3850, 0.9924, 1.4562],
    [36.2672, 0.9895, 1.0975],
    [32.0846, 0.9686, 0.4762],
])

Blur_4_from_scratch = RD_Curve()
Blur_4_from_scratch.add_points([
    [38.2066, 0.9948, 2.3294],
    [36.9144, 0.9919, 1.4069],
    [36.0542, 0.9890, 1.0723],
    [31.8798, 0.9679, 0.4604],
])

new_arch_blur = RD_Curve()
new_arch_blur.add_points([
    [39.2708, 0.9959, 2.1524],
    [37.5332, 0.9922, 1.2951],
    [36.1733, 0.9885, 0.9791],
    [31.8201, 0.9675, 0.4620]
])

HPCoder_mean = RD_Curve()
HPCoder_mean.add_points([
    [26.6045, 0.8806, 0.0853],  # 0813_1237
    # [28.4879, 0.9207, 0.1530],  # 0818_1722
    # [30.3333, 0.9468, 0.2611],  # 0818_1723
    [31.9975, 0.9627, 0.4139],  # 0813_1238
    [34.3747, 0.9805, 0.6013],  # 0813_1239
    # [36.4154, 0.9872, 0.8839],  # 0818_1724
    # [38.5060, 0.9916, 1.2270],  # 0818_1725
    [39.2236, 0.9929, 1.3309],  # 0813_1240
])

HPCoder_mean_4p = RD_Curve()
HPCoder_mean_4p.add_points([
    [26.6045, 0.8806, 0.0853],  # 0813_1237
    [31.9975, 0.9627, 0.4139],  # 0813_1238
    [34.3747, 0.9805, 0.6013],  # 0813_1239
    [39.2236, 0.9929, 1.3309],  # 0813_1240
])

old_HPCoder_mean = RD_Curve()
old_HPCoder_mean.add_points([
    [26.6142, 0.8818, 0.0939],  # 0813_1237
    [32.2381, 0.9685, 0.4462],  # 0813_1238
    [34.0911, 0.9788, 0.6525],  # 0813_1239
    [39.3931, 0.9934, 1.5405],  # 0813_1240
])

multi_normal = RD_Curve()
multi_normal.add_points([
    [26.6484, 0.8871, 0.0948],
    [32.2007, 0.9686, 0.4452],
    [34.0388, 0.9789, 0.6470],
    [39.2933, 0.9933, 1.5045],
])

multi_scaleAE = RD_Curve()
multi_scaleAE.add_points([
    [26.6598, 0.8869, 0.0935],
    [32.1647, 0.9685, 0.4408],
    [34.0552, 0.9791, 0.6469],
    [39.3165, 0.9934, 1.5058],
])

single_normal = RD_Curve()
single_normal.add_points([
    [26.6033, 0.8854, 0.0935],
    [32.2124, 0.9689, 0.4501],
    [34.0540, 0.9789, 0.6519],
    [39.3951, 0.9935, 1.5365],
])

single_scaleAE = RD_Curve()
single_scaleAE.add_points([
    [26.6094, 0.8871, 0.0942],
    [32.2083, 0.9689, 0.4487],
    [34.0552, 0.9790, 0.6529],
    [39.4095, 0.9935, 1.5421],
])

multi_scalePyramid = RD_Curve()
multi_scalePyramid.add_points([
    [26.6432, 0.8848, 0.0941],
    [32.2307, 0.9690, 0.4406],
    [34.1410, 0.9791, 0.6392],
    [39.2864, 0.9933, 1.4475],
])

multi_distance = RD_Curve()
multi_distance.add_points([
    [26.6640, 0.8865, 0.0931],
    [32.1922, 0.9687, 0.4413],
    [34.2091, 0.9796, 0.6438],
    [39.2733, 0.9933, 1.4736],
])

scalePyramid_distortionPretrain = RD_Curve()
scalePyramid_distortionPretrain.add_points([
    [26.7468, 0.8761, 0.0857],
    [32.1676, 0.9673, 0.4148],
    [34.2629, 0.9800, 0.6196],
    [38.9753, 0.9922, 1.3652]
])

scalePyramidMod_distortionPretrain = RD_Curve()
scalePyramidMod_distortionPretrain.add_points([
    [26.7216, 0.8794, 0.0858],
    [32.1379, 0.9672, 0.4115],
    [34.2273, 0.9795, 0.6118],
    [39.0446, 0.9924, 1.3645]
])

scalePyramidMod_highPretrain = RD_Curve()
scalePyramidMod_highPretrain.add_points([
    [26.8044, 0.8812, 0.0878],  # 0829_1636
    [28.2763, 0.9161, 0.1538],  # 0829_1637
    [30.0260, 0.9415, 0.2621],  # 0829_1638
    [31.8436, 0.9596, 0.4109],
    [34.2987, 0.9797, 0.6086],  # 0829_1639
    [36.4125, 0.9872, 0.8630],  # 0829_1640
    [38.5171, 0.9918, 1.1860],  # 0829_1641
    [39.1699, 0.9928, 1.3003],
])

scalePyramidMod_full = RD_Curve()
scalePyramidMod_full.add_points([
    [26.8132, 0.8837, 0.0844],  # 0822_1628
    [28.4840, 0.9197, 0.1517],  # 0822_1633
    # [30.3765, 0.9473, 0.2586],  # 0822_1634
    [31.8436, 0.9596, 0.4109],  # 0822_1630
    [34.3799, 0.9803, 0.5945],  # 0822_1631
    [36.4510, 0.9873, 0.8616],  # 0822_1635
    [38.5012, 0.9916, 1.1902],  # 0822_1636
    [39.1699, 0.9928, 1.3003],  # 0822_1632
])

singleChannelPyramid = RD_Curve()
singleChannelPyramid.add_points([
    [26.7202, 0.8809, 0.0850],  # 0831_1852
    [30.3380, 0.9473, 0.2593],  # 0831_1853
    [34.3277, 0.9802, 0.5964],  # 0831_1854
    [39.2018, 0.9928, 1.3224],  # 0831_1855
])

singleChannelDoublePyramid = RD_Curve()
singleChannelDoublePyramid.add_points([
    [26.7451, 0.8818, 0.0850],  # 0831_1929
    [30.3690, 0.9477, 0.2598],  # 0831_1930
    [34.3389, 0.9803, 0.5966],  # 0831_1931
    [39.1697, 0.9927, 1.3176],  # 0831_1932
])

singleChannelKernelPyramid = RD_Curve()
singleChannelKernelPyramid.add_points([
    [26.7325, 0.8820, 0.0856],  # 0831_2233
    [30.3518, 0.9474, 0.2602],  # 0831_2234
    [34.3371, 0.9804, 0.5972],  # 0831_2235
    [39.1653, 0.9928, 1.3179],  # 0831_2236
])

scalePyramidMod_enc_opt = RD_Curve()
scalePyramidMod_enc_opt.add_points([
    [26.8599, 0.8881, 0.0838],
    [28.5455, 0.9232, 0.1500],
    # [],
    [31.8794, 0.9603, 0.4071],
    [34.4529, 0.9808, 0.5895],
    [36.5167, 0.9875, 0.8544],
    [38.5811, 0.9918, 1.1851],
    [39.2289, 0.9929, 1.2916],
])

mean_HPCoder = RD_Curve()
mean_HPCoder.add_points([
    [39.6269, 0.9939, 1.4765],
    [34.3874, 0.9805, 0.6184],
    [32.4183, 0.9699, 0.3925],
    [26.8890, 0.8945, 0.0797]
])

mean_HPCoder_1_update = RD_Curve()
mean_HPCoder_1_update.add_points([
    [38.9404, 0.9937, 1.5455],
    [34.2793, 0.9809, 0.6322],
    [32.4242, 0.9701, 0.3963],
    [26.9432, 0.8945, 0.0796]
])

mean_HPCoder_30_update = RD_Curve()
mean_HPCoder_30_update.add_points([
    [39.1141, 0.9937, 1.7585],
    [34.5419, 0.9820, 0.7496],
    [32.7824, 0.9720, 0.4462],
    [27.2753, 0.8987, 0.0914]
])

mean_HPCoder_30_update_slr = RD_Curve()
mean_HPCoder_30_update_slr.add_points([
    [39.6193, 0.9940, 1.4945],
    [34.6715, 0.9816, 0.6412],
    [32.6610, 0.9708, 0.4003],
    [27.0664, 0.8954, 0.0799]
])

mean_HPCoder_30_update_uslr = RD_Curve()
mean_HPCoder_30_update_uslr.add_points([
    [39.6804, 0.9939, 1.4767],
    [34.4622, 0.9807, 0.6192],
    [32.4668, 0.9700, 0.3921],
    [26.9233, 0.8945, 0.0793]
])

mean_HPCoder_100_update_slr = RD_Curve()
mean_HPCoder_100_update_slr.add_points([
    [39.4235, 0.9939, 1.5732],
    [34.7119, 0.9821, 0.6690],
    [32.7857, 0.9716, 0.4151],
    [27.1886, 0.8970, 0.0827]
])

mean_HPCoder_100_update_uslr = RD_Curve()
mean_HPCoder_100_update_uslr.add_points([
    [39.7087, 0.9940, 1.4804],
    [34.5555, 0.9811, 0.6234],
    [32.5365, 0.9703, 0.3931],
    [26.9705, 0.8948, 0.0791]
])

mean_HPCoder_100_update_uslr_dis = RD_Curve()
mean_HPCoder_100_update_uslr_dis.add_points([
    [39.8254, 0.9941, 1.5101],
    [34.6605, 0.9815, 0.6393],
    [32.6013, 0.9709, 0.4017],
    [27.0086, 0.8964, 0.0812]
])

mean_HPCoder_30_update_uhlr_latent = RD_Curve()
mean_HPCoder_30_update_uhlr_latent.add_points([
    [38.5055, 0.9926, 1.6706],
    [33.9449, 0.9794, 0.7309],
    [32.2763, 0.9689, 0.4660],
    [27.0272, 0.8937, 0.0943]
])

mean_HPCoder_30_update_hlr_latent = RD_Curve()
mean_HPCoder_30_update_hlr_latent.add_points([
    [39.7625, 0.9941, 1.4979],
    [34.5865, 0.9812, 0.6318],
    [32.6133, 0.9708, 0.4016],
    [27.0148, 0.8966, 0.0811]
])

mean_HPCoder_meta_30_update = RD_Curve()
mean_HPCoder_meta_30_update.add_points([
    [39.0882, 0.9936, 1.8225],
    [34.5396, 0.9820, 0.7509],
    [32.7826, 0.9720, 0.4479],
    [27.3314, 0.8997, 0.0932]
])

mean_HPCoder_meta_1_update = RD_Curve()
mean_HPCoder_meta_1_update.add_points([
    [38.4076, 0.9935, 1.5266],
    [34.2724, 0.9809, 0.6310],
    [32.4387, 0.9702, 0.3969],
    [26.9831, 0.8950, 0.0804]
])

mean_HPCoder_meta = RD_Curve()
mean_HPCoder_meta.add_points([
    [39.5875, 0.9939, 1.4791],
    [34.3788, 0.9805, 0.6196],
    [32.4217, 0.9699, 0.3933],
    [26.9352, 0.8950, 0.0804]
])

mean_HPCoder_meta2_1_update = RD_Curve()
mean_HPCoder_meta2_1_update.add_points([
    [39.6175, MS-SSIM: 0.9939, rate: 1.4758],
    [34.3788, 0.9805, 0.6196],
    [32.4217, 0.9699, 0.3933],
    [26.9352, 0.8950, 0.0804]
])

fig_plot = RD_Plot()

fig_plot.add_curve(BPG.bpp, BPG.PSNR, color='black', label='BPG')
fig_plot.add_curve(mean_HPCoder.bpp, mean_HPCoder.PSNR, label='HPCoder w/ mean (baseline)', marker='x')
fig_plot.add_curve(mean_HPCoder_1_update.bpp, mean_HPCoder_1_update.PSNR, label='baseline 1 update lr=1e-4', marker='x')
fig_plot.add_curve(mean_HPCoder_30_update.bpp, mean_HPCoder_30_update.PSNR, label='baseline 30 update lr=1e-4', marker='x')
# fig_plot.add_curve(mean_HPCoder_30_update_slr.bpp, mean_HPCoder_30_update_slr.PSNR, label='baseline 30 update lr=1e-5', marker='x')
# fig_plot.add_curve(mean_HPCoder_100_update_slr.bpp, mean_HPCoder_100_update_slr.PSNR, label='baseline 100 update lr=1e-5', marker='x')
# fig_plot.add_curve(mean_HPCoder_30_update_uslr.bpp, mean_HPCoder_30_update_uslr.PSNR, label='baseline 30 update lr=1e-6', marker='x')
# fig_plot.add_curve(mean_HPCoder_100_update_uslr.bpp, mean_HPCoder_100_update_uslr.PSNR, label='baseline 100 update lr=1e-6', marker='x')
# fig_plot.add_curve(mean_HPCoder_100_update_uslr_dis.bpp, mean_HPCoder_100_update_uslr_dis.PSNR, label='baseline 100 update lr=1e-6 distortion', marker='x')
# fig_plot.add_curve(mean_HPCoder_30_update_uhlr_latent.bpp, mean_HPCoder_30_update_uhlr_latent.PSNR, label='baseline 30 update lr=1e-2 latent', marker='x')
# fig_plot.add_curve(mean_HPCoder_30_update_hlr_latent.bpp, mean_HPCoder_30_update_hlr_latent.PSNR, label='baseline 30 update lr=1e-3 latent', marker='x')
fig_plot.add_curve(mean_HPCoder_meta.bpp, mean_HPCoder_meta.PSNR, label='meta', marker='x')
fig_plot.add_curve(mean_HPCoder_meta_1_update.bpp, mean_HPCoder_meta_1_update.PSNR, label='meta 1 update lr=1e-4', marker='x')
fig_plot.add_curve(mean_HPCoder_meta_30_update.bpp, mean_HPCoder_meta_30_update.PSNR, label='meta 30 update lr=1e-4', marker='x')

fig_plot.plot()
