import torch
import random
import os
from glob import glob
import subprocess
import numpy as np
import math
import torch.nn as nn

from torch import stack
from torch.utils.data import Dataset as torchData
from torch.utils.data import DataLoader
from torchvision import transforms
from subprocess import Popen, PIPE

from util.vision import imgloader, rgb_transform
from PIL import Image

class BVI_DVC_dataset(torchData):
    """Video Dataset with 265/264 Iframe

    Args:
        root
        mode
        frames
        transform
    """

    def __init__(self, root, qp: int, frames, transform=rgb_transform):
        super().__init__()
        #self.folder = sorted(glob(root + 'Sequences/D*/'))
        self.folder = sorted(glob(root + 'Cropped_Sequences/*/'))
        self.frames = frames
        self.transform = transform

        self.qp = qp
        self.Iframe_dir = f'iframe/qp_{qp}/'

        self.gop_list = []

        for path in self.folder:
            gop_num = 64 // frames

            for gop_idx in range(gop_num):
                self.gop_list.append([path,
                                      1 + frames * gop_idx,
                                      1 + frames * (gop_idx + 1)])

        self.prepare_iframe()

    def prepare_iframe(self):
        print("Preparing I-frame for training data....")
        for gop in self.gop_list:
            seq_path, frame_start, _ = gop

            frame_path = os.path.join(seq_path, 'frame_{:02d}.png'.format(frame_start))
            iframe_path = frame_path.replace('Sequences', self.Iframe_dir)
            bin_path = iframe_path.replace('qp', 'bin')

            if not os.path.exists(iframe_path):
                os.makedirs(os.path.dirname(iframe_path), exist_ok=True)
                os.makedirs(os.path.dirname(bin_path), exist_ok=True)

                subprocess.call(f'bpgenc -q {self.qp} -o {bin_path} {frame_path}'.split(' '))
                subprocess.call(f'bpgdec -o {iframe_path} {bin_path}'.split(' '))

    def __len__(self):
        return len(self.gop_list)

    @property
    def info(self):
        gop = self[0]
        return "\nGop size: {}".format(gop.shape)

    def __getitem__(self, index):
        seq_path, frame_start, frame_end = self.gop_list[index]
        seed = random.randint(0, 1e9)
        imgs = []

        for f in range(frame_start, frame_end):
            random.seed(seed)
            torch.manual_seed(seed)

            frame_path = os.path.join(seq_path, 'frame_{:02d}.png'.format(f))
            if f == frame_start:
                file = frame_path.replace('Sequences', self.Iframe_dir)
            else:
                file = frame_path
            imgs.append(self.transform(imgloader(file)))

        return stack(imgs)


class VideoData(torchData):
    """Video Dataset

    Args:
        root
        mode
        frames
        transform
    """

    def __init__(self, root, frames, transform=rgb_transform):
        super().__init__()
        self.folder = glob(root + 'img/*/*/')
        self.frames = frames
        self.transform = transform

    def __len__(self):
        return len(self.folder)
        #return 1

    @property
    def info(self):
        gop = self[0]
        return "\nGop size: {}".format(gop.shape)

    def __getitem__(self, index):
        path = self.folder[index]
        seed = random.randint(0, 1e9)
        imgs = []
        for f in range(self.frames):
            random.seed(seed)
            file = path + str(f) + '.png'
            imgs.append(self.transform(imgloader(file)))

        return stack(imgs)


class VideoDataIframe(VideoData):
    """Video Dataset with 265/264 Iframe

    Args:
        root
        mode
        frames
        transform
    """

    def __init__(self, root, mode, frames, transform=rgb_transform, bpg=True):
        super().__init__(root, frames, transform)
        #self.folder = glob(root + 'img/00014/0818/')
        self.mode = mode
        self.bpg = bpg
        self.Iframe_dir = 'Iframe{}'.format(self.mode.split('_')[0])
        
        black_seqs = []
        #for training_seq in self.folder:
        #    file = training_seq + '0.png'
        #    img = self.transform(imgloader(file))
        #    if len(torch.nonzero(img)) < 10:
        #        print(training_seq, ': ', len(torch.nonzero(img)))
        #        black_seqs.append(training_seq)
        #f = open('dump_info/black_seq.txt', 'r')
        #seq = f.readline()
        #while seq:
        #    seq = seq[:-1]
        #    print(seq)
        #    self.folder.remove(seq)
        #    seq = f.readline()

    def __getitem__(self, index):
        path = self.folder[index]
        seed = random.randint(0, 1e9)
        imgs = []
        for f in range(self.frames):
            random.seed(seed)
            if f == 0 and self.bpg:
                file = path.replace('img', self.Iframe_dir) + 'Iframe' + self.mode + '_' + str(f + 1) + '.png'
            else:
                file = path + str(f) + '.png'
            imgs.append(self.transform(imgloader(file)))

        return stack(imgs)
        #return stack(imgs), path



class VideoTestDataIframe(torchData):
    def __init__(self, root, lmda, first_gop=False, sequence=('U', 'B'), GOP=12):
        super(VideoTestDataIframe, self).__init__()
        
        assert GOP in [12, 16, 32], ValueError
        self.root = root
        self.lmda = lmda
        self.qp = {256: 37, 512: 32, 1024: 27, 2048: 22, 4096: 22}[lmda]

        self.seq_name = []
        seq_len = []
        gop_size = []
        dataset_name_list = []
        if 'U' in sequence:
            self.seq_name.extend(['Beauty', 'Bosphorus', 'HoneyBee', 'Jockey', 'ReadySteadyGo', 'ShakeNDry', 'YachtRide'])
            if GOP in [12, 16]:
                seq_len.extend([600, 600, 600, 600, 600, 300, 600])
            else:
                seq_len.extend([96]*7)
                # seq_len.extend([600, 600, 600, 600, 600, 300, 600])
            gop_size.extend([GOP]*7)
            dataset_name_list.extend(['UVG']*7)

            #seq_len.extend([120]*7)
            #self.seq_name.extend(['Beauty'])
            #seq_len.extend([96])
            #gop_size.extend([GOP])
            #dataset_name_list.extend(['UVG'])
        if 'B' in sequence:
            self.seq_name.extend(['Kimono1', 'BQTerrace', 'Cactus', 'BasketballDrive', 'ParkScene'])
            if GOP in [12, 16]:
                seq_len.extend([100]*5)
                if GOP == 12:
                    GOP=10
                #seq_len.extend([240, 600, 500, 500, 240])
            else:
                seq_len.extend([96]*5)
                # seq_len.extend([100]*5)
            gop_size.extend([GOP]*5)
            dataset_name_list.extend(['HEVC-B']*5)
            #self.seq_name.extend(['BQTerrace', 'Cactus', 'ParkScene'])
            #seq_len.extend([96]*3)
            #gop_size.extend([GOP]*3)
            #dataset_name_list.extend(['HEVC-B']*3)
        if 'C' in sequence:
            #self.seq_name.extend(['BasketballDrill', 'BQMall', 'PartyScene', 'RaceHorses'])
            self.seq_name.extend(['BasketballDrill-832x448', 'BQMall-832x448', 'PartyScene-832x448', 'RaceHorses-832x448'])
            if GOP in [12, 16]:
                seq_len.extend([100]*4)
                if GOP == 12:
                    GOP=10
            else:
                seq_len.extend([96]*4)
            gop_size.extend([GOP]*4)
            #dataset_name_list.extend(['HEVC-C', 'HEVC-C', 'HEVC-C', 'HEVC-C', ])
            dataset_name_list.extend(['HEVC-C-832x448']*4)
        if 'D' in sequence:
            #self.seq_name.extend(['BasketballPass', 'BQSquare', 'BlowingBubbles', 'RaceHorses1'])
            self.seq_name.extend(['BasketballPass-384x192', 'BQSquare-384x192', 'BlowingBubbles-384x192', 'RaceHorses1-384x192'])
            if GOP in [12, 16]:
                seq_len.extend([100]*4)
                if GOP == 12:
                    GOP=10
            else:
                seq_len.extend([96]*4)
            gop_size.extend([GOP]*4)
            #dataset_name_list.extend(['HEVC-D', 'HEVC-D', 'HEVC-D', 'HEVC-D', ])
            dataset_name_list.extend(['HEVC-D-384x192']*4)
        if 'E' in sequence:
            #self.seq_name.extend(['vidyo1', 'vidyo3', 'vidyo4'])
            self.seq_name.extend(['vidyo1-1280x704', 'vidyo3-1280x704', 'vidyo4-1280x704'])
            if GOP in [12, 16]:
                seq_len.extend([100]*3)
                if GOP == 12:
                    GOP=10
            else:
                seq_len.extend([96]*3)
            gop_size.extend([GOP]*3)
            #dataset_name_list.extend(['HEVC-E', 'HEVC-E', 'HEVC-E'])
            dataset_name_list.extend(['HEVC-E-1280x704']*3)
        if 'M' in sequence:
            # MCL_list = ['videoSRC01', 'videoSRC12', 'videoSRC13', 'videoSRC18', 'videoSRC20']
            MCL_list = []
            for i in range(1, 31):
               MCL_list.append('videoSRC'+str(i).zfill(2))
            #    
            self.seq_name.extend(MCL_list)
            if GOP in [12, 16]:
               seq_len.extend([150, 150, 150, 150, 125, 125, 125, 125, 125, 150,
                               150, 150, 150, 150, 150, 150, 120, 125, 150, 125,
                               120, 120, 120, 120, 120, 150, 150, 150, 120, 150])
            else:
               seq_len.extend([96]*30)
            gop_size.extend([GOP]*30)
            dataset_name_list.extend(['MCL_JCV']*30)

        ### K is CLIC 2022 dataset
        if 'K' in sequence:
            self.seq_name.extend(['a06845dd7d1d808e4f4743b7f08f2bf75a9a72264d4fb16505caf6e334611003',
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
                                  '9299df423938da4fd7f51736070420d2bb39d33972729b46a16180d07262df12'])
        
            dataset_name_list.extend(['CLIC_2022'] * 30)
            if GOP == 12:
                seq_len.extend([302, 244, 252, 252, 244, 255, 243, 303, 303, 302, 305, 602, 305, 252, 305, 254, 303, 304, 242,
                                302, 255, 302, 255, 303, 254, 255, 252, 294, 505, 255])
                gop_size.extend([12] * 30)
            else:
                seq_len.extend([96] * 30)
                gop_size.extend([32] * 30)

            gop_size.extend([32] * 30)
            #self.seq_name.extend([
            #    '209921b14cef20d62002e2b0c21ad692226135b52fee7eead315039ca51c470c',
            #    '5d8f03cf5c6a469004a0ca73948ad64fa6d222b3b807f155a66684387f5d208a'
            #])
            #seq_len.extend([96]*2)
            #gop_size.extend([GOP]*2)
            #dataset_name_list.extend(['CLIC_2022']*2)
        if 'U_small' in sequence:
            self.seq_name.extend(['Beauty', 'Bosphorus', 'HoneyBee', 'Jockey', 'ReadySteadyGo', 'ShakeNDry', 'YachtRide'])
            if GOP in [12, 16]:
                seq_len.extend([600, 600, 600, 600, 600, 300, 600])
            else:
                seq_len.extend([96]*7)
                # seq_len.extend([600, 600, 600, 600, 600, 300, 600])
            gop_size.extend([GOP]*7)
            dataset_name_list.extend(['UVG_small']*7)


        seq_len = dict(zip(self.seq_name, seq_len))
        gop_size = dict(zip(self.seq_name, gop_size))
        dataset_name_list = dict(zip(self.seq_name, dataset_name_list))

        self.gop_list = []

        for seq_name in self.seq_name:
            if first_gop:
                gop_num = 1
            else:
                gop_num = seq_len[seq_name] // gop_size[seq_name]
                # gop_num = int(math.ceil(seq_len[seq_name] / gop_size[seq_name]))
            for gop_idx in range(gop_num):
                # if gop_idx == gop_num - 1:
                #     self.gop_list.append([dataset_name_list[seq_name],
                #                           seq_name,
                #                           1 + gop_size[seq_name] * gop_idx,
                #                           2 + gop_size[seq_name] * gop_idx,])

                self.gop_list.append([dataset_name_list[seq_name],
                                    seq_name,
                                    1 + gop_size[seq_name] * gop_idx,
                                    1 + gop_size[seq_name] * (gop_idx + 1)])
        
    def __len__(self):
        return len(self.gop_list)
        #return 1

    def __getitem__(self, idx):
        dataset_name, seq_name, frame_start, frame_end = self.gop_list[idx]
        seed = random.randint(0, 1e9)
        imgs = []

        for frame_idx in range(frame_start, frame_end):
            random.seed(seed)

            #raw_path = os.path.join(self.root, 'raw_video_1080', seq_name, 'frame_{:d}.png'.format(frame_idx))
            if dataset_name == 'CLIC_2022':
                raw_path = os.path.join(self.root, dataset_name, 'rgb', seq_name, f'frame_{str(frame_idx)}.png')
            elif dataset_name == 'UVG_small':
                raw_path = os.path.join(self.root, 'TestVideo', 'raw_video_270', seq_name, 'frame_{:d}.png'.format(frame_idx))
            else:
                raw_path = os.path.join(self.root, 'TestVideo', 'raw_video_1080', seq_name, 'frame_{:d}.png'.format(frame_idx))

            if frame_idx == frame_start:
                # img_path = os.path.join(self.root, 'bpg', str(self.qp), 'decoded', seq_name, f'frame_{frame_idx}.png')

                # if not os.path.exists(img_path):
                #     # Compress data on-the-fly when they are not previously compressed.
                #     bin_path = img_path.replace('decoded', 'bin').replace('png', 'bin')

                #     os.makedirs(os.path.dirname(bin_path), exist_ok=True)
                #     os.makedirs(os.path.dirname(img_path), exist_ok=True)

                #     subprocess.call(f'bpgenc -f 444 -q {self.qp} -o {bin_path} {raw_path}'.split(' '))
                #     subprocess.call(f'bpgdec -o {img_path} {bin_path}'.split(' '))

                imgs.append(transforms.ToTensor()(imgloader(raw_path)))
            
            imgs.append(transforms.ToTensor()(imgloader(raw_path)))

        
        return dataset_name, seq_name, stack(imgs), frame_start
