import argparse
import os
import csv
from skimage import io
from functools import partial

import numpy as np
import torch
import torch_compression as trc
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from torch import nn, optim
from torch.utils.data import DataLoader
from torch_compression.modules.entropy_models import EntropyBottleneck
from torchvision import transforms

from dataloader import VideoDataIframe, VideoTestDataIframe
from models import *
from util.psnr import mse2psnr
from util.vision import PlotFlow, save_image
from util.ssim import MS_SSIM
from util.sampler import Resampler
from flownets import PWCNet

phase = {'init': 0, 'trainFeat': 0, 'trainME': 20, 'trainMV': 30, 'trainRes': 50, 'trainALL': 200}

iframe_byte = {'BasketballDrive': [419262, 123510, 57019, 29717],
               'BQTerrace': [605223, 339686, 165894, 87796],
               'Cactus': [602907, 251346, 120563, 61076],
               'Kimono1': [211160, 102242, 56147, 31387],
               'ParkScene': [533806, 279359, 142826, 68741],
               'Beauty': [434972.5, 119323.0, 30791.7, 11012.4],
               'Bosphorus': [176082.9, 91791.6, 47956.2, 24223.6],
               'HoneyBee': [268335.9, 127247.8, 70806.1, 41526.4],
               'Jockey': [160811.12, 60595.54, 30956.46, 16949.18],
               'ReadySteadyGo': [264395.3, 150905.0, 86937.7, 49373.4],
               'ShakeNDry': [317914.0, 166663.7, 92376.8, 50788.6],
               'YachtRide': [251661., 141493.7, 78157., 41076.2]}
plot_flow = PlotFlow().cuda()


class CompressesModel(LightningModule):
    """Basic Compress Model"""

    def __init__(self):
        super(CompressesModel, self).__init__()

    def named_main_parameters(self, prefix=''):
        for name, param in self.named_parameters(prefix=prefix, recurse=True):
            if 'quantiles' not in name:
                yield (name, param)

    def main_parameters(self):
        for _, param in self.named_main_parameters():
            yield param

    def named_aux_parameters(self, prefix=''):
        for name, param in self.named_parameters(prefix=prefix, recurse=True):
            if 'quantiles' in name:
                yield (name, param)

    def aux_parameters(self):
        for _, param in self.named_aux_parameters():
            yield param

    def aux_loss(self):
        aux_loss = []
        for m in self.modules():
            if isinstance(m, EntropyBottleneck):
                aux_loss.append(m.aux_loss())

        return torch.stack(aux_loss).mean()


class Pframe(CompressesModel):
    """torchDVC Pframe"""
    train_dataset: VideoDataIframe
    val_dataset: VideoTestDataIframe
    test_dataset: VideoTestDataIframe

    def __init__(self, args, m_coder, r_coder):
        super(Pframe, self).__init__()
        self.hparams = args

        self.criterion = nn.MSELoss(reduction='none') if not self.hparams.ssim else MS_SSIM(data_range=1.).cuda()

        if args.restore == 'resume':
            self.feature_extract = FeatureExtractor()

            self.me_net = StackedMotionEstimation()

            self.Motion = m_coder(in_channels=64, out_channels=64, kernel_size=3)

            self.MCNet = MotionCompensation()
            self.FrameRec = FrameReconstruction()

            self.Residual = r_coder()
        else:
            self.me_net = PWCNet()
            self.Resampler = Resampler()

    def load_args(self, args):
        self.hparams = args

    def motion_estimate(self, ref_frame, coding_frame):
        if self.hparams.restore == 'resume':
            feat0_com = self.feature_extract(ref_frame)
            feat1_raw = self.feature_extract(coding_frame)

            motion = self.me_net(torch.cat([feat0_com, feat1_raw], dim=1))
            feat1_com = self.MCNet(feat0_com, motion)

            mc_frame = self.FrameRec(feat1_com)

            return mc_frame, motion
        else:
            motion = self.me_net(ref_frame, coding_frame)
            rec_frame = self.Resampler(ref_frame, motion)

            return rec_frame, motion

    def motion_forward(self, ref_frame, coding_frame):
        feat0_com = self.feature_extract(ref_frame)
        feat1_raw = self.feature_extract(coding_frame)

        motion = self.me_net(torch.cat([feat0_com, feat1_raw], dim=1))
        motion_hat, likelihood_m = self.Motion(motion)
        feat1_com = self.MCNet(feat0_com, motion_hat)

        mc_frame = self.FrameRec(feat1_com)

        return mc_frame, likelihood_m, [likelihood_m]

    def forward(self, ref_frame, coding_frame, p_order=0, scale_factor=1):
        mc_frame, likelihood_m, motion_info = self.motion_forward(ref_frame, coding_frame)

        res_frame = coding_frame - mc_frame
        res_hat, likelihood_r = self.Residual(res_frame)
        reconstructed = mc_frame + res_hat

        likelihoods = likelihood_m + likelihood_r
        return reconstructed, likelihoods, {'mc_frame': mc_frame}

    def training_step(self, batch, batch_idx):
        epoch = self.current_epoch
        batch = batch.cuda()

        if epoch <= phase['trainFeat']:
            _phase = 'Feat'
            coding_frame = batch[:, 2]

            feat1_raw = self.feature_extract(coding_frame)

            reconstructed = self.FrameRec(feat1_raw)

            distortion = self.criterion(coding_frame, reconstructed)

            loss = distortion.mean()

            psnr = mse2psnr(distortion.mean().item())
            rate = 0.

            logs = {'train/loss': loss.item(),
                    'train/psnr': psnr,
                    'train/rate': rate}

        elif epoch <= phase['trainME']:
            _phase = 'ME'
            # Take uncompressed frame as reference frame (first frame is compressed I-frame)
            ref_frame = batch[:, 1]
            coding_frame = batch[:, 2]
            reconstructed = self.motion_estimate(ref_frame, coding_frame)

            distortion = self.criterion(coding_frame, reconstructed)

            loss = self.hparams.lmda * distortion.mean()

            psnr = mse2psnr(distortion.mean().item())
            rate = 0.

            logs = {'train/loss': loss.item(),
                    'train/psnr': psnr,
                    'train/rate': rate}

        elif epoch <= phase['trainMV']:
            _phase = 'MV'

            if epoch <= phase['trainMV'] - 1:
                self.requires_grad_(False)
                self.Motion.requires_grad_(True)
            else:
                self.requires_grad_(True)

            # Take uncompressed frame as reference frame (first frame is compressed I-frame)
            ref_frame = batch[:, 1]
            coding_frame = batch[:, 2]
            reconstructed, likelihoods, _ = self.motion_forward(ref_frame, coding_frame)

            distortion = self.criterion(coding_frame, reconstructed)
            rate = trc.estimate_bpp(likelihoods, input=coding_frame)

            loss = self.hparams.lmda * distortion.mean() + rate.mean()

            psnr = mse2psnr(distortion.mean().item())
            rate = rate.mean().item()

            logs = {'train/loss': loss.item(),
                    'train/psnr': psnr,
                    'train/rate': rate}

        else:
            if epoch <= phase['trainRes']:
                _phase = 'RES'
            else:
                _phase = 'ALL'
            ref_frame = batch[:, 0]
            reconstructed = batch[:, 0]

            loss = torch.tensor(0., dtype=torch.float, device=ref_frame.device)
            dist_list = []
            rate_list = []
            frame_count = 0

            for frame_idx in range(1, 7):
                frame_count += 1
                ref_frame = reconstructed

                if _phase == 'RES':
                    if frame_idx > 1:
                        break
                    ref_frame = ref_frame.detach()

                else:
                    if (frame_idx - 1) * 3 > epoch - phase['trainRes']:
                        break

                coding_frame = batch[:, frame_idx]

                reconstructed, likelihoods, _ = self(ref_frame, coding_frame)

                distortion = self.criterion(coding_frame, reconstructed)
                rate = trc.estimate_bpp(likelihoods, input=coding_frame)

                loss = loss + self.hparams.lmda * distortion.mean() + rate.mean()
                dist_list.append(distortion.mean())
                rate_list.append(rate.mean())

            distortion = torch.mean(torch.tensor(dist_list))
            rate = torch.mean(torch.tensor(rate_list))
            loss = loss / frame_count

            psnr = mse2psnr(distortion.mean().item())
            rate = rate.mean().item()

            logs = {'train/loss': loss.item(),
                    'train/psnr': psnr,
                    'train/rate': rate}

        if epoch > phase['trainFeat']:
            if epoch <= phase['trainMV']:
                aux_loss = self.Motion.aux_loss()
            else:
                aux_loss = self.aux_loss()

            logs['train/aux_loss'] = aux_loss.item()

            loss = loss + aux_loss

        self.log_dict(logs)

        self.log('psnr', psnr, prog_bar=True, logger=False)
        self.log('rate', rate, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch, batch_idx):
        seq_name, batch, _ = batch
        seq_name = seq_name[0]

        ref_frame = batch[:, 0]
        batch = batch[:, 1:]

        epoch = self.current_epoch
        gop_size = batch.size(1)

        height, width = ref_frame.size()[2:]

        psnr_list = []
        rate_list = []
        loss_list = []
        align = trc.util.Alignment(divisor=256.)

        for frame_idx in range(gop_size):
            if frame_idx != 0:
                if epoch <= phase['trainFeat']:
                    feat1_raw = self.feature_extract(align.align(batch[:, frame_idx]))

                    ref_frame = self.FrameRec(feat1_raw)
                    rate = 1.

                elif epoch <= phase['trainME']:
                    ref_frame = self.motion_estimate(align.align(ref_frame), align.align(batch[:, frame_idx]))
                    rate = 1.

                elif epoch <= phase['trainMV']:
                    ref_frame, likelihoods, _ = self.motion_forward(align.align(ref_frame),
                                                                    align.align(batch[:, frame_idx]))
                    rate = trc.estimate_bpp(likelihoods, input=ref_frame).mean().item()
                else:
                    ref_frame, likelihoods, _ = self(align.align(ref_frame),
                                                     align.align(batch[:, frame_idx]))
                    rate = trc.estimate_bpp(likelihoods, input=ref_frame).mean().item()

                ref_frame = align.resume(ref_frame)

            else:
                intra_index = {256: 3, 512: 2, 1024: 1, 2048: 0}[self.hparams.lmda]

                rate = iframe_byte[seq_name][intra_index] * 8 / height / width

            mse = self.criterion(ref_frame, batch[:, frame_idx]).mean().item()
            psnr = mse2psnr(mse)
            loss = self.hparams.lmda * mse + rate

            psnr_list.append(psnr)
            rate_list.append(rate)
            loss_list.append(loss)

        psnr = np.mean(psnr_list)
        rate = np.mean(rate_list)
        loss = np.mean(loss_list)

        logs = {'seq_name': seq_name, 'val_loss': loss, 'val_psnr': psnr, 'val_rate': rate}

        return {'val_log': logs}

    def validation_epoch_end(self, outputs):
        dataset_name = {'HEVC': ['BasketballDrive', 'BQTerrace', 'Cactus', 'Kimono1', 'ParkScene'],
                        'UVG': ['Beauty', 'Bosphorus', 'HoneyBee', 'Jockey', 'ReadySteadyGo', 'ShakeNDry', 'YachtRide']}

        uvg_rd = {'psnr': [], 'rate': []}
        hevc_rd = {'psnr': [], 'rate': []}
        loss = []

        for logs in [log['val_log'] for log in outputs]:
            seq_name = logs['seq_name']

            if seq_name in dataset_name['UVG']:
                uvg_rd['psnr'].append(logs['val_psnr'])
                uvg_rd['rate'].append(logs['val_rate'])
            elif seq_name in dataset_name['HEVC']:
                hevc_rd['psnr'].append(logs['val_psnr'])
                hevc_rd['rate'].append(logs['val_rate'])
            else:
                print("Unexpected sequence name:", seq_name)
                raise NotImplementedError

            loss.append(logs['val_loss'])

        avg_loss = np.mean(loss)

        logs = {'val/loss': avg_loss,
                'val/UVG psnr': np.mean(uvg_rd['psnr']), 'val/UVG rate': np.mean(uvg_rd['rate']),
                'val/HEVC-B psnr': np.mean(hevc_rd['psnr']), 'val/HEVC-B rate': np.mean(hevc_rd['rate'])}

        self.log_dict(logs)

        return None

    def plot_flow(self, ref_frame, coding_frame, seq_name, frame_idx):
        feat0_com = self.feature_extract(ref_frame)
        feat1_raw = self.feature_extract(coding_frame)

        motion = self.me_net(torch.cat([feat0_com, feat1_raw], dim=1))
        motion_hat, likelihood_m = self.Motion(motion)
        dcn_motion = self.MCNet.get_weighted_flow(motion_hat)

        _, _, h, w = dcn_motion.shape
        dcn_motion = dcn_motion.view(-1, 2, h, w)

        flow_map = [plot_flow(dcn_motion[i*9:(i+1)*9]) for i in range(8)]

        os.makedirs(self.hparams.save_dir + f'/flow/{seq_name}', exist_ok=True)
        for group in range(1, 9):
            save_image(flow_map[group-1],
                       self.hparams.save_dir + f'/flow/{seq_name}/f{int(frame_idx)}_flow_g{group}.png', nrow=3)

        return

    def test_step(self, batch, batch_idx):
        seq_name, batch, frame_id_start = batch
        frame_id = int(frame_id_start)
        seq_name = seq_name[0]

        ref_frame = batch[:, 0]
        batch = batch[:, 1:]

        gop_size = batch.size(1)

        height, width = ref_frame.size()[2:]
        estimate_bpp = partial(trc.estimate_bpp, num_pixels=height * width)

        psnr_list = []
        mse_list = []
        rate_list = []
        log_list = []
        align = trc.util.Alignment(divisor=256.)

        code = 'fvc' if self.hparams.restore == 'resume' else 'pwc'
        os.makedirs(self.hparams.save_dir + f'/me_dump_{code}/{seq_name}', exist_ok=True)

        for frame_idx in range(gop_size):
            if frame_idx != 0:
                rec_frame, motion = self.motion_estimate(align.align(ref_frame), align.align(batch[:, frame_idx]))
                rec_frame = align.resume(rec_frame)

                if self.hparams.restore != 'resume' and frame_id_start < 20:
                    os.makedirs(self.hparams.save_dir + f'/flow/{seq_name}', exist_ok=True)
                    flow_map = align.resume(plot_flow(motion))
                    save_image(flow_map, self.hparams.save_dir + f'/flow/{seq_name}/f{int(frame_idx)}_flow.png',
                               nrow=1)
                    np.save(self.hparams.save_dir + f'/flow/{seq_name}/f{int(frame_idx)}_flow.npy',
                            motion[0].cpu().numpy())

                ref_frame = rec_frame

                mse = self.criterion(ref_frame, batch[:, frame_idx]).mean().item()
                psnr = mse2psnr(mse)

                # save_image(ref_frame[0], self.hparams.save_dir + f'/me_dump_{code}/{seq_name}/f{frame_idx}.png')

                rate = 0.

                log_list.append({'psnr': psnr, 'rate': rate})

            else:
                intra_index = {256: 3, 512: 2, 1024: 1, 2048: 0}[self.hparams.lmda]

                rate = iframe_byte[seq_name][intra_index] * 8 / height / width
                mse = self.criterion(ref_frame, batch[:, frame_idx]).mean().item()
                psnr = mse2psnr(mse)

                log_list.append({'psnr': psnr, 'rate': rate})

            psnr_list.append(psnr)
            rate_list.append(rate)
            mse_list.append(mse)

            frame_id += 1

        psnr = np.mean(psnr_list)
        rate = np.mean(rate_list)

        logs = {'seq_name': seq_name, 'test_psnr': psnr, 'test_rate': rate,
                'batch_idx': batch_idx, 'log_list': log_list}

        return {'test_log': logs}

    def test_epoch_end(self, outputs):
        dataset_name = {'HEVC': ['BasketballDrive', 'BQTerrace', 'Cactus', 'Kimono1', 'ParkScene'],
                        'UVG': ['Beauty', 'Bosphorus', 'HoneyBee', 'Jockey', 'ReadySteadyGo', 'ShakeNDry', 'YachtRide']}

        uvg_rd = {'psnr': [], 'rate': []}
        hevc_rd = {'psnr': [], 'rate': []}

        single_seq_psnr = {}
        single_seq_rate = {}
        single_seq_log = {}
        single_seq_gop = {}

        for logs in [log['test_log'] for log in outputs]:
            seq_name = logs['seq_name']

            if seq_name in dataset_name['UVG']:
                uvg_rd['psnr'].append(logs['test_psnr'])
                uvg_rd['rate'].append(logs['test_rate'])
            elif seq_name in dataset_name['HEVC']:
                hevc_rd['psnr'].append(logs['test_psnr'])
                hevc_rd['rate'].append(logs['test_rate'])
            else:
                print("Unexpected sequence name:", seq_name)
                raise NotImplementedError

            if seq_name not in list(single_seq_psnr.keys()):
                single_seq_psnr[seq_name] = []
                single_seq_rate[seq_name] = []
                single_seq_log[seq_name] = []

            single_seq_gop[seq_name] = len(logs['log_list'])
            single_seq_log[seq_name].extend(logs['log_list'])
            single_seq_psnr[seq_name].append(logs['test_psnr'])
            single_seq_rate[seq_name].append(logs['test_rate'])

        # os.makedirs(self.hparams.save_dir + f'/report', exist_ok=True)
        #
        # for seq_name, log_list in single_seq_log.items():
        #     with open(self.hparams.save_dir + f'/report/{seq_name}.csv', 'w', newline='') as report:
        #         writer = csv.writer(report, delimiter=',')
        #
        #         writer.writerow(['frame', 'PSNR', 'total bits'])
        #
        #         for idx in range(len(log_list)):
        #             writer.writerow([f'frame_{idx+1}', log_list[idx]['psnr'], log_list[idx]['rate']])

        logs = {'test/UVG psnr': np.mean(uvg_rd['psnr']), 'test/UVG rate': np.mean(uvg_rd['rate']),
                'test/HEVC-B psnr': np.mean(hevc_rd['psnr']), 'test/HEVC-B rate': np.mean(hevc_rd['rate'])}

        for seq_name in single_seq_psnr.keys():
            print('{}, \t{:.4f}, {:.4f}'.format(seq_name,
                                                np.mean(single_seq_psnr[seq_name]), np.mean(single_seq_rate[seq_name])))
        print('============================')
        print('UVG,    {:.4f}, {:.4f}'.format(np.mean(uvg_rd['psnr']), np.mean(uvg_rd['rate'])))
        print('HEVC-B, {:.4f}, {:.4f}'.format(np.mean(hevc_rd['psnr']), np.mean(hevc_rd['rate'])))

        self.log_dict(logs)

        return None

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers

        optimizer = optim.Adam([dict(params=self.main_parameters(), lr=self.hparams.lr),
                                dict(params=self.aux_parameters(), lr=self.hparams.lr * 10)])
        scheduler = {
             'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.3, 5, verbose=True),
             'monitor': 'val/loss',
             'strict': True,
        }

        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, on_tpu=None,
                       using_native_amp=None, using_lbfgs=None):

        def clip_gradient(opt, grad_clip):
            for group in opt.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)

        clip_gradient(optimizer, 5)

        optimizer.step()
        optimizer.zero_grad()

    def compress(self, ref_frame, coding_frame, p_order):
        flow = self.MENet(ref_frame, coding_frame)

        flow_hat, mv_strings, mv_shape = self.Motion.compress(flow, return_hat=True, p_order=p_order)

        strings, shapes = [mv_strings], [mv_shape]

        mc_frame = self.mc_net_forward(ref_frame, flow_hat)

        predicted = mc_frame

        res = coding_frame - predicted
        res_hat, res_strings, res_shape = self.Residual.compress(res, return_hat=True)
        reconstructed = predicted + res_hat
        strings.append(res_strings)
        shapes.append(res_shape)

        return reconstructed, strings, shapes

    def decompress(self, ref_frame, strings, shapes, p_order):
        # TODO: Modify to make AR function work
        mv_strings = strings[0]
        mv_shape = shapes[0]

        flow_hat = self.Motion.decompress(mv_strings, mv_shape, p_order=p_order)

        mc_frame = self.mc_net_forward(ref_frame, flow_hat)

        predicted = mc_frame
        res_strings, res_shape = strings[1], shapes[1]

        res_hat = self.Residual.decompress(res_strings, res_shape)
        reconstructed = predicted + res_hat

        return reconstructed

    def setup(self, stage):
        self.logger.experiment.log_parameters(self.hparams)

        dataset_root = os.getenv('DATASET')
        qp = {256: 37, 512: 32, 1024: 27, 2048: 22}[self.hparams.lmda]

        if stage == 'fit':
            transformer = transforms.Compose([
                transforms.RandomCrop((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])

            self.train_dataset = VideoDataIframe(dataset_root + "vimeo_septuplet/", 'BPG_QP' + str(qp), 7,
                                                 transform=transformer)
            self.val_dataset = VideoTestDataIframe(dataset_root + "TestVideo", self.hparams.lmda, first_gop=True)

        elif stage == 'test':
            self.test_dataset = VideoTestDataIframe(dataset_root + "TestVideo", self.hparams.lmda, first_gop=True)

        else:
            raise NotImplementedError

    def train_dataloader(self):
        # REQUIRED
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.hparams.batch_size,
                                  num_workers=self.hparams.num_workers,
                                  shuffle=True)
        return train_loader

    def val_dataloader(self):
        # OPTIONAL
        val_loader = DataLoader(self.val_dataset,
                                batch_size=1,
                                num_workers=self.hparams.num_workers,
                                shuffle=False)
        return val_loader

    def test_dataloader(self):
        # OPTIONAL
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=1,
                                 num_workers=self.hparams.num_workers,
                                 shuffle=False)
        return test_loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the arguments for this LightningModule
        """
        # MODEL specific
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', '-lr', dest='lr', default=5e-5, type=float)
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--lmda', default=2048, choices=[256, 512, 1024, 2048], type=int)
        parser.add_argument('--patch_size', default=256, type=int)
        parser.add_argument('--ssim', action="store_true")
        parser.add_argument('--debug', action="store_true")

        # training specific (for this model)
        parser.add_argument('--num_workers', default=16, type=int)
        parser.add_argument('--save_dir')

        return parser


if __name__ == '__main__':
    # sets seeds for numpy, torch, etc...
    # must do for DDP to work well
    seed_everything(888888)
    experiment_name = 'PWC'
    project_name = 'Feat-Domain-Warping'

    save_root = os.getenv('LOG', './') + 'torchDVC/'

    parser = argparse.ArgumentParser(add_help=True)

    # add args from trainer
    parser = Trainer.add_argparse_args(parser)

    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = Pframe.add_model_specific_args(parser)

    trc.add_coder_args(parser)

    # training specific
    parser.add_argument('--checkpoint_dir', default=save_root)
    parser.add_argument('--restore', type=str, choices=['none', 'load', 'resume'], default='none')
    parser.add_argument('--test', "-T", action="store_true")
    parser.add_argument('--epoch', type=int, default=-1)
    parser.set_defaults(gpus=1)

    # parse params
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    save_path = os.path.join(save_root, project_name, experiment_name + '-' + str(args.lmda))

    res_coder = trc.get_coder_from_args(args)
    args.architecture = 'GoogleMotionHPCoder'
    motion_coder = trc.get_coder_from_args(args)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        save_last=True,
        verbose=True,
        monitor='val/loss',
        mode='min',
        prefix=''
    )

    checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}_last"

    db = None

    if args.gpus > 1:
        db = 'ddp'

    if args.restore == 'resume':
        prev_exp_key = {
            256: 'e7d31350b9344a72af4aea605fba47ca',
            512: 'e48f3986d06c4a9c8810439c40a79a86',
            1024: 'ca9d64d52d454884af304088d925f477',
            2048: '015dfccdaace49efbe9768e5aed4969b'
        }[args.lmda]

        assert args.epoch >= 0, "Please assign epoch number for resuming."
        args.save_dir = os.path.join(save_root,
                                     'FeatDVC_PSNR_' + str(args.lmda),
                                     experiment_name,
                                     f'eval_{args.epoch}')

        comet_logger = CometLogger(
            api_key="OrUWh5s0QzFPQsNwQoxPXlm8i",
            experiment_key=prev_exp_key,
            project_name=project_name,
            workspace="cp-chang",
            experiment_name=experiment_name + "-" + str(args.lmda),
            disabled=args.test or args.debug
        )

        trainer = Trainer.from_argparse_args(args,
                                             checkpoint_callback=checkpoint_callback,
                                             distributed_backend=db,
                                             logger=comet_logger,
                                             default_root_dir=save_root,
                                             resume_from_checkpoint=os.path.join(save_root, project_name,
                                                                                 prev_exp_key, "checkpoints",
                                                                                 f"epoch={args.epoch}.ckpt"),
                                             check_val_every_n_epoch=1,
                                             num_sanity_val_steps=-1,
                                             terminate_on_nan=True,
                                             )

        checkpoint = torch.load(os.path.join(save_root, project_name, prev_exp_key, "checkpoints",
                                             f"epoch={args.epoch}.ckpt"),
                                map_location=(lambda storage, loc: storage))

        model = Pframe(args, motion_coder, res_coder).cuda()
        model.load_state_dict(checkpoint['state_dict'])

    elif args.restore == 'load':
        comet_logger = CometLogger(
            api_key="OrUWh5s0QzFPQsNwQoxPXlm8i",
            project_name=project_name,
            workspace="cp-chang",
            experiment_name=experiment_name + "-" + str(args.lmda),
            disabled=args.test or args.debug
        )

        iter_num = 400000

        trainer = Trainer.from_argparse_args(args,
                                             checkpoint_callback=checkpoint_callback,
                                             gpus=args.gpus,
                                             distributed_backend=db,
                                             logger=comet_logger,
                                             default_root_dir=save_root,
                                             check_val_every_n_epoch=1,
                                             num_sanity_val_steps=-1,
                                             terminate_on_nan=True,
                                             )

        checkpoint = torch.load(
            os.path.join(save_root, 'FeatDVC_PSNR_' + str(args.lmda), 'ICCV-reproduce', f'model_{iter_num}.ckpt'),
            map_location=(lambda storage, loc: storage))

        model = Pframe(args, motion_coder, res_coder).cuda()
        model.load_state_dict(checkpoint['model'], strict=False)

        trainer.global_step = 4400000
        trainer.current_epoch = phase['trainMC'] + 1

    else:
        comet_logger = CometLogger(
            api_key="OrUWh5s0QzFPQsNwQoxPXlm8i",
            project_name=project_name,
            workspace="cp-chang",
            experiment_name=experiment_name + "-" + str(args.lmda),
            disabled=args.test or args.debug
        )

        iter_num = None
        args.save_dir = os.path.join(save_root,
                                     'FeatDVC_PSNR_' + str(args.lmda),
                                     experiment_name,
                                     f'eval_{20}')

        trainer = Trainer.from_argparse_args(args,
                                             checkpoint_callback=checkpoint_callback,
                                             gpus=args.gpus,
                                             distributed_backend=db,
                                             logger=comet_logger,
                                             default_root_dir=save_root,
                                             check_val_every_n_epoch=1,
                                             num_sanity_val_steps=-1,
                                             terminate_on_nan=True,
                                             )

        model = Pframe(args, motion_coder, res_coder).cuda()

    # comet_logger.experiment.log_code(file_name='torchDVC.py')

    if args.test:
        trainer.test(model)
    else:
        if args.epoch == -1 and iter_num is None:
            trainer.tune(model)
        trainer.fit(model)
