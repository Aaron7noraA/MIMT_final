import argparse
import os
import csv
from functools import partial

import flowiz as fz
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_compression as trc

from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from torch import nn, optim
from torch.utils.data import DataLoader
from torch_compression.modules.entropy_models import EntropyBottleneck
from torchvision import transforms
from torchvision.utils import make_grid

from dataloader import VideoDataIframe, VideoTestDataIframe, VideoTestDataIframeYUV
from flownets import PWCNet
from models import Refinement
from util.psnr import mse2psnr
from util.sampler import Resampler
from util.ssim import MS_SSIM
from util.vision import PlotFlow, save_image, YUV2RGB, RGB2YUV, RGB2YUV420, plot_yuv
from util.functional import interpolate_as, space_to_depth, depth_to_space

# phase = {'init': 0, 'trainMV': 40000, 'trainMC': 100000, 'trainRNN': 200000, 'trainALL': 1000000}
# phase = {'init': 0, 'trainMV': 40000, 'trainMC': 100000, 'trainRNN': 4420000, 'trainALL': 10000000}

# This script should start from residual net training
#phase = {'init': 0, 'trainMV': 10, 'trainMC': 20, 'trainRes_2frames': 60, 'trainRes_fullgop':200, 'trainALL_2frames': 300, 'trainALL_fullgop': 1000}
phase = {'init': 0, 'trainMV': 20, 'trainMC': 60, 'trainRes_2frames': 75, 'trainALL_2frames': 90, 'trainALL_fullgop': 1000}

# this was originally used for rate estimate for each sequence
# these numbers correspond to those avg rate in different rate-points/QPs.
# but now you can simply delete this section.
# it's no longer used in this code.
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
    """Pframe for YUV420"""
    train_dataset: VideoDataIframe
    val_dataset: VideoTestDataIframeYUV
    test_dataset: VideoTestDataIframeYUV

    def __init__(self, args, coder, ar_coder):
        super(Pframe, self).__init__()
        self.args = args
        self.criterion = nn.MSELoss(reduction='none') if not self.args.ssim else MS_SSIM(data_range=1.).cuda()

        self.MENet = PWCNet()

        self.Motion = coder(in_channels=2, out_channels=2, kernel_size=3)

        self.Resampler = Resampler()
        self.MCNet = Refinement(6, 64, out_channels=3) # channel # of Y:U:V=4:1:1 ; 2 inputs

        self.Residual = coder(in_channels=3, out_channels=3)

    def load_args(self, args):
        self.args = args

    def YUVCriterion(self, y, gt):
        y_Y, y_U, y_V = y
        gt_Y, gt_U, gt_V = gt
        distortion_Y = torch.mean(self.criterion(gt_Y, y_Y), dim=(2,3))
        distortion_U = torch.mean(self.criterion(gt_U, y_U), dim=(2,3))
        distortion_V = torch.mean(self.criterion(gt_V, y_V), dim=(2,3))
        #distortion = ((distortion_Y + 1e-9)**(6/8) * (distortion_U + 1e-9)**(1/8) * (distortion_V + 1e-9)**(1/8)).mean()
        distortion = (distortion_Y*(6/8) + distortion_U*(1/8) + distortion_V*(1/8)).mean()
        
        return distortion

    def motion_forward(self, ref_frame, coding_frame):
        ref_frame_RGB = YUV2RGB(ref_frame, up=True)
        coding_frame_RGB = YUV2RGB(coding_frame, up=True)
        
        #ref_frame_Ys    = torch.cat([ref_frame_Y,    ref_frame_Y,    ref_frame_Y]   , dim=1) # Copy Y for 3 times for ME
        #coding_frame_Ys = torch.cat([coding_frame_Y, coding_frame_Y, coding_frame_Y], dim=1) # Copy Y for 3 times for ME
        
        #flow = self.MENet(ref_frame_Ys, coding_frame_Ys)
        flow = self.MENet(ref_frame_RGB, coding_frame_RGB) # ME in RGB domain
        
        flow_hat, likelihood_m = self.Motion(flow)
        mc = self.mc_net_forward(ref_frame, flow_hat)

        return mc, {'m_likelihood': likelihood_m, 'flow_hat': flow_hat}

    def forward(self, ref_frame, coding_frame, p_order=0, res_visual=False, visual_prefix = ''):
        mc, m_info = self.motion_forward(ref_frame, coding_frame)

        coding_frame_Y, coding_frame_U, coding_frame_V = coding_frame
        coding_frame = torch.cat([space_to_depth(coding_frame_Y, block_size=2), coding_frame_U, coding_frame_V], dim=1)

        reconstructed, likelihood_r, mc_hat, _, _, _ = self.Residual(coding_frame, output=mc, cond_coupling_input=mc,
                                                                       visual=res_visual, figname=visual_prefix)

        return reconstructed, m_info['m_likelihood'], likelihood_r, m_info['flow_hat'], mc, mc_hat


    def training_step(self, batch, batch_idx):
        epoch = self.current_epoch

        batch = batch.cuda()
        ref_frame = batch[:, 0]

        ref_frame_Y, ref_frame_U, ref_frame_V = RGB2YUV(ref_frame, down=True)

        if epoch <= phase['trainMV']:
            _phase = 'MV'
            coding_frame = batch[:, 1]
            coding_frame_Y, coding_frame_U, coding_frame_V = RGB2YUV(coding_frame, down=True)

            #ref_frame_Ys    = torch.cat([ref_frame_Y, ref_frame_Y, ref_frame_Y]         , dim=1) # Copy Y for 3 times for ME
            #coding_frame_Ys = torch.cat([coding_frame_Y, coding_frame_Y, coding_frame_Y], dim=1) # Copy Y for 3 times for ME

            flow = self.MENet(ref_frame, coding_frame) # ME in RGB domain

            flow_hat, likelihood_m = self.Motion(flow)

            flow_hat_Y = flow_hat
            reconstructed_Y = self.Resampler(ref_frame_Y, flow_hat_Y)

            flow_hat_UV = F.interpolate(flow_hat, scale_factor=0.5, mode='bilinear', align_corners=False)
            #flow_hat_UV = flow_hat_UV.detach() ####################
            reconstructed_U = self.Resampler(ref_frame_U, flow_hat_UV / 2) # Divide by 2 since resolution is downed
            reconstructed_V = self.Resampler(ref_frame_V, flow_hat_UV / 2) # Divide by 2 since resolution is downed

            likelihoods = likelihood_m

            reconstructed_YUV = (reconstructed_Y, reconstructed_U, reconstructed_V)
            coding_frame_YUV = (coding_frame_Y, coding_frame_U, coding_frame_V)
            distortion = self.YUVCriterion(reconstructed_YUV, coding_frame_YUV)

            rate = trc.estimate_bpp(likelihoods, input=coding_frame_Y)

            loss = self.args.lmda * distortion.mean() + rate.mean()
            logs = {'train/loss': loss.item(),
                    'train/distortion': distortion.mean().item(),
                    'train/rate': rate.mean().item()}

        elif epoch <= phase['trainMC']:
            _phase = 'MC'
            coding_frame = batch[:, 1]
            coding_frame_Y, coding_frame_U, coding_frame_V = RGB2YUV(coding_frame, down=True)

            #ref_frame_Ys    = torch.cat([ref_frame_Y, ref_frame_Y, ref_frame_Y]         , dim=1) # Copy Y for 3 times for ME
            #coding_frame_Ys = torch.cat([coding_frame_Y, coding_frame_Y, coding_frame_Y], dim=1) # Copy Y for 3 times for ME

            flow = self.MENet(ref_frame, coding_frame) # ME in RGB domain

            flow_hat, likelihood_m = self.Motion(flow)

            mc = self.mc_net_forward((ref_frame_Y, ref_frame_U, ref_frame_V), flow_hat)

            reconstructed_Y, reconstructed_U, reconstructed_V = depth_to_space(mc[:, :4], block_size=2), mc[:, 4:5], mc[:, 5:]
            
            likelihoods = likelihood_m

            reconstructed_YUV = (reconstructed_Y, reconstructed_U, reconstructed_V)
            coding_frame_YUV = (coding_frame_Y, coding_frame_U, coding_frame_V)
            distortion = self.YUVCriterion(reconstructed_YUV, coding_frame_YUV)
            
            rate = trc.estimate_bpp(likelihoods, input=coding_frame_Y)

            loss = self.args.lmda * distortion.mean() + rate.mean()
            logs = {'train/loss': loss.item(),
                    'train/distortion': distortion.mean().item(),
                    'train/rate': rate.mean().item()}
        else:
            if epoch <= phase['trainRes_2frames']:
                _phase = 'RES'
            else:
                _phase = 'ALL'
            reconstructed_Y, reconstructed_U, reconstructed_V = ref_frame_Y, ref_frame_U, ref_frame_V

            loss = torch.tensor(0., dtype=torch.float, device=ref_frame.device)
            dist_list = []
            rate_list = []
            mc_error_list = []
            warping_loss_list = []

            for frame_idx in range(1, 7):
                ref_frame_Y, ref_frame_U, ref_frame_V = reconstructed_Y, reconstructed_U, reconstructed_V

                if frame_idx > 1 and epoch <= phase['trainALL_2frames']: # 2-frame training stages
                        break
                
                ref_frame_Y, ref_frame_U, ref_frame_V = ref_frame_Y.detach(), ref_frame_U.detach(), ref_frame_V.detach() # Detach when RNN training

                coding_frame = batch[:, frame_idx]
                coding_frame_Y, coding_frame_U, coding_frame_V = RGB2YUV(coding_frame, down=True)

                #ref_frame_Ys    = torch.cat([ref_frame_Y,    ref_frame_Y,    ref_frame_Y]   , dim=1) # Copy Y for 3 times for ME
                #coding_frame_Ys = torch.cat([coding_frame_Y, coding_frame_Y, coding_frame_Y], dim=1) # Copy Y for 3 times for ME

                if _phase == 'RES': # Train res_coder only

                    with torch.no_grad():
                        flow = self.MENet(ref_frame, coding_frame) # ME in RGB domain
                        flow_hat, likelihood_m = self.Motion(flow)
                        mc = self.mc_net_forward((ref_frame_Y, ref_frame_U, ref_frame_V), flow_hat)
                        reconstructed_Y, reconstructed_U, reconstructed_V = depth_to_space(mc[:, :4], block_size=2), mc[:, 4:5], mc[:, 5:]

                        reconstructed_YUV = (reconstructed_Y.detach(), reconstructed_U.detach(), reconstructed_V.detach())
                        coding_frame_YUV = (coding_frame_Y, coding_frame_U, coding_frame_V)
                        warping_loss = self.YUVCriterion(reconstructed_YUV, coding_frame_YUV)
                        
                else:
                    flow = self.MENet(ref_frame, coding_frame) # ME in RGB domain
                    flow_hat, likelihood_m = self.Motion(flow)
                    mc = self.mc_net_forward((ref_frame_Y, ref_frame_U, ref_frame_V), flow_hat)
                    reconstructed_Y, reconstructed_U, reconstructed_V = depth_to_space(mc[:, :4], block_size=2), mc[:, 4:5], mc[:, 5:]

                    reconstructed_YUV = (reconstructed_Y.detach(), reconstructed_U.detach(), reconstructed_V.detach())
                    coding_frame_YUV = (coding_frame_Y, coding_frame_U, coding_frame_V)
                    warping_loss = self.YUVCriterion(reconstructed_YUV, coding_frame_YUV)
                
                coding_frame = torch.cat([space_to_depth(coding_frame_Y, block_size=2), coding_frame_U, coding_frame_V], dim=1)
                reconstructed, likelihood_r, mc_hat, _, _, _ = self.Residual(coding_frame, output=mc, cond_coupling_input=mc)
                
                reconstructed_Y, reconstructed_U, reconstructed_V = depth_to_space(reconstructed[:, :4], block_size=2), reconstructed[:, 4:5], reconstructed[:, 5:]

                #likelihoods = likelihood_m + likelihood_r # Their shapes are different
                
                reconstructed_YUV = (reconstructed_Y, reconstructed_U, reconstructed_V)
                coding_frame_YUV = (coding_frame_Y, coding_frame_U, coding_frame_V)
                distortion = self.YUVCriterion(reconstructed_YUV, coding_frame_YUV)

                rate = trc.estimate_bpp(likelihood_m, input=coding_frame_Y) + trc.estimate_bpp(likelihood_r, input=coding_frame)
                mc_error = nn.MSELoss(reduction='none')(mc, mc_hat)

                loss += self.args.lmda * distortion.mean() + rate.mean() + 0.01 * self.args.lmda * mc_error.mean()
                dist_list.append(distortion.mean())
                warping_loss_list.append(warping_loss.mean())
                rate_list.append(rate.mean())
                mc_error_list.append(mc_error.mean())


            distortion = torch.mean(torch.tensor(dist_list))
            warping_loss = torch.mean(torch.tensor(warping_loss_list))
            rate = torch.mean(torch.tensor(rate_list))
            mc_error = torch.mean(torch.tensor(mc_error_list))
            logs = {
                    'train/loss': loss.item() if epoch <= phase['trainALL_2frames'] else loss.item() / 6, # divide by 6 when RNN training stages
                    'train/distortion': distortion.item(), 
                    'train/PSNR': mse2psnr(distortion.item()), 
                    'train/rate': rate.item(), 
                    'train/mc_error': mc_error.item(),
                    'train/warping_loss': warping_loss.item()
                   }

        # if epoch <= phase['trainMC']:
        #     auxloss = self.Motion.aux_loss()
        # else:
        #     auxloss = self.aux_loss()
        #
        # logs['train/aux_loss'] = auxloss.item()
        #
        # loss = loss + auxloss

        self.log_dict(logs)

        return loss if epoch <= phase['trainALL_2frames'] else loss / 6

    def validation_step(self, batch, batch_idx):
        def get_psnr(mse):
            if mse > 0:
                psnr = 10 * (torch.log(1 * 1 / mse) / np.log(10))
            else:
                psnr = mse + 100
            return psnr

        def create_grid(img):
            return make_grid(torch.unsqueeze(img, 1)).cpu().detach().numpy()[0]

        def upload_img(tnsr, tnsr_name, ch="first", grid=True):
            if grid:
                tnsr = create_grid(tnsr)

            self.logger.experiment.log_image(tnsr, name=tnsr_name, step=self.current_epoch,
                                             image_channels=ch, overwrite=False)

        seq_name, batch_Y, batch_U, batch_V, frame_id_start = batch

        ref_frame_Y = batch_Y[:, 0] # Put reference frame in first position
        ref_frame_U = batch_U[:, 0]
        ref_frame_V = batch_V[:, 0]

        batch_Y = batch_Y[:, 1:] # GT
        batch_U = batch_U[:, 1:]
        batch_V = batch_V[:, 1:]
        #ref_frame = batch[:, 0]
        #batch = batch[:, 1:]
        seq_name = seq_name[0]

        gop_size = batch_Y.size(1)

        height, width = ref_frame_Y.size()[2:]

        psnr_list = []
        mc_psnr_list = []
        mse_list = []
        rate_list = []
        m_rate_list = []
        loss_list = []
        align = trc.util.Alignment()
        align_UV = trc.util.Alignment(divisor=32)

        #reconstructed = ref_frame
        #coding_frame = batch[:, 0]
        #reconstructed_Y, reconstructed_U, reconstructed_V = RGB2YUV(ref_frame, down=True)
        #coding_frame_Y, coding_frame_U, coding_frame_V = RGB2YUV(coding_frame, down=True)
        reconstructed_Y, reconstructed_U, reconstructed_V = ref_frame_Y, ref_frame_U, ref_frame_V
        coding_frame_Y, coding_frame_U, coding_frame_V = batch_Y[:, 0], batch_U[:, 0], batch_V[:, 0]

        for frame_idx in range(gop_size):
            if frame_idx != 0:
                ref_frame_Y, ref_frame_U, ref_frame_V = (
                                                            align.align(reconstructed_Y), 
                                                            align_UV.align(reconstructed_U), 
                                                            align_UV.align(reconstructed_V), 
                                                        )

                coding_frame_Y, coding_frame_U, coding_frame_V = (
                                                                    align.align(batch_Y[:, frame_idx]), 
                                                                    align_UV.align(batch_U[:, frame_idx]), 
                                                                    align_UV.align(batch_V[:, frame_idx]), 
                                                                 )
                #ref_frame = align.align(reconstructed)
                #ref_frame_Y, ref_frame_U, ref_frame_V = RGB2YUV(ref_frame, down=True)

                #coding_frame = align.align(batch[:, frame_idx])
                #coding_frame_Y, coding_frame_U, coding_frame_V = RGB2YUV(coding_frame, down=True)

                reconstructed, likelihood_m, likelihood_r, flow_hat, mc_frame, mc_hat = self((ref_frame_Y, ref_frame_U, ref_frame_V),
                                                                                             (coding_frame_Y, coding_frame_U, coding_frame_V),
                                                                                             p_order=frame_idx)

                reconstructed = (depth_to_space(reconstructed[:, :4], block_size=2), reconstructed[:, 4:5], reconstructed[:, 5:])
                reconstructed_Y, reconstructed_U, reconstructed_V = reconstructed
                reconstructed = YUV2RGB(reconstructed, up=True)
                
                mc_frame = (depth_to_space(mc_frame[:, :4], block_size=2), mc_frame[:, 4:5], mc_frame[:, 5:])
                mc_frame_Y, mc_frame_U, mc_frame_V = mc_frame
                mc_frame = YUV2RGB(mc_frame, up=True)
                
                mc_hat = (depth_to_space(mc_hat[:, :4], block_size=2), mc_hat[:, 4:5], mc_hat[:, 5:])
                mc_hat_Y, mc_hat_U, mc_hat_V = mc_hat
                mc_hat = YUV2RGB(mc_hat, up=True)

                ref_frame = YUV2RGB((ref_frame_Y, ref_frame_U, ref_frame_V), up=True)
                ref_frame = align.resume(ref_frame).clamp(0, 1)

                reconstructed_Y, reconstructed_U, reconstructed_V = (
                                                                     align.resume(reconstructed_Y).clamp(0, 1), 
                                                                     align_UV.resume(reconstructed_U).clamp(0, 1), 
                                                                     align_UV.resume(reconstructed_V).clamp(0, 1)
                                                                    )
                reconstructed = align.resume(reconstructed).clamp(0, 1)

                coding_frame_Y, coding_frame_U, coding_frame_V = (
                                                                  align.resume(coding_frame_Y).clamp(0, 1), 
                                                                  align_UV.resume(coding_frame_U).clamp(0, 1), 
                                                                  align_UV.resume(coding_frame_V).clamp(0, 1)
                                                                 )
                coding_frame = YUV2RGB((coding_frame_Y, coding_frame_U, coding_frame_V), up=True)
                coding_frame = align.resume(coding_frame).clamp(0, 1)

                mc_frame_Y, mc_frame_U, mc_frame_V = (
                                                      align.resume(mc_frame_Y).clamp(0, 1), 
                                                      align_UV.resume(mc_frame_U).clamp(0, 1), 
                                                      align_UV.resume(mc_frame_V).clamp(0, 1)
                                                     )
                
                mc_frame = align.resume(mc_frame).clamp(0, 1)

                mc_hat_Y, mc_hat_U, mc_hat_V = (
                                                align.resume(mc_hat_Y).clamp(0, 1), 
                                                align_UV.resume(mc_hat_U).clamp(0, 1), 
                                                align_UV.resume(mc_hat_V).clamp(0, 1)
                                               )
                mc_hat = align.resume(mc_hat).clamp(0, 1)

                rate = trc.estimate_bpp(likelihood_m, input=ref_frame_Y).item() + \
                       trc.estimate_bpp(likelihood_r, input=ref_frame).item()
                m_rate = trc.estimate_bpp(likelihood_m, input=ref_frame_Y).item()
                

                reconstructed_YUV = (reconstructed_Y, reconstructed_U, reconstructed_V)
                coding_frame_YUV = (coding_frame_Y, coding_frame_U, coding_frame_V)
                #print(reconstructed_Y.shape, reconstructed_U.shape, reconstructed_V.shape, ' ; ',coding_frame_Y.shape, coding_frame_U.shape, coding_frame_V.shape)
                mse = self.YUVCriterion(reconstructed_YUV, coding_frame_YUV)
                psnr = 1/8 * (
                                6 * mse2psnr(torch.mean(self.criterion(reconstructed_Y, coding_frame_Y), dim=(2,3))) + \
                                1 * mse2psnr(torch.mean(self.criterion(reconstructed_U, coding_frame_U), dim=(2,3))) + \
                                1 * mse2psnr(torch.mean(self.criterion(reconstructed_V, coding_frame_V), dim=(2,3)))
                             )
                
                mc_frame_YUV = (mc_frame_Y, mc_frame_U, mc_frame_V)
                mc_mse = self.YUVCriterion(mc_frame_YUV, coding_frame_YUV)
                mc_psnr = 1/8 * (
                                6 * mse2psnr(torch.mean(self.criterion(mc_frame_Y, coding_frame_Y), dim=(2,3))) + \
                                1 * mse2psnr(torch.mean(self.criterion(mc_frame_U, coding_frame_U), dim=(2,3))) + \
                                1 * mse2psnr(torch.mean(self.criterion(mc_frame_V, coding_frame_V), dim=(2,3)))
                             )

                mc_error = nn.MSELoss(reduction='none')(mc_frame, mc_hat)

                if frame_idx == 1:
                    #flow_map = plot_flow(flow_hat.to('cuda:0'))
                    flow_rgb = torch.from_numpy(
                                    fz.convert_from_flow(flow_hat[0].permute(1, 2, 0).cpu().numpy()) / 255).permute(2, 0, 1)
                    upload_img(flow_rgb.cpu().numpy(), f'{seq_name}_flow_map.png', grid=False)
                    upload_img(ref_frame.cpu().numpy()[0], f'{seq_name}_ref_frame.png', grid=False)
                    upload_img(coding_frame.cpu().numpy()[0], f'{seq_name}_gt_frame.png', grid=False)
                    upload_img(mc_frame.cpu().numpy()[0], f'{seq_name}_MC_frame.png', grid=False)
                    upload_img(reconstructed.cpu().numpy()[0],
                               seq_name + '_reconstructed_{:.3f}.png'.format(mc_psnr),
                               grid=False)
                    upload_img(mc_hat.cpu().numpy()[0], f'{seq_name}_predicted_MC_frame.png', grid=False)

                loss = self.args.lmda * mse + rate + 0.01 * self.args.lmda * mc_error.mean()

                mc_psnr_list.append(mc_psnr)
                m_rate_list.append(m_rate)

            else:
                intra_index = {256: 3, 512: 2, 1024: 1, 2048: 0}[self.args.lmda]

                rate = iframe_byte[seq_name][intra_index] * 8 / height / width

                reconstructed_YUV = (reconstructed_Y, reconstructed_U, reconstructed_V)
                coding_frame_YUV = (coding_frame_Y, coding_frame_U, coding_frame_V)
                mse = self.YUVCriterion(reconstructed_YUV, coding_frame_YUV)
                psnr = 1/8 * (
                                6 * mse2psnr(torch.mean(self.criterion(reconstructed_Y, coding_frame_Y), dim=(2,3))) + \
                                1 * mse2psnr(torch.mean(self.criterion(reconstructed_U, coding_frame_U), dim=(2,3))) + \
                                1 * mse2psnr(torch.mean(self.criterion(reconstructed_V, coding_frame_V), dim=(2,3)))
                             )
                #psnr = mse2psnr(mse)

                loss = self.args.lmda * mse + rate

            mse = mse.item()
            loss = loss.item()

            psnr_list.append(psnr)
            rate_list.append(rate)
            mse_list.append(mse)
            loss_list.append(loss)

        psnr = np.mean(psnr_list)
        mc_psnr = np.mean(mc_psnr_list)
        rate = np.mean(rate_list)
        m_rate = np.mean(m_rate_list)
        mse = np.mean(mse_list)
        loss = np.mean(loss_list)

        logs = {'seq_name': seq_name, 'val_loss': loss, 'val_mse': mse, 'val_psnr': psnr, 'val_rate': rate, 
                'val_mc_psnr': mc_psnr, 'val_m_rate': m_rate}
        
        return {'val_log': logs}

    def validation_epoch_end(self, outputs):
        dataset_name = {'HEVC': ['BasketballDrive', 'BQTerrace', 'Cactus', 'Kimono1', 'ParkScene'],
                        'UVG': ['Beauty', 'Bosphorus', 'HoneyBee', 'Jockey', 'ReadySteadyGo', 'ShakeNDry', 'YachtRide']}

        uvg_rd = {'psnr': [], 'rate': [], 'mc_psnr': [], 'm_rate': []}
        hevc_rd = {'psnr': [], 'rate': [], 'mc_psnr': [], 'm_rate': []}
        loss = []

        for logs in [log['val_log'] for log in outputs]:
            seq_name = logs['seq_name']

            if seq_name in dataset_name['UVG']:
                uvg_rd['psnr'].append(logs['val_psnr'])
                uvg_rd['rate'].append(logs['val_rate'])
                uvg_rd['mc_psnr'].append(logs['val_mc_psnr'])
                uvg_rd['m_rate'].append(logs['val_m_rate'])
            elif seq_name in dataset_name['HEVC']:
                hevc_rd['psnr'].append(logs['val_psnr'])
                hevc_rd['rate'].append(logs['val_rate'])
                hevc_rd['mc_psnr'].append(logs['val_mc_psnr'])
                hevc_rd['m_rate'].append(logs['val_m_rate'])
            else:
                print("Unexpected sequence name:", seq_name)
                raise NotImplementedError

            loss.append(logs['val_loss'])

        avg_loss = np.mean(loss)

        logs = {'val/loss': avg_loss,
                'val/UVG psnr': np.mean(uvg_rd['psnr']), 'val/UVG rate': np.mean(uvg_rd['rate']),
                'val/UVG mc_psnr': np.mean(uvg_rd['mc_psnr']), 'val/UVG m_rate': np.mean(uvg_rd['m_rate']),
                'val/HEVC-B psnr': np.mean(hevc_rd['psnr']), 'val/HEVC-B rate': np.mean(hevc_rd['rate']),
                'val/HEVC-B mc_psnr': np.mean(hevc_rd['mc_psnr']), 'val/HEVC-B m_rate': np.mean(hevc_rd['m_rate']),
                }
        self.log_dict(logs)

        return None

    def test_step(self, batch, batch_idx):
        metrics_name = ['PSNR', 'Rate', 'Mo_Rate', 'MC-PSNR', 'MCerr-PSNR', 'PSNR-Y', 'PSNR-U', 'PSNR-V', 'MC-PSNR-Y', 'MC-PSNR-U', 'MC-PSNR-V']
        metrics = {}
        for m in metrics_name:
            metrics[m] = []
        # PSNR: PSNR(gt, ADQ)
        # Rate
        # Mo_Rate: Motion Rate
        # MC-PSNR: PSNR(gt, mc_frame)
        # MCerr-PSNR: PSNR(x_2, mc_frame)
        # PSNR-Y: PSNR(gt_Y, ADQ_Y)
        # PSNR-U: PSNR(gt_U, ADQ_U)
        # PSNR-V: PSNR(gt_V, ADQ_V)
        # MC-PSNR-Y: PSNR(gt_Y, mc_frame_Y)
        # MC-PSNR-U: PSNR(gt_U, mc_frame_U)
        # Mc-PSNR-V: PSNR(gt_V, mc_frame_V)
                        
        seq_name, batch_Y, batch_U, batch_V, frame_id_start = batch
        frame_id = int(frame_id_start)

        ref_frame_Y = batch_Y[:, 0] # Put reference frame in first position
        ref_frame_U = batch_U[:, 0]
        ref_frame_V = batch_V[:, 0]

        batch_Y = batch_Y[:, 1:] # GT
        batch_U = batch_U[:, 1:]
        batch_V = batch_V[:, 1:]
        seq_name = seq_name[0]

        gop_size = batch_Y.size(1)
        height, width = ref_frame_Y.size()[2:]
        estimate_bpp = partial(trc.estimate_bpp, num_pixels=height * width)


        psnr_list = []
        mc_psnr_list = []
        mc_hat_psnr_list = []
        BDQ_psnr_list = []
        rate_list = []
        m_rate_list = []
        log_list = []
        align = trc.util.Alignment()
        align_UV = trc.util.Alignment(divisor=32)

        reconstructed_Y, reconstructed_U, reconstructed_V = ref_frame_Y, ref_frame_U, ref_frame_V
        coding_frame_Y, coding_frame_U, coding_frame_V = batch_Y[:, 0], batch_U[:, 0], batch_V[:, 0]

        for frame_idx in range(gop_size):
            if frame_idx != 0:
                ref_frame_Y, ref_frame_U, ref_frame_V = (
                                                            align.align(reconstructed_Y), 
                                                            align_UV.align(reconstructed_U), 
                                                            align_UV.align(reconstructed_V) 
                                                        )
                coding_frame_Y, coding_frame_U, coding_frame_V = (
                                                                    align.align(batch_Y[:, frame_idx]), 
                                                                    align_UV.align(batch_U[:, frame_idx]), 
                                                                    align_UV.align(batch_V[:, frame_idx])
                                                                 )
                coding_frame_yuv = torch.cat([space_to_depth(coding_frame_Y, 2), coding_frame_U, coding_frame_V], dim=1)

                # reconstruced frame will be next ref_frame
                #if batch_idx % 100 == 0:
                if False:
                    os.makedirs(os.path.join(self.args.save_dir, 'visualize_ANFIC', f'batch_{batch_idx}'), exist_ok=True)
                    reconstructed, likelihood_m, likelihood_r, flow_hat, mc_frame, mc_hat = self((ref_frame_Y, ref_frame_U, ref_frame_V),
                                                                                                 (coding_frame_Y, coding_frame_U, coding_frame_V),
                                                                                                 p_order=frame_idx,
                                                                                                 res_visual=True,
                                                                                                 visual_prefix=os.path.join(
                                                                                                                    self.args.save_dir, 
                                                                                                                    'visualize_ANFIC',
                                                                                                                    f'batch_{batch_idx}',
                                                                                                                    f'frame_{frame_idx}',
                                                                                                                )
                                                                                               )
                else:
                    reconstructed, likelihood_m, likelihood_r, flow_hat, mc_frame, mc_hat = self((ref_frame_Y, ref_frame_U, ref_frame_V),
                                                                                                 (coding_frame_Y, coding_frame_U, coding_frame_V),
                                                                                                 p_order=frame_idx)
                reconstructed_yuv = reconstructed
                reconstructed = (depth_to_space(reconstructed[:, :4], block_size=2), reconstructed[:, 4:5], reconstructed[:, 5:])
                reconstructed_Y, reconstructed_U, reconstructed_V = reconstructed
                reconstructed = YUV2RGB(reconstructed, up=True)
                
                mc_frame_yuv = mc_frame
                mc_frame = (depth_to_space(mc_frame[:, :4], block_size=2), mc_frame[:, 4:5], mc_frame[:, 5:])
                mc_frame_Y, mc_frame_U, mc_frame_V  = mc_frame
                mc_frame = YUV2RGB(mc_frame, up=True)
                
                mc_hat = (depth_to_space(mc_hat[:, :4], block_size=2), mc_hat[:, 4:5], mc_hat[:, 5:])
                mc_hat_Y, mc_hat_U, mc_hat_V  = mc_hat
                mc_hat = YUV2RGB(mc_hat, up=True)

                ref_frame = YUV2RGB((ref_frame_Y, ref_frame_U, ref_frame_V), up=True)

                reconstructed_Y, reconstructed_U, reconstructed_V = (
                                                                     reconstructed_Y.clamp(0, 1), 
                                                                     reconstructed_U.clamp(0, 1), 
                                                                     reconstructed_V.clamp(0, 1)
                                                                    )
                reconstructed = reconstructed.clamp(0, 1)

                coding_frame_Y, coding_frame_U, coding_frame_V = (
                                                                  coding_frame_Y, 
                                                                  coding_frame_U, 
                                                                  coding_frame_V
                                                                 )
                coding_frame = YUV2RGB((coding_frame_Y, coding_frame_U, coding_frame_V), up=True)

                mc_frame_Y, mc_frame_U, mc_frame_V = (
                                                      mc_frame_Y.clamp(0, 1), 
                                                      mc_frame_U.clamp(0, 1), 
                                                      mc_frame_V.clamp(0, 1)
                                                     )
                
                mc_frame = mc_frame.clamp(0, 1)

                mc_hat_Y, mc_hat_U, mc_hat_V = (
                                                mc_hat_Y.clamp(0, 1), 
                                                mc_hat_U.clamp(0, 1), 
                                                mc_hat_V.clamp(0, 1)
                                               )
                mc_hat = mc_hat.clamp(0, 1)

                os.makedirs(self.args.save_dir + f'/{seq_name}/flow', exist_ok=True)
                os.makedirs(self.args.save_dir + f'/{seq_name}/gt_frame', exist_ok=True)
                os.makedirs(self.args.save_dir + f'/{seq_name}/mc_frame', exist_ok=True)
                os.makedirs(self.args.save_dir + f'/{seq_name}/rec_frame', exist_ok=True)
                os.makedirs(self.args.save_dir + f'/{seq_name}/mc_hat', exist_ok=True)
                #os.makedirs(self.args.save_dir + f'/{seq_name}/gt_frame_yuv', exist_ok=True)
                #os.makedirs(self.args.save_dir + f'/{seq_name}/rec_frame_yuv', exist_ok=True)
                #os.makedirs(self.args.save_dir + f'/{seq_name}/mc_frame_yuv', exist_ok=True)

                if frame_id_start < 8 and seq_name in ['BasketballDrive', 'Kimono1', 'HoneyBee', 'Jockey']:
                    flow_map = plot_flow(flow_hat)
                    save_image(flow_map, self.args.save_dir + f'/{seq_name}/flow/f{int(frame_idx)}_flow.png', nrow=1)

                    save_image(coding_frame[0], self.args.save_dir + f'/{seq_name}/gt_frame/'
                               f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(mc_frame[0], self.args.save_dir + f'/{seq_name}/mc_frame/'
                               f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(reconstructed[0], self.args.save_dir + f'/{seq_name}/rec_frame/'
                               f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(mc_hat[0], self.args.save_dir + f'/{seq_name}/mc_hat/'
                               f'frame_{int(frame_id_start + frame_idx)}.png')

                    #save_image(plot_yuv(coding_frame_yuv[0]), self.args.save_dir + f'/{seq_name}/gt_frame_yuv/'
                    #           f'frame_{int(frame_id_start + frame_idx)}.png')
                    #save_image(plot_yuv(mc_frame_yuv[0]), self.args.save_dir + f'/{seq_name}/mc_frame_yuv/'
                    #           f'frame_{int(frame_id_start + frame_idx)}.png')
                    #save_image(plot_yuv(reconstructed_yuv[0]), self.args.save_dir + f'/{seq_name}/rec_frame_yuv/'
                    #           f'frame_{int(frame_id_start + frame_idx)}.png')
                
                rate = trc.estimate_bpp(likelihood_m, input=reconstructed_Y).item() + trc.estimate_bpp(likelihood_r, input=reconstructed).item()
                m_rate = trc.estimate_bpp(likelihood_m, input=reconstructed_Y).item()
                metrics['Mo_Rate'].append(m_rate)
                 
                psnr_Y = mse2psnr(torch.mean(self.criterion(reconstructed_Y, coding_frame_Y), dim=(2,3)))
                psnr_U = mse2psnr(torch.mean(self.criterion(reconstructed_U, coding_frame_U), dim=(2,3)))
                psnr_V = mse2psnr(torch.mean(self.criterion(reconstructed_V, coding_frame_V), dim=(2,3)))

                metrics['PSNR-Y'].append(psnr_Y)
                metrics['PSNR-U'].append(psnr_U)
                metrics['PSNR-V'].append(psnr_V)

                psnr = 1/8 * (6 * psnr_Y + psnr_U + psnr_V)
                
                mc_psnr_Y = mse2psnr(torch.mean(self.criterion(mc_frame_Y, coding_frame_Y), dim=(2,3)))
                mc_psnr_U = mse2psnr(torch.mean(self.criterion(mc_frame_U, coding_frame_U), dim=(2,3)))
                mc_psnr_V = mse2psnr(torch.mean(self.criterion(mc_frame_V, coding_frame_V), dim=(2,3)))
                
                metrics['MC-PSNR-Y'].append(mc_psnr_Y)
                metrics['MC-PSNR-U'].append(mc_psnr_U)
                metrics['MC-PSNR-V'].append(mc_psnr_V)

                mc_psnr = 1/8 * (6 * mc_psnr_Y + mc_psnr_U + mc_psnr_V)
                metrics['MC-PSNR'].append(mc_psnr)
                
                
                mc_err_psnr = 1/8 * (
                                6 * mse2psnr(torch.mean(self.criterion(mc_frame_Y, mc_hat_Y), dim=(2,3))) + \
                                1 * mse2psnr(torch.mean(self.criterion(mc_frame_U, mc_hat_U), dim=(2,3))) + \
                                1 * mse2psnr(torch.mean(self.criterion(mc_frame_V, mc_hat_V), dim=(2,3)))
                             )
                metrics['MCerr-PSNR'].append(mc_err_psnr)
               
                log_list.append({'PSNR': psnr, 'Rate': rate, 'MC-PSNR': mc_psnr, 'MCerr-PSNR': mc_err_psnr,
                                 'my': estimate_bpp(likelihood_m[0]).item(), 'mz': estimate_bpp(likelihood_m[1]).item(),
                                 'ry': estimate_bpp(likelihood_r[0]).item(), 'rz': estimate_bpp(likelihood_r[1]).item(),
                                 'PSNR-Y': psnr_Y, 'PSNR-U': psnr_U, 'PSNR-V': psnr_V,
                                 'MC-PSNR-Y': mc_psnr_Y, 'MC-PSNR-U': mc_psnr_U, 'MC-PSNR-V': mc_psnr_V})

            else:
                qp = {256: 37, 512: 32, 1024: 27, 2048: 22}[self.args.lmda]
                dataset_root = os.getenv('DATASET')

                # Read the binary files directly for accurate bpp estimate.
                size_byte = os.path.getsize(f'{dataset_root}TestVideo/x265/{qp}/bin_420/{seq_name}/frame_{frame_id}.mkv')
                rate = size_byte * 8 / height / width

                reconstructed_YUV = (reconstructed_Y, reconstructed_U, reconstructed_V)
                coding_frame_YUV = (coding_frame_Y, coding_frame_U, coding_frame_V)
                psnr_Y = mse2psnr(torch.mean(self.criterion(reconstructed_Y, coding_frame_Y), dim=(2,3)))
                psnr_U = mse2psnr(torch.mean(self.criterion(reconstructed_U, coding_frame_U), dim=(2,3)))
                psnr_V = mse2psnr(torch.mean(self.criterion(reconstructed_V, coding_frame_V), dim=(2,3)))

                psnr = 1/8 * (6 * psnr_Y + psnr_U + psnr_V)

                log_list.append({'PSNR': psnr, 'Rate': rate})

            metrics['PSNR'].append(psnr)
            metrics['Rate'].append(rate)


            frame_id += 1

        for m in metrics_name:
            metrics[m] = np.mean(metrics[m])
        
        logs = {'seq_name': seq_name, 'metrics': metrics, 'log_list': log_list,}

        return {'test_log': logs}


    def test_epoch_end(self, outputs):
        dataset_name = {'HEVC-B': ['BasketballDrive', 'BQTerrace', 'Cactus', 'Kimono1', 'ParkScene'],
                        'UVG': ['Beauty', 'Bosphorus', 'HoneyBee', 'Jockey', 'ReadySteadyGo', 'ShakeNDry', 'YachtRide'],
                        'HEVC-C': ['BasketballDrill', 'BQMall', 'PartyScene', 'RaceHorses']}
        metrics_name = list(outputs[0]['test_log']['metrics'].keys()) # Get all metrics' names

        uvg_rd = {}
        hevc_b_rd = {}
        hevc_c_rd = {}
        for metrics in metrics_name:
            uvg_rd[metrics] = []
            hevc_b_rd[metrics] = []
            hevc_c_rd[metrics] = []

        single_seq_logs = {}
        for metrics in metrics_name:
            single_seq_logs[metrics] = {}

        single_seq_logs['LOG'] = {}
        single_seq_logs['GOP'] = {} # Will not be printed currently
        single_seq_logs['Seq_Names'] = []

        for logs in [log['test_log'] for log in outputs]:
            seq_name = logs['seq_name']
            for metrics in logs['metrics'].keys():
                if seq_name in dataset_name['UVG']:
                    uvg_rd[metrics].append(logs['metrics'][metrics])
                elif seq_name in dataset_name['HEVC-B']:
                    hevc_b_rd[metrics].append(logs['metrics'][metrics])
                elif seq_name in dataset_name['HEVC-C']:
                    hevc_c_rd[metrics].append(logs['metrics'][metrics])
                else:
                    print("Unexpected sequence name:", seq_name)
                    raise NotImplementedError
            
            # Initialize
            if seq_name not in single_seq_logs['Seq_Names']:
                single_seq_logs['Seq_Names'].append(seq_name)
                for metrics in metrics_name:
                    single_seq_logs[metrics][seq_name] = []
                single_seq_logs['LOG'][seq_name] = []
                single_seq_logs['GOP'][seq_name] = []
            
            # Collect metrics logs
            for metrics in metrics_name:
                    single_seq_logs[metrics][seq_name].append(logs['metrics'][metrics])
            single_seq_logs['LOG'][seq_name].extend(logs['log_list'])
            single_seq_logs['GOP'][seq_name] = len(logs['log_list'])


        os.makedirs(self.args.save_dir + f'/report', exist_ok=True)

        for seq_name, log_list in single_seq_logs['LOG'].items():
            with open(self.args.save_dir + f'/report/{seq_name}.csv', 'w', newline='') as report:
                writer = csv.writer(report, delimiter=',')
                columns = ['frame'] + list(log_list[1].keys())
                writer.writerow(columns)

                #writer.writerow(['frame', 'PSNR', 'total bits', 'MC-PSNR', 'my', 'mz', 'ry', 'rz', 'MCerr-PSNR'])

                for idx in range(len(log_list)):
                    writer.writerow([f'frame_{idx+1}'] + list(log_list[idx].values()))
                    
        for metrics in metrics_name:
            logs['test/UVG '+metrics] = np.mean(uvg_rd[metrics])
            logs['test/HEVC-B '+metrics] = np.mean(hevc_b_rd[metrics])
            logs['test/HEVC-C '+metrics] = np.mean(hevc_c_rd[metrics])


        # Summary
        logs = {}
        print_log = '{:>16} '.format('Sequence_Name')
        for metrics in metrics_name:
            print_log += '{:>12}'.format(metrics)
        print_log += '\n'
        
        for seq_name in single_seq_logs['Seq_Names']:
            print_log += '{:>16} '.format(seq_name)

            for metrics in metrics_name:
                print_log += '{:12.4f}'.format(np.mean(single_seq_logs[metrics][seq_name]))

            print_log += '\n'
        print_log += '================================================\n'
        print_log += '{:>16} '.format('UVG')
        for metrics in metrics_name:
            print_log += '{:12.4f}'.format(np.mean(uvg_rd[metrics]))
        print_log += '\n'
        
        print_log += '{:>16} '.format('HEVC-B')
        for metrics in metrics_name:
            print_log += '{:12.4f}'.format(np.mean(hevc_b_rd[metrics]))
        print_log += '\n'
        print_log += '{:>16} '.format('HEVC-C')
        for metrics in metrics_name:
            print_log += '{:12.4f}'.format(np.mean(hevc_c_rd[metrics]))
        print_log += '\n'

        print(print_log)

        with open(self.args.save_dir + f'/brief_summary.txt', 'w', newline='') as report:
            report.write(print_log)

        self.log_dict(logs)

        return None

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        lr_step = [300000, 600000]
        lr_gamma = 0.1

        optimizer = optim.Adam([dict(params=self.main_parameters(), lr=self.args.lr),
                                dict(params=self.aux_parameters(), lr=self.args.lr * 10)])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_step, lr_gamma)

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

    def mc_net_forward(self, ref_frame, coding_frame):
        ref_frame_Y, ref_frame_U, ref_frame_V = ref_frame
        ref_frame_Y_s2d = space_to_depth(ref_frame_Y, block_size=2)

        coding_frame_Y = coding_frame
        warped_Y = self.Resampler(ref_frame_Y, coding_frame_Y)
        warped_Y_s2d = space_to_depth(warped_Y, block_size=2)

        coding_frame_UV = F.interpolate(coding_frame, scale_factor=0.5, mode='bilinear', align_corners=False)
        #coding_frame_UV = coding_frame_UV.detach() ####################
        warped_U = self.Resampler(ref_frame_U, coding_frame_UV / 2) # Divide by 2 since resolution is downed
        warped_V = self.Resampler(ref_frame_V, coding_frame_UV / 2) # Divide by 2 since resolution is downed

        mc_net_input = [ref_frame_Y_s2d, ref_frame_U, ref_frame_V, warped_Y_s2d, warped_U, warped_V]
        if self.MCNet is not None:
            mc_frame = self.MCNet(*mc_net_input)
        else:
            mc_frame = torch.cat(mc_net_input, dim=1)

        return mc_frame

    def compress(self, ref_frame, coding_frame, p_order):
        flow = self.MENet(ref_frame, coding_frame)

        flow_hat, mv_strings, mv_shape = self.Motion.compress(flow, return_hat=True, p_order=p_order)

        strings, shapes = [mv_strings], [mv_shape]

        mc_frame = self.mc_net_forward(ref_frame, flow_hat)

        predicted = mc_frame

        #res = coding_frame - predicted
        reconstructed, res_strings, res_shape = self.Residual.compress(coding_frame, mc_frame, return_hat=True)
        #reconstructed = predicted + res_hat
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

        reconstructed = self.Residual.decompress(res_strings, res_shape)
        #reconstructed = predicted + res_hat

        return reconstructed

    def setup(self, stage):
        self.logger.experiment.log_parameters(self.args)

        dataset_root = os.getenv('DATASET')
        qp = {256: 37, 512: 32, 1024: 27, 2048: 22}[self.args.lmda]

        if stage == 'fit':
            transformer = transforms.Compose([
                transforms.RandomCrop((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])

            self.train_dataset = VideoDataIframe(dataset_root + "vimeo_septuplet/", 'BPG_QP' + str(qp), 7,
                                                 transform=transformer)
            self.val_dataset = VideoTestDataIframeYUV(dataset_root + "TestVideo", self.args.lmda, first_gop=True)

        elif stage == 'test':
            self.test_dataset = VideoTestDataIframeYUV(dataset_root + "TestVideo", self.args.lmda)

        else:
            raise NotImplementedError

    def train_dataloader(self):
        # REQUIRED
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.args.batch_size,
                                  num_workers=self.args.num_workers,
                                  shuffle=True)
        return train_loader

    def val_dataloader(self):
        # OPTIONAL
        val_loader = DataLoader(self.val_dataset,
                                batch_size=1,
                                num_workers=self.args.num_workers,
                                shuffle=False)
        return val_loader

    def test_dataloader(self):
        # OPTIONAL
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=1,
                                 num_workers=self.args.num_workers,
                                 shuffle=False)
        return test_loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the arguments for this LightningModule
        """
        # MODEL specific
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', '-lr', dest='lr', default=1e-4, type=float)
        parser.add_argument('--batch_size', default=16, type=int)
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

    save_root = os.getenv('LOG', './') + 'torchDVC/'

    parser = argparse.ArgumentParser(add_help=True)

    # add args from trainer
    parser = Trainer.add_argparse_args(parser)

    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = Pframe.add_model_specific_args(parser)

    trc.add_coder_args(parser)

    # training specific
    parser.add_argument('--restore', type=str, choices=['none', 'resume', 'load', 'load_from_DVC_tgtIn'], default='none')
    parser.add_argument('--restore_exp_key', type=str, default=None)
    parser.add_argument('--restore_exp_epoch', type=int, default=49)
    parser.add_argument('--test', "-T", action="store_true")
    parser.add_argument('--experiment_name', type=str, default='basic')
    parser.add_argument('--project_name', type=str, default="ANFIC_for_Residual_Coding")
    parser.set_defaults(gpus=1)

    # parse params
    args = parser.parse_args()


    experiment_name = args.experiment_name
    project_name = args.project_name

    torch.backends.cudnn.deterministic = True
    save_path = os.path.join(save_root, project_name, experiment_name + '-' + str(args.lmda))

    res_coder = trc.get_coder_from_args(args)
    pred_coder = trc.get_coder_from_args(args)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        save_last=True,
        #every_n_epochs=10, # Save at least every 10 epochs
        period=3, # Save at least every 3 epochs
        verbose=True,
        monitor='val/loss',
        mode='min',
        prefix=''
    )

    db = None
    if args.gpus > 1:
        db = 'ddp'

    if args.restore == 'resume':
        comet_logger = CometLogger(
            api_key="bFaTNhLcuqjt1mavz02XPVwN8",
            project_name=project_name,
            workspace="tl32rodan",
            experiment_name=experiment_name + "-" + str(args.lmda),
            experiment_key = args.restore_exp_key,
            disabled=args.test or args.debug
        )
        args.save_dir = os.path.join(save_root, project_name, experiment_name + '-' + str(args.lmda))

        trainer = Trainer.from_argparse_args(args,
                                             checkpoint_callback=checkpoint_callback,
                                             gpus=args.gpus,
                                             distributed_backend=db,
                                             logger=comet_logger,
                                             default_root_dir=save_root,
                                             check_val_every_n_epoch=1,
                                             num_sanity_val_steps=0,
                                             terminate_on_nan=True,
                                             )

        epoch_num = args.restore_exp_epoch

        if args.restore_exp_key is None:
            prev_exp_key = {
                256: '1bae2dce18354df5913d476a428f3c61',
                512: 'f95aeb36be2a4ad2afe7b8d9296b8c3c',
                1024: 'a212b6be126744c49f2e6bac9469184b',
                2048: '022f238d7e6f46c8851b170040537e39'
            }[args.lmda]

            checkpoint = torch.load(os.path.join(save_root, project_name, prev_exp_key, "checkpoints",
                                                 f"epoch={epoch_num}.ckpt"),
                                    map_location=(lambda storage, loc: storage))
        else: # When prev_exp_key is specified in args
            checkpoint = torch.load(os.path.join(save_root, project_name, args.restore_exp_key, "checkpoints",
                                                 f"epoch={epoch_num}.ckpt"),
                                    map_location=(lambda storage, loc: storage))
        

        trainer.global_step = 590000
        trainer.current_epoch = epoch_num + 1


        # Change res_coder
        import copy
        args_copy = copy.deepcopy(args)
        args_copy.architecture = "CondANFHyperPriorCoderYUV"
        args_copy.output_nought = False
        args_copy.cond_coupling = True
        args_copy.num_features = 128
        args_copy.num_filters = 128
        args_copy.num_hyperpriors = 128
        args_copy.Mean = True
        args_copy.quant_mode = "RUN" # Random Uniform Noise ; suggested by James

        model = Pframe(args, res_coder, pred_coder).cuda()
        model.MCNet = Refinement(12, 64, out_channels=6) # channel # of Y:U:V=4:1:1 ; 2 inputs
        model.Residual = trc.get_coder_from_args(args_copy)()
        
        model.load_state_dict(checkpoint['state_dict'], strict=True)

    elif args.restore == 'load':
        comet_logger = CometLogger(
            api_key="bFaTNhLcuqjt1mavz02XPVwN8",
            project_name=project_name,
            workspace="tl32rodan",
            experiment_name=experiment_name + "-" + str(args.lmda),
            disabled=args.test or args.debug
        )
        args.save_dir = os.path.join(save_root, project_name, experiment_name + '-' + str(args.lmda))

        trainer = Trainer.from_argparse_args(args,
                                             checkpoint_callback=checkpoint_callback,
                                             gpus=args.gpus,
                                             distributed_backend=db,
                                             logger=comet_logger,
                                             default_root_dir=save_root,
                                             check_val_every_n_epoch=1,
                                             num_sanity_val_steps=-1,
                                             terminate_on_nan=True,
                                             #automatic_optimization=False # For manual backward
                                             )

        epoch_num = args.restore_exp_epoch
        if args.restore_exp_key is None:
            prev_exp_key = {
                256: '1bae2dce18354df5913d476a428f3c61',
                512: 'f95aeb36be2a4ad2afe7b8d9296b8c3c',
                1024: 'a212b6be126744c49f2e6bac9469184b',
                2048: '022f238d7e6f46c8851b170040537e39'
            }[args.lmda]

            checkpoint = torch.load(os.path.join(save_root, project_name, prev_exp_key, "checkpoints",
                                                 f"epoch={epoch_num}.ckpt"),
                                    map_location=(lambda storage, loc: storage))
        else: # When prev_exp_key is specified in args
            checkpoint = torch.load(os.path.join(save_root, project_name, args.restore_exp_key, "checkpoints",
                                                 f"epoch={epoch_num}.ckpt"),
                                    map_location=(lambda storage, loc: storage))
        

        #trainer.global_step = 4400000
        trainer.current_epoch = phase['trainALL_2frames'] + 1
        #trainer.current_epoch = phase['trainRes_fullgop'] + 1


        # Change res_coder
        import copy
        args_copy = copy.deepcopy(args)
        args_copy.architecture = "CondANFHyperPriorCoderYUV"
        args_copy.output_nought = False
        args_copy.cond_coupling = True
        args_copy.num_features = 128
        args_copy.num_filters = 128
        args_copy.num_hyperpriors = 128
        args_copy.in_channels = 6 # Y:U:V = 4:1:1
        args_copy.out_channels = 6 # Y:U:V = 4:1:1
        args_copy.Mean = True
        args_copy.quant_mode = "RUN" # Random Uniform Noise ; suggested by James

        model = Pframe(args, res_coder, pred_coder).cuda()
        model.MCNet = Refinement(12, 64, out_channels=6) # channel # of Y:U:V=4:1:1 ; 2 inputs
        model.Residual = trc.get_coder_from_args(args_copy)()
        
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    
    elif args.restore == 'load_from_DVC_tgtIn':
        comet_logger = CometLogger(
            api_key="bFaTNhLcuqjt1mavz02XPVwN8",
            project_name=project_name,
            workspace="tl32rodan",
            experiment_name=experiment_name + "-" + str(args.lmda),
            disabled=args.test or args.debug
        )

        args.save_dir = os.path.join(save_root,'DVC_baseline', experiment_name + '-' + str(args.lmda))

        trainer = Trainer.from_argparse_args(args,
                                             checkpoint_callback=checkpoint_callback,
                                             gpus=args.gpus,
                                             distributed_backend=db,
                                             logger=comet_logger,
                                             default_root_dir=save_root,
                                             check_val_every_n_epoch=1,
                                             num_sanity_val_steps=-1,
                                             terminate_on_nan=True,
                                             #automatic_optimization=False # For manual backward
                                             )

        checkpoint = torch.load(os.path.join(save_root, 'DVC_baseline', 'base_model_{}.ckpt'.format(args.lmda)),
                                map_location=(lambda storage, loc: storage))

        #trainer.global_step = 4400000
        #trainer.current_epoch = phase['trainMC'] + 1
        trainer.current_epoch = phase['trainALL_2frames'] + 1
        
        # Load DVC
        model = Pframe(args, res_coder, pred_coder).cuda()
        model.load_state_dict(checkpoint['model'], strict=True)

        # Load res_coder of tgtIn-mc-cond-YUV
        epoch_num = args.restore_exp_epoch
        args.save_dir = os.path.join(save_root, project_name, experiment_name + '-' + str(args.lmda))
        if args.restore_exp_key is None:
            prev_exp_key = {
                256: '1bae2dce18354df5913d476a428f3c61',
                512: 'f95aeb36be2a4ad2afe7b8d9296b8c3c',
                1024: 'a212b6be126744c49f2e6bac9469184b',
                2048: '022f238d7e6f46c8851b170040537e39'
            }[args.lmda]

            checkpoint = torch.load(os.path.join(save_root, project_name, prev_exp_key, "checkpoints",
                                                 f"epoch={epoch_num}.ckpt"),
                                    map_location=(lambda storage, loc: storage))
        else: # When prev_exp_key is specified in args
            checkpoint = torch.load(os.path.join(save_root, project_name, args.restore_exp_key, "checkpoints",
                                                 f"epoch={epoch_num}.ckpt"),
                                    map_location=(lambda storage, loc: storage))

        # Change res_coder
        import copy
        args_copy = copy.deepcopy(args)
        args_copy.architecture = "CondANFHyperPriorCoderYUV"
        args_copy.output_nought = False
        args_copy.cond_coupling = True
        args_copy.num_features = 128
        args_copy.num_filters = 128
        args_copy.num_hyperpriors = 128
        args_copy.Mean = True
        args_copy.quant_mode = "RUN" # Random Uniform Noise ; suggested by James

        _model = Pframe(args, res_coder, pred_coder).cuda()
        _model.MCNet = Refinement(12, 64, out_channels=6) # channel # of Y:U:V=4:1:1 ; 2 inputs
        _model.Residual = trc.get_coder_from_args(args_copy)()
        _model.load_state_dict(checkpoint['state_dict'], strict=True)
        
        model.MCNet = _model.MCNet # Replace MCNet
        model.Residual = _model.Residual # Replace resCoder

    else:
        comet_logger = CometLogger(
            api_key="bFaTNhLcuqjt1mavz02XPVwN8",
            project_name=project_name,
            workspace="tl32rodan",
            experiment_name=experiment_name + "-" + str(args.lmda),
            disabled=args.test or args.debug
        )
        

        trainer = Trainer.from_argparse_args(args,
                                             checkpoint_callback=checkpoint_callback,
                                             gpus=args.gpus,
                                             distributed_backend=db,
                                             logger=comet_logger,
                                             default_root_dir=save_root,
                                             check_val_every_n_epoch=1,
                                             num_sanity_val_steps=0,
                                             terminate_on_nan=True,
                                             #automatic_optimization=False # For manual backward 
                                             )

        # Change res_coder
        import copy
        args_copy = copy.deepcopy(args)
        args_copy.architecture = "CondANFHyperPriorCoderYUV"
        args_copy.output_nought = False
        args_copy.cond_coupling = True
        args_copy.num_features = 128
        args_copy.num_filters = 128
        args_copy.num_hyperpriors = 128
        args_copy.Mean = True
        args_copy.quant_mode = "RUN" # Random Uniform Noise ; suggested by James

        model = Pframe(args, res_coder, pred_coder).cuda()
        model.Residual = trc.get_coder_from_args(args_copy)()

    # comet_logger.experiment.log_code(file_name='torchDVC.py')

    if args.test:
        trainer.test(model)
    else:
        trainer.fit(model)
