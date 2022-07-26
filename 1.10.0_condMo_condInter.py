import argparse
import os
import csv
import sys

import yaml
import comet_ml
import flowiz as fz
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_compression as trc
import seaborn as sns
import matplotlib.pylab as plt

from functools import partial
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from torch import nn, optim
from torch.utils.data import DataLoader
from torch_compression.modules.entropy_models import EntropyBottleneck
from torch_compression.modules.conditional_module import conditional_warping, gen_discrete_condition, set_condition
from torchvision import transforms
from torchvision.utils import make_grid

from dataloader import VideoData, VideoDataYUV, VideoTestDataYUV
from dataloader_CLIC import CLICVideoDataYUV, CLICVideoTestDataYUV
from flownets import PWCNet, SPyNet
from SDCNet import SDCNet_3M
from models import Refinement
from util.psnr import mse2psnr
from util.sampler import Resampler
from util.ssim import MS_SSIM
from util.vision import PlotHeatMap, PlotFlow, save_image, YUV2RGB, plot_yuv
from util.vision import RGB2YUV_v2 as RGB2YUV
from util.vision import RGB2YUV420_v2 as RGB2YUV420
from util.functional import interpolate_as, space_to_depth, depth_to_space

plot_flow = PlotFlow().cuda()
plot_bitalloc = PlotHeatMap("RB").cuda()

phase = {
         'init': 0, 
         'trainME': 20, 
         'trainMC': 40, 
         'trainRes_I+P': 68, 
         'trainAll_I+P': 98, 
         'trainAll_fullgop': 1000
        }

# Custom pytorch-lightning trainer ; provide feature that configuring trainer.current_epoch
class CompressesModelTrainer(Trainer):
    def __init__(self, **kwargs):
        super(CompressesModelTrainer, self).__init__(**kwargs)

    @property
    def current_epoch(self) -> int:
        return self.fit_loop.current_epoch

    @current_epoch.setter
    def current_epoch(self, value):
        self.fit_loop.current_epoch = value

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
    def __init__(self, args, mo_coder, cond_mo_coder, res_coder):
        super(Pframe, self).__init__()
        self.args = args
        self.criterion = nn.MSELoss(reduction='none') if not self.args.ssim else MS_SSIM(data_range=1.).cuda()

        self.MENet = SPyNet(trainable=True)
        self.MWNet = SDCNet_3M(sequence_length=3) # Motion warping network
        self.MWNet.__delattr__('flownet')

        self.Motion = mo_coder
        self.CondMotion = cond_mo_coder

        self.Resampler = Resampler()
        self.MCNet = Refinement(6, 64, out_channels=3)

        self.Residual = res_coder
        self.frame_buffer = list()
        self.flow_buffer = list()

        self.automatic_optimization = False

    def load_args(self, args):
        self.args = args

    def YUVCriterion(self, y, gt):
        y_Y, y_U, y_V = y
        gt_Y, gt_U, gt_V = gt
        distortion_Y = torch.mean(self.criterion(gt_Y, y_Y), dim=(2,3))
        distortion_U = torch.mean(self.criterion(gt_U, y_U), dim=(2,3))
        distortion_V = torch.mean(self.criterion(gt_V, y_V), dim=(2,3))
        #distortion = ((distortion_Y + 1e-9)**(6/8) * (distortion_U + 1e-9)**(1/8) * (distortion_V + 1e-9)**(1/8)).mean()

        if self.args.loss_weight == '611':
            distortion = (distortion_Y * (6 / 8) + distortion_U * (1 / 8) + distortion_V * (1 / 8)).mean()
        elif self.args.loss_weight == '411':
            distortion = (distortion_Y * (4 / 6) + distortion_U * (1 / 6) + distortion_V * (1 / 6)).mean()
        elif self.args.loss_weight == '211':
            distortion = (distortion_Y * (2 / 4) + distortion_U * (1 / 4) + distortion_V * (1 / 4)).mean()
        elif self.args.loss_weight == '111':
            distortion = (distortion_Y * (1 / 3) + distortion_U * (1 / 3) + distortion_V * (1 / 3)).mean()
        
        return distortion

    def generate_flow(self, ref_frame, coding_frame):
        ref_frame_Y, ref_frame_U, ref_frame_V = ref_frame
        coding_frame_Y, coding_frame_U, coding_frame_V = coding_frame

        if args.ME_type == "YYY":
            _ref = torch.cat([ref_frame_Y, ref_frame_Y, ref_frame_Y], dim=1)  # Copy Y for 3 times for ME
            _coding = torch.cat([coding_frame_Y, coding_frame_Y, coding_frame_Y], dim=1)  # Copy Y for 3 times for ME
        elif args.ME_type == "YUV":
            _ref = torch.cat([
                              ref_frame_Y,
                              F.interpolate(torch.cat([ref_frame_U, ref_frame_V], dim=1), 
                                            scale_factor=2, mode='bilinear', align_corners=False)
                             ], dim=1)
            _coding = torch.cat([
                                 coding_frame_Y,
                                 F.interpolate(torch.cat([coding_frame_U, coding_frame_V], dim=1), 
                                               scale_factor=2, mode='bilinear', align_corners=False)
                                ], dim=1)
        elif args.ME_type == "RGB":
            _ref = YUV2RGB((ref_frame_Y, ref_frame_U, ref_frame_V), up=True)
            _coding = YUV2RGB((coding_frame_Y, coding_frame_U, coding_frame_V), up=True)
        else:
            assert False
        
        flow = self.MENet(_ref, _coding)

        return flow

    def warping_yuv(self, ref_frame, flow):
        ref_frame_Y, ref_frame_U, ref_frame_V = ref_frame

        # Warp luma component
        warped_Y = self.Resampler(ref_frame_Y, flow)
        flow_shape = flow.shape[-2:]

        # Warp chroma component
        # Divide by 2 since resolution is downed
        flow_UV = F.interpolate(flow, size=(flow_shape[0]//2, flow_shape[1]//2), mode='bilinear', align_corners=False) / 2
        #flow_UV = F.interpolate(flow, scale_factor=0.5, mode='bilinear', align_corners=False) / 2
        warped_U = self.Resampler(ref_frame_U, flow_UV) 
        warped_V = self.Resampler(ref_frame_V, flow_UV)

        return warped_Y, warped_U, warped_V, flow_UV

    def mc_net_forward(self, ref_frame, flow):
        warped_Y, warped_U, warped_V, flow_hat_UV = self.warping_yuv(ref_frame, flow)
        
        mc_net_input = [space_to_depth(ref_frame_Y, block_size=2), ref_frame_U, ref_frame_V, 
                        space_to_depth(warped_Y, block_size=2), warped_U, warped_V]

        mc_frame = self.MCNet(*mc_net_input)

        return mc_frame, flw_hat_UV

    def motion_forward(self, ref_frame, coding_frame, predict=False, visual=False, visual_prefix=''):
        ref_frame_Y, ref_frame_U, ref_frame_V = ref_frame
        coding_frame_Y, coding_frame_U, coding_frame_V = coding_frame

        self.MWNet.clear_buffer()

        flow = self.generate_flow(ref_frame, coding_frame)
        if predict:
            assert len(self.frame_buffer) == 3 or len(self.frame_buffer) == 2

            if len(self.frame_buffer) == 3:
                frame_buffer = [self.frame_buffer[0], self.frame_buffer[1], self.frame_buffer[2]]

            else:
                frame_buffer = [self.frame_buffer[0], self.frame_buffer[0], self.frame_buffer[1]]

            pred_frame, pred_flow = self.MWNet(frame_buffer,
                                               self.flow_buffer if len(self.flow_buffer) == 2 else None, True)

            flow_hat, likelihood_m, pred_flow_x2, _, _, _ = self.CondMotion(flow, output=pred_flow, cond_coupling_input=pred_flow, pred_prior_input=pred_frame,
                                                                             visual=visual, figname=visual_prefix+'_motion')

            self.MWNet.append_flow(flow_hat.detach())

            likelihoods = likelihood_m
            data = {'m_likelihood': likelihood_m, 'flow': flow, 'flow_hat': flow_hat, 
                    'pred_frame': pred_frame, 'pred_flow': pred_flow, 'pred_flow_x2': pred_flow_x2}

        else:
            flow_hat, likelihood_m = self.Motion(flow)

            data = {'m_likelihood': likelihood_m, 'flow': flow, 'flow_hat': flow_hat}

        return data

    def forward(self, ref_frame, coding_frame, predict=False, visual=False, visual_prefix=''):
        m_info = self.motion_forward(ref_frame, coding_frame, predict=predict, visual=visual, visual_prefix=visual_prefix)
        
        mc_frame, flow_hat_UV = self.mc_net_forward(ref_frame, m_info['flow_hat'])
        m_info['flow_hat_UV'] = flow_hat_UV

        coding_frame_Y, coding_frame_U, coding_frame_V = coding_frame
        coding_frame = torch.cat([space_to_depth(coding_frame_Y, block_size=2), coding_frame_U, coding_frame_V], dim=1)

        reconstructed, likelihood_r, mc_hat, _, _, _ = self.Residual(coding_frame, output=mc, cond_coupling_input=mc,
                                                                     visual=visual, figname=visual_prefix)
        
        rec444 = torch.cat([
                            depth_to_space(reconstructed[:, :4], block_size=2),
                            F.interpolate(reconstructed[:, 4:], scale_factor=2, mode='bilinear', align_corners=False)
                           ], dim=1).detach()

        self.frame_buffer.append(rec444)

        return reconstructed, m_info['m_likelihood'], likelihood_r, m_info, mc, mc_hat

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
            
        epoch = self.current_epoch
        
        batch_Y, batch_U, batch_V = batch
        batch_Y, batch_U, batch_V = batch_Y.cuda(), batch_U.cuda(), batch_V.cuda()

        ref_frame_Y, ref_frame_U, ref_frame_V = batch_Y[:, 0], batch_U[:, 0], batch_V[:, 0]

        # I-frame ; inference only
        if self.args.lmda == 65536:
            lmda_ = 5e-1  # 5e-1
        elif self.args.lmda == 16384:
            lmda_ = 1e-1  # 1e-1
        elif self.args.lmda == 4096:
            lmda_ = 5e-2  # 5e-2
        elif self.args.lmda == 1024:
            lmda_ = 1e-2  # 1e-2
        else:
            assert False
        lmda_I = [lmda_ for i in range(0, self.args.batch_size)]
        lmda_I = torch.Tensor(lmda_I).view(-1, 1).cuda()

        if iter == 0:
            print(lmda_I.flatten().cpu().numpy())
        set_condition(self.Icoder, lmda_I)

        #ref_frame = batch[:, 0]
        #ref_frame = RGB2YUV420(ref_frame)
        ref_frame = torch.cat([space_to_depth(ref_frame_Y, 2), ref_frame_U, ref_frame_V], dim=1)
        with torch.no_grad():
            yuv_tilde, likelihoods_I, _, _, _, _ = self.Icoder(ref_frame)

        ref_frame_Y = depth_to_space(yuv_tilde[:, :4], block_size=2).clamp(0, 1).detach()
        ref_frame_U = yuv_tilde[:, 4:5].clamp(0, 1).detach()
        ref_frame_V = yuv_tilde[:, 5:].clamp(0, 1).detach()
        
        loss_list = []
        dist_list = []
        rate_list = []
        mc_error_list = []
        pred_error_list = []

        if epoch <= phase['trainMC']:
            for name, param in model.named_parameters():
                param.requires_grad_(True)
            _phase = 'Motion'

            #coding_frame = batch[:, 1]
            #coding_frame_Y, coding_frame_U, coding_frame_V = RGB2YUV_v2(coding_frame, down=True)
            coding_frame_Y, coding_frame_U, coding_frame_V = batch_Y[:, 1], batch_U[:, 1], batch_V[:, 1]
            
            m_info = self.motion_forward(ref_frame, coding_frame, predict=False)

            if epoch <= phase['trainME']:
                warped_Y, warped_U, warped_V, flow_hat_UV = self.warping_yuv(ref_frame, m_info['flow_hat'])
                reconstructed_YUV = (warped_Y, warped_U, warped_V)
            else:
                mc, flow_hat_UV = self.mc_net_forward(ref_frame, m_info['flow_hat'])
                
                reconstructed_Y, reconstructed_U, reconstructed_V = depth_to_space(mc[:, :4], block_size=2), mc[:, 4:5], mc[:, 5:]
                
                reconstructed_YUV = (reconstructed_Y, reconstructed_U, reconstructed_V)

            m_info['flow_hat_UV'] = flow_hat_UV
            coding_frame_YUV = (coding_frame_Y, coding_frame_U, coding_frame_V)

            distortion = self.YUVCriterion(reconstructed_YUV, coding_frame_YUV)
            rate = trc.estimate_bpp(likelihood_m, input=coding_frame_Y)

            loss = self.args.lmda * distortion.mean() + rate.mean()
            
            loss_list.append(loss)
            dist_list.append(distortion.mean())
            rate_list.append(rate.mean())

            # One the other P-frame
            self.buffer = []
            for idx in range(3):
                self.buffer.append(torch.cat([
                                              batch_Y[:, idx],
                                              F.interpolate(torch.cat([batch_U[:, idx], batch_V[:, idx]], dim=1), 
                                                            scale_factor=2, mode='bilinear', align_corners=False)
                                             ], dim=1)
                                  )
            flow_1 = self.MENet((space_to_depth(batch_Y[:, 1], 2), batch_U[:, 1], batch_V[:, 1]),
                                (space_to_depth(batch_Y[:, 2], 2), batch_U[:, 2], batch_V[:, 2]))
            self.flow_buffer = [data['flow_hat'], flow_1]
            ref_frame_Y, ref_frame_U, ref_frame_V = batch_Y[:, 2], batch_U[:, 2], batch_V[:, 2]
            coding_frame_Y, coding_frame_U, coding_frame_V = batch_Y[:, 3], batch_U[:, 3], batch_V[:, 3]

            m_info = self.motion_forward(ref_frame, coding_frame, predict=True)
            
            if epoch <= phase['trainME']:
                warped_Y, warped_U, warped_V, flow_hat_UV = self.warping_yuv(ref_frame, m_info['flow_hat'])
                reconstructed_YUV = (warped_Y, warped_U, warped_V)
            else:
                mc, flow_hat_UV = self.mc_net_forward(ref_frame, m_info['flow_hat'])
                
                reconstructed_Y, reconstructed_U, reconstructed_V = depth_to_space(mc[:, :4], block_size=2), mc[:, 4:5], mc[:, 5:]
                
                reconstructed_YUV = (reconstructed_Y, reconstructed_U, reconstructed_V)
            
            m_info['flow_hat_UV'] = flow_hat_UV
            coding_frame_YUV = (coding_frame_Y, coding_frame_U, coding_frame_V)

            distortion = self.YUVCriterion(reconstructed_YUV, coding_frame_YUV)
            rate = trc.estimate_bpp(likelihood_m, input=coding_frame_Y)

            pred_hat_Y, pred_hat_U, pred_hat_V, _ = self.warping_yuv(ref_frame, m_info['pred_flow_hat'])
            pred_frame_hat = torch.cat([
                                        pred_hat_Y,
                                        F.interpolate(pred_hat_U, scale_factor=2, mode='bilinear', align_corners=False),
                                        F.interpolate(pred_hat_V, scale_factor=2, mode='bilinear', align_corners=False),
                                       ], dim=1)
            pred_frame_error = nn.MSELoss(reduction='none')(m_info['pred_frame'], pred_frame_hat)
            loss = self.args.lmda * distortion.mean() + rate.mean() + 0.01 * self.args.lmda * pred_frame_error.mean()
            
            loss_list.append(loss)
            dist_list.append(distortion.mean())
            rate_list.append(rate.mean())

            total_loss = torch.mean(loss_list)

            logs = {
                    'train/loss': total_loss.item(),
                    'train/distortion': torch.mean(distortion).item(),
                    'train/rate': torch.mean(rate).item(),
                    'train/pred_frame_error': pred_frame_error.mean().item()
                   }
            self.manual_backward(loss)
            opt.zero_grad()
            opt.step()

        else:
            if epoch <= phase['trainRes_I+P']:
                _phase = 'Res'
            else:
                _phase = 'All'
            reconstructed_Y, reconstructed_U, reconstructed_V = ref_frame_Y, ref_frame_U, ref_frame_V

            self.MWNet.clear_buffer()
            for frame_idx in range(1, 10):
                ref_frame_Y, ref_frame_U, ref_frame_V = reconstructed_Y, reconstructed_U, reconstructed_V
                
                if frame_idx > 3 and epoch <= phase['trainAll_I+P']: # 2-frame training stages
                        break

                # Detach reference frame when RNN training
                ref_frame_Y, ref_frame_U, ref_frame_V = ref_frame_Y.detach(), ref_frame_U.detach(), ref_frame_V.detach() 

                coding_frame_Y, coding_frame_U, coding_frame_V = batch_Y[:, frame_idx], batch_U[:, frame_idx], batch_V[:, frame_idx]

                # Disable gradients of motion modules when _phase == Res
                with torch.set_grad_enabled( _phase == 'All'): 
                    m_info = self.motion_forward(ref_frame, coding_frame, predict=(frame_idx != 1))
                    
                    mc, flow_hat_UV = self.mc_net_forward(ref_frame, m_info['flow_hat'])
                    m_info['flow_hat_UV'] = flow_hat_UV

                if len(self.frame_buffer) == 4:
                    self.frame_buffer.pop(0)
                    assert len(self.frame_buffer) == 3, str(len(self.frame_buffer))

                coding_frame = torch.cat([space_to_depth(coding_frame_Y, block_size=2), coding_frame_U, coding_frame_V], dim=1)

                rec, likelihood_r, mc_hat, _, _, _ = self.Residual(coding_frame, output=mc, cond_coupling_input=mc)
                
                rec_Y, rec_U, rec_V = depth_to_space(rec[:, :4], block_size=2), rec[:, 4:5], rec[:, 5:]

                reconstructed_YUV = (rec_Y, rec_U, rec_V)
                coding_frame_YUV = (coding_frame_Y, coding_frame_U, coding_frame_V)
                distortion = self.YUVCriterion(reconstructed_YUV, coding_frame_YUV)

                rate = trc.estimate_bpp(m_info['m_likelihood'], input=coding_frame_Y) + trc.estimate_bpp(likelihood_r, input=coding_frame_Y)
                mc_error = nn.MSELoss(reduction='none')(mc, mc_hat)

                pred_hat_Y, pred_hat_U, pred_hat_V, _ = self.warping_yuv(ref_frame, m_info['pred_flow_hat'])
                pred_frame_hat = torch.cat([
                                            pred_hat_Y,
                                            F.interpolate(pred_hat_U, scale_factor=2, mode='bilinear', align_corners=False),
                                            F.interpolate(pred_hat_V, scale_factor=2, mode='bilinear', align_corners=False),
                                           ], dim=1)
                pred_frame_error = nn.MSELoss(reduction='none')(m_info['pred_frame'], pred_frame_hat)
                loss = self.args.lmda * distortion.mean() + rate.mean() + \
                       0.01 * self.args.lmda * (mc_error.mean() + pred_frame_error.mean())
                loss_list.append(loss)
                
                dist_list.append(distortion.mean())
                rate_list.append(rate.mean())
                mc_error_list.append(mc_error.mean())
                pred_error_list.append(pred_frame_error.mean())

                self.manual_backward(loss)
                opt.step()
                opt.zero_grad()

            total_loss = torch.mean(torch.tensor(loss_list))
            distortion = torch.mean(torch.tensor(dist_list))
            rate = torch.mean(torch.tensor(rate_list))
            mc_error = torch.mean(torch.tensor(mc_error_list))
            pred_frame_error = torch.mean(torch.tensor(pred_error_list))
            logs = {
                    'train/loss': total_loss.item(),
                    'train/distortion': distortion.item(), 
                    'train/PSNR': mse2psnr(distortion.item()), 
                    'train/rate': rate.item(), 
                    'train/mc_error': mc_error.item(),
                    'train/pred_frame_error': pred_frame_error.item()
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

        #return total_loss
        return None


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

        dataset_name, seq_name, batch_Y, batch_U, batch_V, frame_id_start = batch
        frame_id = int(frame_id_start)

        seq_name = seq_name[0]
        dataset_name = dataset_name[0]

        gop_size = batch_Y.size(1)
        height, width = batch_Y.size()[-2:]

        psnr_list = []
        mc_psnr_list = []
        mse_list = []
        rate_list = []
        m_rate_list = []
        loss_list = []
        align = trc.util.Alignment()
        align_UV = trc.util.Alignment(divisor=32)


        coding_frame_Y, coding_frame_U, coding_frame_V = batch_Y[:, 0], batch_U[:, 0], batch_V[:, 0]

        if self.args.lmda == 65536:
            lmda_ = 5e-1  # 5e-1
        elif self.args.lmda == 16384:
            lmda_ = 1e-1  # 1e-1
        elif self.args.lmda == 4096:
            lmda_ = 5e-2  # 5e-2
        elif self.args.lmda == 1024:
            lmda_ = 1e-2  # 1e-2
        else:
            assert False

        set_condition(self.Icoder, torch.Tensor([lmda_]).view(1, 1).cuda())


        for frame_idx in range(gop_size):
            if frame_idx != 0:
                ref_frame_Y, ref_frame_U, ref_frame_V = (
                                                         align.align(rec_Y), 
                                                         align_UV.align(rec_U), 
                                                         align_UV.align(rec_V), 
                                                        )

                coding_frame_Y, coding_frame_U, coding_frame_V = (
                                                                  align.align(batch_Y[:, frame_idx]), 
                                                                  align_UV.align(batch_U[:, frame_idx]), 
                                                                  align_UV.align(batch_V[:, frame_idx]), 
                                                                 )

                if frame_idx == 1:
                    self.frame_buffer = [align.align(ref_frame)]
                    self.flow_buffer = list()

                rec, likelihood_m, likelihood_r, m_info, mc_frame, mc_hat = self((ref_frame_Y, ref_frame_U, ref_frame_V),
                                                                                 (coding_frame_Y, coding_frame_U, coding_frame_V),
                                                                                 predict= (frame_idx != 1))
                if len(self.frame_buffer) == 4:
                    self.frame_buffer.pop(0)
                    assert len(self.frame_buffer) == 3, str(len(self.frame_buffer))

                flow = m_info['flow']
                flow_hat = m_info['flow_hat']
                flow_hat_UV = m_info['flow_hat_UV']
                pred_flow = m_info['pred_flow']
                pref_flow_hat = m_info['pred_flow_x2']

                rec = (depth_to_space(rec[:, :4], block_size=2), rec[:, 4:5], rec[:, 5:])
                rec_Y, rec_U, rec_V = rec
                rec = YUV2RGB(rec, up=True)
                
                mc_frame = (depth_to_space(mc_frame[:, :4], block_size=2), mc_frame[:, 4:5], mc_frame[:, 5:])
                mc_frame_Y, mc_frame_U, mc_frame_V = mc_frame
                mc_frame = YUV2RGB(mc_frame, up=True)
                
                mc_hat = (depth_to_space(mc_hat[:, :4], block_size=2), mc_hat[:, 4:5], mc_hat[:, 5:])
                mc_hat_Y, mc_hat_U, mc_hat_V = mc_hat
                mc_hat = YUV2RGB(mc_hat, up=True)

                ref_frame = YUV2RGB((ref_frame_Y, ref_frame_U, ref_frame_V), up=True)
                ref_frame = align.resume(ref_frame).clamp(0, 1)

                rec_Y, rec_U, rec_V = (
                                       align.resume(rec_Y).clamp(0, 1), 
                                       align_UV.resume(rec_U).clamp(0, 1), 
                                       align_UV.resume(rec_V).clamp(0, 1)
                                      )
                rec = align.resume(rec).clamp(0, 1)

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
                       trc.estimate_bpp(likelihood_r, input=ref_frame_Y).item()
                m_rate = trc.estimate_bpp(likelihood_m, input=ref_frame_Y).item()
                

                rec_YUV = (rec_Y, rec_U, rec_V)
                coding_frame_YUV = (coding_frame_Y, coding_frame_U, coding_frame_V)
                mse = self.YUVCriterion(rec_YUV, coding_frame_YUV)
                psnr = 1/8 * (
                                6 * mse2psnr(torch.mean(self.criterion(rec_Y, coding_frame_Y), dim=(2,3))) + \
                                1 * mse2psnr(torch.mean(self.criterion(rec_U, coding_frame_U), dim=(2,3))) + \
                                1 * mse2psnr(torch.mean(self.criterion(rec_V, coding_frame_V), dim=(2,3)))
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
                    raw_flow_rgb = torch.from_numpy(
                        fz.convert_from_flow(flow[0].permute(1, 2, 0).cpu().numpy()) / 255).permute(2, 0, 1)
                    upload_img(raw_flow_rgb.cpu().numpy(), f'{seq_name}_flow.png', grid=False)

                    flow_rgb = torch.from_numpy(
                                    fz.convert_from_flow(flow_hat[0].permute(1, 2, 0).cpu().numpy()) / 255).permute(2, 0, 1)
                    upload_img(flow_rgb.cpu().numpy(), f'{seq_name}_flow_map.png', grid=False)
                    flow_rgb = torch.from_numpy(
                                    fz.convert_from_flow(flow_hat_UV[0].permute(1, 2, 0).cpu().numpy()) / 255).permute(2, 0, 1)
                    upload_img(flow_rgb.cpu().numpy(), f'{seq_name}_flow_map_UV.png', grid=False)
                    flow_rgb = torch.from_numpy(
                                    fz.convert_from_flow(pred_flow[0].permute(1, 2, 0).cpu().numpy()) / 255).permute(2, 0, 1)
                    upload_img(flow_rgb.cpu().numpy(), f'{seq_name}_pred_flow.png', grid=False)
                    flow_rgb = torch.from_numpy(
                                    fz.convert_from_flow(pred_flow_x2[0].permute(1, 2, 0).cpu().numpy()) / 255).permute(2, 0, 1)
                    upload_img(flow_rgb.cpu().numpy(), f'{seq_name}_pred_flow_x2.png', grid=False)

                    upload_img(ref_frame.cpu().numpy()[0], f'{seq_name}_ref_frame.png', grid=False)
                    upload_img(coding_frame.cpu().numpy()[0], f'{seq_name}_gt_frame.png', grid=False)
                    upload_img(mc_frame.cpu().numpy()[0], f'{seq_name}_MC_frame.png', grid=False)
                    upload_img(rec.cpu().numpy()[0],
                               seq_name + '_rec_{:.3f}.png'.format(mc_psnr),
                               grid=False)
                    upload_img(mc_hat.cpu().numpy()[0], f'{seq_name}_predicted_MC_frame.png', grid=False)

                loss = self.args.lmda * mse + rate + 0.01 * self.args.lmda * mc_error.mean()

                mc_psnr_list.append(mc_psnr)
                m_rate_list.append(m_rate)

            else:
                coding_frame_Y, coding_frame_U, coding_frame_V = (
                    align.align(batch_Y[:, 0]),
                    align_UV.align(batch_U[:, 0]),
                    align_UV.align(batch_V[:, 0])
                )
                coding_frame_yuv = torch.cat([space_to_depth(coding_frame_Y, 2), coding_frame_U, coding_frame_V], dim=1)

                rec, likelihoods_I, _, _, _, _ = self.Icoder(coding_frame_yuv)

                rec_Y, rec_U, rec_V = (depth_to_space(rec[:, :4], block_size=2), rec[:, 4:5], rec[:, 5:])

                rec_Y, rec_U, rec_V = (
                    rec_Y.clamp(0, 1),
                    rec_U.clamp(0, 1),
                    rec_V.clamp(0, 1)
                )

                rates = [trc.estimate_bpp(likelihoods_I[n], input=rec_Y).item() for n in range(Icoder.num_bitstreams)]
                rate = sum(rates)

                rec_YUV = (rec_Y, rec_U, rec_V)
                coding_frame_YUV = (coding_frame_Y, coding_frame_U, coding_frame_V)
                psnr_Y = mse2psnr(torch.mean(self.criterion(rec_Y, coding_frame_Y), dim=(2, 3)))
                psnr_U = mse2psnr(torch.mean(self.criterion(rec_U, coding_frame_U), dim=(2, 3)))
                psnr_V = mse2psnr(torch.mean(self.criterion(rec_V, coding_frame_V), dim=(2, 3)))

                psnr = 1 / 8 * (6 * psnr_Y + psnr_U + psnr_V)
                mse = self.YUVCriterion(rec_YUV, coding_frame_YUV)
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

        logs = {'dataset_name': dataset_name, 'seq_name': seq_name, 
                'val_loss': loss, 'val_mse': mse, 
                'val_psnr': psnr, 'val_rate': rate, 
                'val_mc_psnr': mc_psnr, 'val_m_rate': m_rate}

        return {'val_log': logs}

    def validation_epoch_end(self, outputs):
        rd_dict = {}
        loss = []

        for logs in [log['val_log'] for log in outputs]:
            dataset_name = logs['dataset_name']
            seq_name = logs['seq_name']

            if not (dataset_name in rd_dict.keys()):
                rd_dict[dataset_name] = {}
                rd_dict[dataset_name]['psnr'] = []
                rd_dict[dataset_name]['rate'] = []
                rd_dict[dataset_name]['mc_psnr'] = []
                rd_dict[dataset_name]['m_rate'] = []

            rd_dict[dataset_name]['psnr'].append(logs['val_psnr'])
            rd_dict[dataset_name]['rate'].append(logs['val_rate'])
            rd_dict[dataset_name]['mc_psnr'].append(logs['val_mc_psnr'])
            rd_dict[dataset_name]['m_rate'].append(logs['val_m_rate'])
   
            loss.append(logs['val_loss'])

        avg_loss = np.mean(loss)
        
        logs = {'val/loss': avg_loss}

        for dataset_name, rd in rd_dict.items():
            logs['val/'+dataset_name+' psnr'] = np.mean(rd['psnr'])
            logs['val/'+dataset_name+' rate'] = np.mean(rd['rate'])
            logs['val/'+dataset_name+' mc_psnr'] = np.mean(rd['mc_psnr'])
            logs['val/'+dataset_name+' m_rate'] = np.mean(rd['m_rate'])

        self.log_dict(logs)

        return None

    def test_step(self, batch, batch_idx):
        metrics_name = ['PSNR', 'Rate', 'Mo_Rate', 'MC-PSNR', 'MCerr-PSNR', 
                        'PSNR-Y', 'PSNR-U', 'PSNR-V', 
                        'MC-PSNR-Y', 'MC-PSNR-U', 'MC-PSNR-V']
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
        # MC-PSNR-V: PSNR(gt_V, mc_frame_V)
                        
        dataset_name, seq_name, batch_Y, batch_U, batch_V, frame_id_start = batch
        frame_id = int(frame_id_start)

        seq_name = seq_name[0]
        dataset_name = dataset_name[0]

        gop_size = batch_Y.size(1)
        height, width = batch_Y.size()[-2:]
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

        if self.args.lmda == 65536:
            lmda_ = 5e-1  # 5e-1
        elif self.args.lmda == 16384:
            lmda_ = 1e-1  # 1e-1
        elif self.args.lmda == 4096:
            lmda_ = 5e-2  # 5e-2
        elif self.args.lmda == 1024:
            lmda_ = 1e-2  # 1e-2
        else:
            assert False

        set_condition(self.Icoder, torch.Tensor([lmda_]).view(1, 1).cuda())
        os.makedirs(self.args.save_dir + f'/{seq_name}/flow', exist_ok=True)
        os.makedirs(self.args.save_dir + f'/{seq_name}/flow_hat', exist_ok=True)
        os.makedirs(self.args.save_dir + f'/{seq_name}/flow_hat_UV', exist_ok=True)
        os.makedirs(self.args.save_dir + f'/{seq_name}/pred_flow', exist_ok=True)
        os.makedirs(self.args.save_dir + f'/{seq_name}/pred_flow_x2', exist_ok=True)
        os.makedirs(self.args.save_dir + f'/{seq_name}/gt_frame', exist_ok=True)
        os.makedirs(self.args.save_dir + f'/{seq_name}/mc_frame', exist_ok=True)
        os.makedirs(self.args.save_dir + f'/{seq_name}/rec_frame', exist_ok=True)
        os.makedirs(self.args.save_dir + f'/{seq_name}/mc_hat', exist_ok=True)
        os.makedirs(self.args.save_dir + f'/{seq_name}/gt_frame_yuv', exist_ok=True)
        os.makedirs(self.args.save_dir + f'/{seq_name}/rec_frame_yuv', exist_ok=True)
        os.makedirs(self.args.save_dir + f'/{seq_name}/bit_alloc_motion', exist_ok=True)
        os.makedirs(self.args.save_dir + f'/{seq_name}/bit_alloc_res', exist_ok=True)
        #os.makedirs(self.args.save_dir + f'/{seq_name}/mc_frame_yuv', exist_ok=True)

        for frame_idx in range(gop_size):
            TO_VISUALIZE = (frame_id_start == 1 and frame_idx < 8 and seq_name in ['ISCAS_Grand_Challenge_Validation1', 'Beauty'])

            if frame_idx != 0:
                ref_frame_Y, ref_frame_U, ref_frame_V = (
                                                         align.align(rec_Y), 
                                                         align_UV.align(rec_U), 
                                                         align_UV.align(rec_V) 
                                                        )
                coding_frame_Y, coding_frame_U, coding_frame_V = (
                                                                  align.align(batch_Y[:, frame_idx]), 
                                                                  align_UV.align(batch_U[:, frame_idx]), 
                                                                  align_UV.align(batch_V[:, frame_idx])
                                                                 )
                coding_frame_yuv = torch.cat([space_to_depth(coding_frame_Y, 2), coding_frame_U, coding_frame_V], dim=1)

                if frame_idx == 1:
                    self.frame_buffer = [align.align(ref_frame)]
                    self.flow_buffer = list()

                # reconstruced frame will be next ref_frame
                if False and TO_VISUALIZE:
                    os.makedirs(os.path.join(self.args.save_dir, 'visualize_ANFIC', f'batch_{batch_idx}'), exist_ok=True)
                    rec, likelihood_m, likelihood_r, m_info, mc_frame, mc_hat = self((ref_frame_Y, ref_frame_U, ref_frame_V),
                                                                                     (coding_frame_Y, coding_frame_U, coding_frame_V),
                                                                                     visual=True,
                                                                                     visual_prefix=os.path.join(
                                                                                                                self.args.save_dir, 
                                                                                                                'visualize_ANFIC',
                                                                                                                f'batch_{batch_idx}',
                                                                                                                f'frame_{frame_idx}',
                                                                                                               ),
                                                                                     predict= (frame_idx != 1)
                                                                                     )
                else:
                    rec, likelihood_m, likelihood_r, m_info, mc_frame, mc_hat = self((ref_frame_Y, ref_frame_U, ref_frame_V),
                                                                                     (coding_frame_Y, coding_frame_U, coding_frame_V),
                                                                                     predict= (frame_idx != 1)
                                                                                    )

                if len(self.frame_buffer) == 4:
                    self.frame_buffer.pop(0)
                    assert len(self.frame_buffer) == 3, str(len(self.frame_buffer))

                flow = m_info['flow']
                flow_hat = m_info['flow_hat']
                flow_hat_UV = m_info['flow_hat_UV']
                pred_flow = m_info['pred_flow']
                pref_flow_hat = m_info['pred_flow_x2']

                rec_yuv = plot_yuv(rec)
                gt_yuv = plot_yuv(coding_frame_yuv)
                rec = (depth_to_space(rec[:, :4], block_size=2), rec[:, 4:5], rec[:, 5:])
                rec_Y, rec_U, rec_V = rec
                rec = YUV2RGB(rec, up=True)
                
                mc_frame_yuv = mc_frame
                mc_frame = (depth_to_space(mc_frame[:, :4], block_size=2), mc_frame[:, 4:5], mc_frame[:, 5:])
                mc_frame_Y, mc_frame_U, mc_frame_V  = mc_frame
                mc_frame = YUV2RGB(mc_frame, up=True)
                
                mc_hat = (depth_to_space(mc_hat[:, :4], block_size=2), mc_hat[:, 4:5], mc_hat[:, 5:])
                mc_hat_Y, mc_hat_U, mc_hat_V = mc_hat
                mc_hat = YUV2RGB(mc_hat, up=True)

                ref_frame = YUV2RGB((ref_frame_Y, ref_frame_U, ref_frame_V), up=True)

                rec_Y, rec_U, rec_V = (rec_Y.clamp(0, 1), rec_U.clamp(0, 1), rec_V.clamp(0, 1))
                rec = rec.clamp(0, 1)

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

                if TO_VISUALIZE:
                    plt.figure(figsize=(20, 12))
                    mo_bitalloc = torch.sum(likelihood_m[0].log()/-np.log(2.), (0, 1)).cpu().numpy()
                    ax = sns.heatmap(mo_bitalloc)
                    plt.savefig(self.args.save_dir + f'/{seq_name}/bit_alloc_motion/'
                               f'frame_{int(frame_id_start + frame_idx)}.png')
                    plt.figure(figsize=(20, 12))
                    res_bitalloc = torch.sum(likelihood_r[0].log()/-np.log(2.), (0, 1)).cpu().numpy()
                    ax = sns.heatmap(res_bitalloc)
                    plt.savefig(self.args.save_dir + f'/{seq_name}/bit_alloc_res/'
                               f'frame_{int(frame_id_start + frame_idx)}.png')
                    if frame_idx == 1:
                        print('mo_bitalloc =', np.sum(mo_bitalloc))
                        print('res_bitalloc =', np.sum(res_bitalloc))
                        print('rough rate =', np.sum(mo_bitalloc)/(1920*1080) + np.sum(res_bitalloc)/(1920*1080))
                        print('rate =', trc.estimate_bpp(likelihood_m, input=rec_Y).item() + trc.estimate_bpp(likelihood_r, input=rec_Y).item())
                        print('my =', estimate_bpp(likelihood_m[0]).item())
                        print('mz =', estimate_bpp(likelihood_m[1]).item())
                        print('ry =', estimate_bpp(likelihood_r[0]).item())
                        print('rz =', estimate_bpp(likelihood_r[1]).item())

                    flow_map = plot_flow(flow)
                    save_image(flow_map, self.args.save_dir + f'/{seq_name}/flow/f{int(frame_idx)}_flow.png', nrow=1)
                    flow_map = plot_flow(flow_hat)
                    save_image(flow_map, self.args.save_dir + f'/{seq_name}/flow_hat/f{int(frame_idx)}_flow.png', nrow=1)
                    flow_map = plot_flow(flow_hat_UV)
                    save_image(flow_map, self.args.save_dir + f'/{seq_name}/flow_hat_UV/f{int(frame_idx)}_flow.png', nrow=1)
                    flow_map = plot_flow(pred_flow)
                    save_image(flow_map, self.args.save_dir + f'/{seq_name}/pred_flow/f{int(frame_idx)}_flow.png', nrow=1)
                    flow_map = plot_flow(pred_flow_x2)
                    save_image(flow_map, self.args.save_dir + f'/{seq_name}/pred_flow_x2/f{int(frame_idx)}_flow.png', nrow=1)

                    save_image(coding_frame[0], self.args.save_dir + f'/{seq_name}/gt_frame/'
                               f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(mc_frame[0], self.args.save_dir + f'/{seq_name}/mc_frame/'
                               f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(rec[0], self.args.save_dir + f'/{seq_name}/rec_frame/'
                               f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(mc_hat[0], self.args.save_dir + f'/{seq_name}/mc_hat/'
                               f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(gt_yuv[0], self.args.save_dir + f'/{seq_name}/gt_frame_yuv/'
                               f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(rec_yuv[0], self.args.save_dir + f'/{seq_name}/rec_frame_yuv/'
                               f'frame_{int(frame_id_start + frame_idx)}.png')

                rate = trc.estimate_bpp(likelihood_m, input=rec_Y).item() + trc.estimate_bpp(likelihood_r, input=rec_Y).item()
                m_rate = trc.estimate_bpp(likelihood_m, input=rec_Y).item()
                metrics['Mo_Rate'].append(m_rate)
                 
                psnr_Y = mse2psnr(torch.mean(self.criterion(rec_Y, coding_frame_Y), dim=(2,3)))
                psnr_U = mse2psnr(torch.mean(self.criterion(rec_U, coding_frame_U), dim=(2,3)))
                psnr_V = mse2psnr(torch.mean(self.criterion(rec_V, coding_frame_V), dim=(2,3)))

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
                coding_frame_Y, coding_frame_U, coding_frame_V = (
                    align.align(batch_Y[:, 0]),
                    align_UV.align(batch_U[:, 0]),
                    align_UV.align(batch_V[:, 0])
                )
                coding_frame_yuv = torch.cat([space_to_depth(coding_frame_Y, 2), coding_frame_U, coding_frame_V], dim=1)

                reconstructed, likelihoods_I, _, _, _, _ = self.Icoder(coding_frame_yuv)

                rec = (depth_to_space(reconstructed[:, :4], block_size=2), reconstructed[:, 4:5], reconstructed[:, 5:])
                rec_Y, rec_U, rec_V = rec
                rec_Y, rec_U, rec_V = (
                    rec_Y.clamp(0, 1),
                    rec_U.clamp(0, 1),
                    rec_V.clamp(0, 1)
                )
                rec_rgb = YUV2RGB(rec, up=True)
                coding_frame = YUV2RGB((coding_frame_Y, coding_frame_U, coding_frame_V), up=True)

                gt_yuv = plot_yuv(coding_frame_yuv)
                rec_yuv = plot_yuv(reconstructed)
                
                if TO_VISUALIZE:
                    save_image(coding_frame[0], self.args.save_dir + f'/{seq_name}/gt_frame/'
                               f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(rec_rgb[0], self.args.save_dir + f'/{seq_name}/rec_frame/'
                               f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(gt_yuv[0], self.args.save_dir + f'/{seq_name}/gt_frame_yuv/'
                               f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(rec_yuv[0], self.args.save_dir + f'/{seq_name}/rec_frame_yuv/'
                               f'frame_{int(frame_id_start + frame_idx)}.png')

                rates = [trc.estimate_bpp(
                    likelihoods_I[n], input=rec_Y).item() for n in range(Icoder.num_bitstreams)]
                rate = sum(rates)

                rec_YUV = (rec_Y, rec_U, rec_V)
                coding_frame_YUV = (coding_frame_Y, coding_frame_U, coding_frame_V)
                psnr_Y = mse2psnr(torch.mean(self.criterion(rec_Y, coding_frame_Y), dim=(2, 3)))
                psnr_U = mse2psnr(torch.mean(self.criterion(rec_U, coding_frame_U), dim=(2, 3)))
                psnr_V = mse2psnr(torch.mean(self.criterion(rec_V, coding_frame_V), dim=(2, 3)))

                psnr = 1 / 8 * (6 * psnr_Y + psnr_U + psnr_V)
                log_list.append({'PSNR': psnr, 'Rate': rate})

            metrics['PSNR'].append(psnr)
            metrics['Rate'].append(rate)


            frame_id += 1

        for m in metrics_name:
            metrics[m] = np.mean(metrics[m])
        
        logs = {'dataset_name': dataset_name, 'seq_name': seq_name, 'metrics': metrics, 'log_list': log_list,}

        return {'test_log': logs}

    def test_epoch_end(self, outputs):
        dataset_name = {'HEVC-B': ['BasketballDrive', 'BQTerrace', 'Cactus', 'Kimono1', 'ParkScene'],
                        'UVG': ['Beauty', 'Bosphorus', 'HoneyBee', 'Jockey', 'ReadySteadyGo', 'ShakeNDry', 'YachtRide'],
                        'HEVC-C': ['BasketballDrill', 'BQMall', 'PartyScene', 'RaceHorses'],
                       }

        metrics_name = list(outputs[0]['test_log']['metrics'].keys())  # Get all metrics' names

        rd_dict = {}

        single_seq_logs = {}
        for metrics in metrics_name:
            single_seq_logs[metrics] = {}

        single_seq_logs['LOG'] = {}
        single_seq_logs['GOP'] = {}  # Will not be printed currently
        single_seq_logs['Seq_Names'] = []

        for logs in [log['test_log'] for log in outputs]:
            dataset_name = logs['dataset_name']
            seq_name = logs['seq_name']

            if not (dataset_name in rd_dict.keys()):
                rd_dict[dataset_name] = {}
                
                for metrics in metrics_name:
                    rd_dict[dataset_name][metrics] = []

            for metrics in logs['metrics'].keys():
                rd_dict[dataset_name][metrics].append(logs['metrics'][metrics])

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

                # writer.writerow(['frame', 'PSNR', 'total bits', 'MC-PSNR', 'my', 'mz', 'ry', 'rz', 'MCerr-PSNR'])

                for idx in range(len(log_list)):
                    writer.writerow([f'frame_{idx + 1}'] + list(log_list[idx].values()))

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
        for dataset_name, rd in rd_dict.items():
            print_log += '{:>16} '.format(dataset_name)

            for metrics in metrics_name:
                logs['test/' + dataset_name + ' ' + metrics] = np.mean(rd[metrics])
                print_log += '{:12.4f}'.format(np.mean(rd[metrics]))

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

    def compress(self, ref_frame, coding_frame):
        #return reconstructed, strings, shapes
        raise NotImplementedError

    def decompress(self, ref_frame, strings, shapes):
        #return reconstructed
        raise NotImplementedError

    def setup(self, stage):
        self.logger.experiment.log_parameters(self.args)

        dataset_root = os.getenv('DATASET')
        qp = {1024: 37, 4096: 32, 16384: 27, 65536: 22}[self.args.lmda]

        if stage == 'fit':
            transformer = transforms.Compose([
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])

            self.train_dataset = VideoDataYUV(dataset_root + "YUV_Sequences/", 10, transform=transformer)
            self.val_dataset = VideoTestDataYUV(dataset_root, self.args.lmda, first_gop=True, sequence=('UVG', 'HEVC-B', 'HEVC-C'))

        elif stage == 'test':
            self.test_dataset = VideoTestDataYUV(dataset_root, self.args.lmda, first_gop=args.test_first_gop, sequence=('ISCAS-val'))
            #self.test_dataset = VideoTestDataIframe(dataset_root + "TestVideo", self.args.lmda)

        else:
            raise NotImplementedError

    def train_dataloader(self):
        # REQUIRED
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.args.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
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
        parser.add_argument('--lmda', default=65536, choices=[1024, 4096, 16384, 65536], type=int)
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

    #save_root = "/work/u4803414/torchDVC/"
    save_root = os.getenv('LOG', './') + 'torchDVC/'

    parser = argparse.ArgumentParser(add_help=True)

    # add args from trainer
    parser = Trainer.add_argparse_args(parser)

    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = Pframe.add_model_specific_args(parser)

    #trc.add_coder_args(parser)

    # training specific
    parser.add_argument("--disable_signalconv", "-DS", action="store_true", help="Enable SignalConv or not.")
    parser.add_argument("--deconv_type", "-DT", type=str, default="Signal", 
                        choices=trc.__DECONV_TYPES__.keys(), help="Configure deconvolution type")

    parser.add_argument('--restore', type=str, choices=['none', 'resume', 'load', 'custom'], default='none')
    parser.add_argument('--restore_exp_key', type=str, default=None)
    parser.add_argument('--restore_exp_epoch', type=int, default=49)
    parser.add_argument('--test', "-T", action="store_true")
    parser.add_argument('--test_first_gop', action="store_true")
    parser.add_argument('--experiment_name', type=str, default='basic')
    parser.add_argument('--project_name', type=str, default="CANFVC-Lite-YUV")

    parser.add_argument('--motion_coder_conf', type=str, default=None)
    parser.add_argument('--cond_motion_coder_conf', type=str, default=None)
    parser.add_argument('--residual_coder_conf', type=str, default=None)
    parser.add_argument('--prev_motion_coder_conf', type=str, default=None)
    parser.add_argument('--prev_cond_motion_coder_conf', type=str, default=None)
    parser.add_argument('--prev_residual_coder_conf', type=str, default=None)

    parser.add_argument('--iframe_coder_ckpt', type=str,
                        default="/work/tl32rodan/models/ANFIC/ANFHyperPriorCoder_1022_1715/")
    parser.add_argument('--ME_type', type=str, choices=['YYY', 'YUV', 'RGB'], default="YUV")
    parser.add_argument('--loss_weight', type=str, default="611")
    parser.set_defaults(gpus=1)

    # parse params
    args = parser.parse_args()

    experiment_name = args.experiment_name
    project_name = args.project_name

    torch.backends.cudnn.deterministic = True
 
    # Set conv/deconv type ; Signal or Standard
    conv_type = "Standard" if args.disable_signalconv else trc.SignalConv2d
    deconv_type = "Transpose" if args.disable_signalconv and args.deconv_type != "Signal" else args.deconv_type
    trc.set_default_conv(conv_type=conv_type, deconv_type=deconv_type)
    
    # Config coders
    assert not (args.motion_coder_conf is None)
    mo_coder_cfg = yaml.safe_load(open(args.motion_coder_conf, 'r'))
    assert mo_coder_cfg['model_architecture'] in trc.__CODER_TYPES__.keys()
    mo_coder_arch = trc.__CODER_TYPES__[mo_coder_cfg['model_architecture']]
    mo_coder = mo_coder_arch(**mo_coder_cfg['model_params'])

    assert not (args.cond_motion_coder_conf is None)
    cond_mo_coder_cfg = yaml.safe_load(open(args.cond_motion_coder_conf, 'r'))
    assert cond_mo_coder_cfg['model_architecture'] in trc.__CODER_TYPES__.keys()
    cond_mo_coder_arch = trc.__CODER_TYPES__[cond_mo_coder_cfg['model_architecture']]
    cond_mo_coder = cond_mo_coder_arch(**cond_mo_coder_cfg['model_params'])
    
    assert not (args.residual_coder_conf is None)
    res_coder_cfg = yaml.safe_load(open(args.residual_coder_conf, 'r'))
    assert res_coder_cfg['model_architecture'] in trc.__CODER_TYPES__.keys()
    res_coder_arch = trc.__CODER_TYPES__[res_coder_cfg['model_architecture']]
    res_coder = res_coder_arch(**res_coder_cfg['model_params'])

    # I-frame coder
    Icoder = trc.hub.AugmentedNormalizedFlowHyperPriorCoder420(
        [128], 320, 192, gdn_mode="standard",
        num_layers=2, use_DQ=True, share_wei=False,
        init_code="gaussian", use_code=False, dec_add=False,
        use_attn=False,
        use_mean=False, use_context=True, condition="GaussianMixtureModel", quant_mode="RUN").cuda()

    cond_warping_I = partial(conditional_warping, conditions=1, ver=2)
    cond_warping_I(Icoder)
    ckpt = torch.load(os.path.join(args.iframe_coder_ckpt, "model.ckpt"), map_location='cpu')
    Icoder.load_state_dict(ckpt['coder'])
    Icoder.eval() # I-frame is only for inference

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        save_last=True,
        every_n_epochs=1, # Save at least every 10 epochs
        verbose=True,
        monitor='val/loss',
        mode='min',
        filename = '{epoch}'
    )

    db = None
    if args.gpus > 1:
        db = 'ddp'

    comet_logger = CometLogger(
        api_key="bFaTNhLcuqjt1mavz02XPVwN8",
        project_name=project_name,
        workspace="tl32rodan",
        experiment_name=experiment_name + "-" + str(args.lmda),
        experiment_key = args.restore_exp_key if args.restore == 'resume' else None,
        disabled=args.test or args.debug
    )
    
    args.save_dir = os.path.join(save_root, project_name, experiment_name + '-' + str(args.lmda))

    if args.restore == 'resume':
        trainer = CompressesModelTrainer.from_argparse_args(args,
                                                            enable_checkpointing=True,
                                                            callbacks=checkpoint_callback,
                                                            gpus=args.gpus,
                                                            strategy=db,
                                                            logger=comet_logger,
                                                            default_root_dir=save_root,
                                                            check_val_every_n_epoch=1,
                                                            num_sanity_val_steps=0,
                                                            log_every_n_steps=50,
                                                            detect_anomaly=True,
                                                           )
        epoch_num = args.restore_exp_epoch
        if args.restore_exp_key is None:
            raise ValueError
        else:  # When prev_exp_key is specified in args
            checkpoint = torch.load(os.path.join(save_root, project_name, args.restore_exp_key, "checkpoints",
                                                 f"epoch={epoch_num}.ckpt"),
                                    map_location=(lambda storage, loc: storage))

        trainer.current_epoch = epoch_num + 1
        
        model = Pframe(args, mo_coder, cond_mo_coder, res_coder).cuda()
        model.Icoder = Icoder
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    
    elif args.restore == 'load':
        trainer = CompressesModelTrainer.from_argparse_args(args,
                                                            enable_checkpointing=True,
                                                            callbacks=checkpoint_callback,
                                                            gpus=args.gpus,
                                                            strategy=db,
                                                            logger=comet_logger,
                                                            default_root_dir=save_root,
                                                            check_val_every_n_epoch=1,
                                                            num_sanity_val_steps=-1,
                                                            log_every_n_steps=10,
                                                            detect_anomaly=True,
                                                           )
        
        epoch_num = args.restore_exp_epoch
        if args.restore_exp_key is None:
            raise ValueError
        else:  # When prev_exp_key is specified in args
            checkpoint = torch.load(os.path.join(save_root, project_name, args.restore_exp_key, "checkpoints",
                                                 f"epoch={epoch_num}.ckpt"),
                                    map_location=(lambda storage, loc: storage))

        trainer.current_epoch = phase['trainAll_I+P'] + 1
        
        model = Pframe(args, mo_coder, cond_mo_coder, res_coder).cuda()
        model.Icoder = Icoder
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    
    elif args.restore == 'custom':
        trainer = CompressesModelTrainer.from_argparse_args(args,
                                                            enable_checkpointing=True,
                                                            callbacks=checkpoint_callback,
                                                            gpus=args.gpus,
                                                            strategy=db,
                                                            logger=comet_logger,
                                                            default_root_dir=save_root,
                                                            check_val_every_n_epoch=1,
                                                            num_sanity_val_steps=-1,
                                                            log_every_n_steps=50,
                                                            detect_anomaly=True,
                                                           )
        
        epoch_num = args.restore_exp_epoch
        if args.restore_exp_key is None:
            raise ValueError
        else:  # When prev_exp_key is specified in args
            checkpoint = torch.load(os.path.join(save_root, project_name, args.restore_exp_key, "checkpoints",
                                                 f"epoch={epoch_num}.ckpt"),
                                    map_location=(lambda storage, loc: storage))

        trainer.current_epoch = phase['trainAll_I+P'] + 1
        # Previous coders
        assert not (args.prev_motion_coder_conf is None)
        prev_mo_coder_cfg = yaml.safe_load(open(args.prev_motion_coder_conf, 'r'))
        assert prev_mo_coder_cfg['model_architecture'] in trc.__CODER_TYPES__.keys()
        prev_mo_coder_arch = trc.__CODER_TYPES__[prev_mo_coder_cfg['model_architecture']]
        prev_mo_coder = prev_mo_coder_arch(**prev_mo_coder_cfg['model_params'])
        
        assert not (args.prev_cond_motion_coder_conf is None)
        prev_cond_mo_coder_cfg = yaml.safe_load(open(args.prev_cond_motion_coder_conf, 'r'))
        assert prev_cond_mo_coder_cfg['model_architecture'] in trc.__CODER_TYPES__.keys()
        prev_cond_mo_coder_arch = trc.__CODER_TYPES__[prev_cond_mo_coder_cfg['model_architecture']]
        prev_cond_mo_coder = prev_cond_mo_coder_arch(**prev_cond_mo_coder_cfg['model_params'])

        assert not (args.prev_residual_coder_conf is None)
        prev_res_coder_cfg = yaml.safe_load(open(args.prev_residual_coder_conf, 'r'))
        assert prev_res_coder_cfg['model_architecture'] in trc.__CODER_TYPES__.key()
        prev_res_coder_arch = trc.__CODER_TYPES__[prev_res_coder_cfg['model_architecture']]
        prev_res_coder = prev_res_coder_arch(**prev_res_coder_cfg['model_params'])
   
        model = Pframe(args, prev_mo_coder, prev_cond_mo_coder, prev_res_coder).cuda()
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        
        model.Motion = motion_coder
        model.CondMotion = cond_motion_coder
        model.Residual = res_coder
        model.Icoder = Icoder
    
    else:
        trainer = CompressesModelTrainer.from_argparse_args(args,
                                                            enable_checkpointing=True,
                                                            callbacks=checkpoint_callback,
                                                            gpus=args.gpus,
                                                            strategy=db,
                                                            logger=comet_logger,
                                                            default_root_dir=save_root,
                                                            check_val_every_n_epoch=10,
                                                            num_sanity_val_steps=-1,
                                                            log_every_n_steps=50,
                                                            detect_anomaly=True,
                                                           )
        
        model = Pframe(args, mo_coder, cond_mo_coder, res_coder).cuda()
        model.Icoder = Icoder

    if args.test:
        trainer.test(model)
    else:
        trainer.fit(model)
