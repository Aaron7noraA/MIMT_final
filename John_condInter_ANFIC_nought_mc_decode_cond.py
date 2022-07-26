import argparse
import collections
import csv
import os
from functools import partial
from logging import raiseExceptions
from pickle import TRUE

import comet_ml
import flowiz as fz
import numpy as np
import torch
import torch_compression as trc
import yaml
from matplotlib import collections
from matplotlib.pyplot import axis
from ptflops import get_model_complexity_info
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from torch import nn, optim
from torch.utils.data import DataLoader
from torch_compression.hub import AugmentedNormalizedFlowHyperPriorCoder
from torch_compression.modules.entropy_models import EntropyBottleneck
from torchinfo import summary
from torchvision import transforms
from torchvision.utils import make_grid

from dataloader import VideoDataIframe, VideoTestDataIframe
from flownets import PWCNet, SPyNet
from models import Refinement
from util.psnr import mse2psnr
from util.sampler import Resampler, warp
from util.ssim import MS_SSIM
from util.vision import PlotFlow, PlotHeatMap, save_image

plot_flow = PlotFlow().cuda()
plot_bitalloc = PlotHeatMap("RB").cuda()

phase = {'trainMV': 5, 
         'trainMC': 9, 
         'trainRes_2frames': 15, 
         'trainRes_fullgop': 18,
         'trainAll_2frames': 22,
         'trainAll_fullgop': 32, #30,#22, 
         # 'trainAll_RNN_1': 35,#25, 
         # 'trainAll_RNN_2': 38,}#28}
         }

# phase = {'trainMV': 15, 
#          'trainRes_2frames_RecOnly': 21, 
#          'trainRes_2frames': 26, 
#          'trainRes_fullgop': 28,
#          'trainAll_2frames': 31,
#          'trainAll_fullgop': 37,
#          'trainAll_RNN_1': 40, 
#          'trainAll_RNN_2': 42}

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
    def __init__(self, args, mo_coder, res_coder):
        super(Pframe, self).__init__()
        self.args = args
        self.criterion = nn.MSELoss(reduction='none') if not self.args.ssim else MS_SSIM(data_range=1.).cuda()
        
        self.if_model = AugmentedNormalizedFlowHyperPriorCoder(128, 320, 192, num_layers=2, use_DQ=True, use_code=False,
                                                               use_context=True, condition='GaussianMixtureModel',
                                                               quant_mode='round')

        self.frame_extractor = nn.Sequential(
                                nn.Conv2d(3, self.args.hidden_channels, kernel_size=5, stride=1, padding=2),
                                nn.LeakyReLU(0.01)
                           )

        if self.args.MENet == 'PWC':
            self.MENet = PWCNet(trainable=False)
        elif self.args.MENet == 'SPy':
            self.MENet = SPyNet(trainable=False)

        self.Motion = mo_coder

        self.Resampler = Resampler()
        # self.MCNet = Refinement(6, 64)
        self.MCNet = Refinement(3 + self.args.hidden_channels, 64, self.args.hidden_channels)
        self.mc_decode = nn.Conv2d(self.args.hidden_channels, 3, 3, padding=1)


        self.Residual = res_coder
        self.output_nought = self.Residual.output_nought


    def load_args(self, args):
        self.args = args

    def motion_forward(self, ref_frame, coding_frame):
        flow = self.MENet(ref_frame, coding_frame)

        if yaml.safe_load(open(self.args.motion_coder_conf, 'r'))['model_architecture'] == "ANFHyperPriorCoder":
            flow_hat, likelihood_m, _, _, _, _ = self.Motion(flow)
        else:
            flow_hat, likelihood_m = self.Motion(flow)

        mc, warped = self.mc_net_forward(ref_frame, flow_hat)

        return mc, {'m_likelihood': likelihood_m, 'flow_hat': flow_hat, 'flow': flow, 'warped': warped}

    def mc_net_forward(self, ref_frame, coding_frame):
        # warped = self.Resampler(ref_frame, coding_frame)
        # if self.MCNet is not None:
        #     mc_net_input = [ref_frame, warped]

        #     mc_frame = self.MCNet(*mc_net_input)
        # else:
        #     mc_frame = warped

        # return mc_frame
        warped_feat = self.Resampler(self.frame_extractor(ref_frame), coding_frame)
        warped = self.Resampler(ref_frame, coding_frame)
        if self.MCNet is not None:
            mc_net_input = [warped, warped_feat]

            mc = self.MCNet(*mc_net_input)
        else:
            mc = warped

        return mc, warped

    def forward(self, ref_frame, coding_frame, p_order=0, res_visual=False, visual_prefix = ''):
        mc, m_info = self.motion_forward(ref_frame, coding_frame)
        decoded_mc = self.mc_decode(mc)

        predicted, intra_info, likelihood_i = mc, 0, ()
        cond_coupling_input = torch.cat((mc, decoded_mc), axis=1)

        if self.output_nought:
            nought = torch.zeros_like(coding_frame)
            reconstructed, likelihood_r, y2, _, _, BDQ = self.Residual(coding_frame, output=nought, cond_coupling_input=cond_coupling_input,
                                                                            visual=res_visual, figname=visual_prefix)
        else:
            reconstructed, likelihood_r, mc_hat, _, _, BDQ = self.Residual(coding_frame, output=decoded_mc, cond_coupling_input=cond_coupling_input,
                                                                            visual=res_visual, figname=visual_prefix)
        
        likelihoods = m_info['m_likelihood'] + likelihood_i + likelihood_r
        
        if self.output_nought:
            return reconstructed, likelihoods, m_info, m_info['flow_hat'], decoded_mc, predicted, intra_info, BDQ, y2
        else:
            return reconstructed, likelihoods, m_info, m_info['flow_hat'], decoded_mc, predicted, intra_info, BDQ, mc_hat
        

    def training_step(self, batch, batch_idx):
        epoch = self.current_epoch

        batch = batch.cuda()
        ref_frame = batch[:, 0]
        # I-frame
        with torch.no_grad():
            ref_frame, _, _, _, _, _ = self.if_model(ref_frame)
        
        if epoch <= phase['trainMV']:
            _phase = 'MV'
            coding_frame = batch[:, 1]

            flow = self.MENet(ref_frame, coding_frame)
            flow_hat, likelihood_m = self.Motion(flow)
            reconstructed = self.Resampler(ref_frame, flow_hat)
            likelihoods = likelihood_m

            distortion = self.criterion(coding_frame, reconstructed)
            rate = trc.estimate_bpp(likelihoods, input=coding_frame)

            loss = self.args.lmda * distortion.mean() + rate.mean()
            logs = {'train/loss': loss.item(),
                    'train/distortion': distortion.mean().item(),
                    'train/rate': rate.mean().item()}

        elif epoch <= phase['trainMC']:
            _phase = 'MC'
            coding_frame = batch[:, 1]

            flow = self.MENet(ref_frame, coding_frame)
            flow_hat, likelihood_m = self.Motion(flow)
            mc, warp = self.mc_net_forward(ref_frame, flow_hat)
            reconstructed = self.mc_decode(mc)
            likelihoods = likelihood_m

            distortion = self.criterion(coding_frame, reconstructed)
            rate = trc.estimate_bpp(likelihoods, input=coding_frame)

            loss = self.args.lmda * distortion.mean() + rate.mean()
            logs = {'train/loss': loss.item(),
                    'train/distortion': distortion.mean().item(),
                    'train/rate': rate.mean().item()}

        else:
            if self.args.restore == 'finetune':
                self.requires_grad_(True)

            if epoch <= phase['trainRes_fullgop']:
                _phase = 'RES'
            else:
                _phase = 'ALL'
            reconstructed = ref_frame

            loss = torch.tensor(0., dtype=torch.float, device=reconstructed.device)
            loss_list = []
            dist_list = []
            rate_list = []
            mc_error_list = []
            y2_error_list = []
            warping_loss_list = []

            for frame_idx in range(1, 7):
                ref_frame = reconstructed

                if frame_idx > 1:
                    if epoch <= phase['trainRes_2frames'] or (phase['trainRes_fullgop'] < epoch and epoch <= phase['trainAll_2frames']): # 2-frame training stages
                        break

                if epoch <= phase['trainRes_fullgop'] and frame_idx > 4:
                    break

                if epoch <= phase['trainAll_fullgop']:
                    ref_frame = ref_frame.detach() # Detach when RNN training
                elif frame_idx > 4:
                    break

                coding_frame = batch[:, frame_idx]

                if _phase == 'RES':
                    for module in [self.MENet, self.Motion, self.MCNet, self.mc_decode]:
                        for param in module.parameters(): 
                                self.optimizers().state[param] = {} # remove all state (step, exp_avg, exp_avg_sg)

                    self.Motion.requires_grad_(False)
                    self.MCNet.requires_grad_(False)
                    self.mc_decode.requires_grad_(False)
                    self.frame_extractor.requires_grad_(False)

                else:
                    self.Motion.requires_grad_(True)
                    self.MCNet.requires_grad_(True)
                    self.mc_decode.requires_grad_(True)
                    self.frame_extractor.requires_grad_(True)

                flow = self.MENet(ref_frame, coding_frame)
                flow_hat, likelihood_m = self.Motion(flow)

                mc, warp = self.mc_net_forward(ref_frame, flow_hat)
                decoded_mc = self.mc_decode(mc)

                cond_coupling_input = torch.cat((mc, decoded_mc), axis=1)
                
                if self.output_nought:
                    nought = torch.zeros_like(coding_frame)
                    reconstructed, likelihood_r, y2, _, _, _ = self.Residual(coding_frame, output=nought, cond_coupling_input=cond_coupling_input)
                else:
                    reconstructed, likelihood_r, mc_hat, _, _, _ = self.Residual(coding_frame, output=decoded_mc, cond_coupling_input=cond_coupling_input)

                likelihoods = likelihood_m + likelihood_r

                distortion = self.criterion(coding_frame, reconstructed)
                rate = trc.estimate_bpp(likelihoods, input=coding_frame)
                
                    
                if self.output_nought:
                    y2_error = nn.MSELoss(reduction='none')(nought, y2)
                    loss += self.args.lmda * distortion.mean() + rate.mean() + 0.01 * self.args.lmda * y2_error.mean() 
                    y2_error_list.append(y2_error.mean().detach())
                    
                else:
                    mc_error = nn.MSELoss(reduction='none')(decoded_mc, mc_hat)
                    loss += self.args.lmda * distortion.mean() + rate.mean() + 0.01 * self.args.lmda * mc_error.mean()
                    mc_error_list.append(mc_error.mean().detach())
                
                dist_list.append(distortion.mean().detach())
                rate_list.append(rate.mean().detach())
                

            loss /= frame_idx
            distortion = torch.mean(torch.tensor(dist_list))
            rate = torch.mean(torch.tensor(rate_list))
            if self.output_nought:
                y2_error = torch.mean(torch.tensor(y2_error_list))
                logs = {
                            'train/loss': loss.item(),
                            'train/distortion': distortion.item(), 
                            'train/PSNR': mse2psnr(distortion.item()), 
                            'train/rate': rate.item(), 
                            'train/y2_error': y2_error.item(),
                            'batch_size': self.args.batch_size
                        }
            else:
                mc_error = torch.mean(torch.tensor(mc_error_list))
                logs = {
                            'train/loss': loss.item(),
                            'train/distortion': distortion.item(), 
                            'train/PSNR': mse2psnr(distortion.item()), 
                            'train/rate': rate.item(), 
                            'train/mc_error': mc_error.item(),
                            'batch_size': self.args.batch_size
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

        return loss
    
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

        dataset_name, seq_name, batch, frame_id_start = batch
        frame_id = int(frame_id_start)

        ref_frame = batch[:, 0]

        batch = batch[:, 1:]
        seq_name = seq_name[0]
        dataset_name = dataset_name[0]

        gop_size = batch.size(1)

        height, width = ref_frame.size()[2:]

        psnr_list = []
        mc_psnr_list = []
        mse_list = []
        rate_list = []
        m_rate_list = []
        loss_list = []
        y2_error_list = []
        mc_error_list = []
        align = trc.util.Alignment()

        for frame_idx in range(gop_size):
            if frame_idx != 0:
                if self.output_nought:
                    rec_frame, likelihoods, m_info, flow_hat, mc_frame, _, _, BDQ, y2 = self(align.align(ref_frame),
                                                                                      align.align(batch[:, frame_idx]),
                                                                                      p_order=frame_idx)
                else:
                    rec_frame, likelihoods, m_info, flow_hat, mc_frame, _, _, BDQ, mc_hat = self(align.align(ref_frame),
                                                                                      align.align(batch[:, frame_idx]),
                                                                                      p_order=frame_idx)
                
                ref_frame = align.resume(ref_frame).clamp(0, 1)
                rec_frame = align.resume(rec_frame).clamp(0, 1)
                coding_frame = align.resume(batch[:, frame_idx]).clamp(0, 1)
                mc_frame = align.resume(mc_frame).clamp(0, 1)
                if self.output_nought:
                    y2 = align.resume(y2).clamp(0, 1)
                else:
                    mc_hat = align.resume(mc_hat).clamp(0, 1)
                
                BDQ = align.resume(BDQ).clamp(0, 1)

                rate = trc.estimate_bpp(likelihoods, input=ref_frame).mean().item()
                m_rate = trc.estimate_bpp(likelihoods[0], input=ref_frame).mean().item() + \
                         trc.estimate_bpp(likelihoods[1], input=ref_frame).mean().item()


                if frame_idx == 1:
                    mse = torch.mean((rec_frame - coding_frame).pow(2))
                    mc_mse = torch.mean((mc_frame - coding_frame).pow(2))
                    psnr = get_psnr(mse).cpu().item()
                    mc_psnr = get_psnr(mc_mse).cpu().item()

                    #flow_map = plot_flow(flow_hat.to('cuda:0'))
                    flow_rgb = torch.from_numpy(
                                    fz.convert_from_flow(m_info['flow'][0].permute(1, 2, 0).cpu().numpy()) / 255).permute(2, 0, 1)
                    upload_img(flow_rgb.cpu().numpy(), f'{seq_name}_raw_flow.png', grid=False)

                    flow_rgb = torch.from_numpy(
                                    fz.convert_from_flow(flow_hat[0].permute(1, 2, 0).cpu().numpy()) / 255).permute(2, 0, 1)
                    upload_img(flow_rgb.cpu().numpy(), f'{seq_name}_flow_map.png', grid=False)
                    upload_img(ref_frame.cpu().numpy()[0], f'{seq_name}_ref_frame.png', grid=False)
                    upload_img(coding_frame.cpu().numpy()[0], f'{seq_name}_gt_frame.png', grid=False)
                    upload_img(mc_frame.cpu().numpy()[0], f'{seq_name}_MC_frame.png', grid=False)
                    upload_img(rec_frame.cpu().numpy()[0],
                               seq_name + '_rec_frame_{:.3f}.png'.format(psnr),
                               grid=False)
                    upload_img(BDQ.cpu().numpy()[0], f'{seq_name}_before_DQ_frame.png', grid=False)
                    
                    if self.output_nought:
                        upload_img(y2.cpu().numpy()[0], f'{seq_name}_y2.png', grid=False)
                    else:
                        upload_img(mc_hat.cpu().numpy()[0], f'{seq_name}_predicted_MC_frame.png', grid=False)
                
                else:
                    upload_img(rec_frame.cpu().numpy()[0],
                               seq_name + '_rec_frame_{:.3f}_{}.png'.format(psnr, frame_idx),
                               grid=False)

                ref_frame = rec_frame

                mse = self.criterion(ref_frame, batch[:, frame_idx]).mean().item()
                psnr = mse2psnr(mse)
                mc_mse = self.criterion(mc_frame, batch[:, frame_idx]).mean().item()
                mc_psnr = mse2psnr(mc_mse)
                
                if self.output_nought:
                    nought = torch.zeros_like(coding_frame)
                    y_error = nn.MSELoss(reduction='none')(nought, y2).mean().item()
                    loss = self.args.lmda * mse + rate + 0.01 * self.args.lmda * y_error
                    y2_error_list.append(y_error)
                else:
                    mc_error = nn.MSELoss(reduction='none')(mc_frame, mc_hat).mean().item()
                    loss = self.args.lmda * mse + rate + 0.01 * self.args.lmda * mc_error
                    mc_error_list.append(mc_error)
                
                #loss = self.args.lmda * (mse + mc_mse) + rate

                mc_psnr_list.append(mc_psnr)
                m_rate_list.append(m_rate)
            else:
                with torch.no_grad():
                    rec_frame, likelihoods, _, _, _, _ = self.if_model(align.align(batch[:, frame_idx]))

                rec_frame = align.resume(rec_frame).clamp(0, 1)
                rate = trc.estimate_bpp(likelihoods, input=rec_frame).mean().item()

                mse = self.criterion(rec_frame, batch[:, frame_idx]).mean().item()
                if self.args.ssim:
                    psnr = mse
                else:
                    psnr = mse2psnr(mse)
                loss = self.args.lmda * mse + rate

                ref_frame = rec_frame

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
        if self.output_nought:
            y2_error = np.mean(y2_error_list)
        else:
            mc_error = np.mean(mc_error_list)

        if self.output_nought:
            logs = {'dataset_name': dataset_name, 'seq_name': seq_name, 
                    'val_loss': loss, 'val_mse': mse, 
                    'val_psnr': psnr, 'val_rate': rate, 
                    'val_mc_psnr': mc_psnr, 'val_m_rate': m_rate,
                    'val_y2_error': y2_error}
        else:
            logs = {'dataset_name': dataset_name, 'seq_name': seq_name, 
                    'val_loss': loss, 'val_mse': mse, 
                    'val_psnr': psnr, 'val_rate': rate, 
                    'val_mc_psnr': mc_psnr, 'val_m_rate': m_rate,
                    'val_mc_error': mc_error}

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
                if self.output_nought:
                    rd_dict[dataset_name]['y2_error'] = []
                else:
                    rd_dict[dataset_name]['mc_error'] = []
                    
            rd_dict[dataset_name]['psnr'].append(logs['val_psnr'])
            rd_dict[dataset_name]['rate'].append(logs['val_rate'])
            rd_dict[dataset_name]['mc_psnr'].append(logs['val_mc_psnr'])
            rd_dict[dataset_name]['m_rate'].append(logs['val_m_rate'])
            if self.output_nought:
                rd_dict[dataset_name]['y2_error'].append(logs['val_y2_error'])
            else:
                rd_dict[dataset_name]['mc_error'].append(logs['val_mc_error'])
   
            loss.append(logs['val_loss'])

        avg_loss = np.mean(loss)
        
        logs = {'val/loss': avg_loss}

        for dataset_name, rd in rd_dict.items():
            logs['val/'+dataset_name+' psnr'] = np.mean(rd['psnr'])
            logs['val/'+dataset_name+' rate'] = np.mean(rd['rate'])
            logs['val/'+dataset_name+' mc_psnr'] = np.mean(rd['mc_psnr'])
            logs['val/'+dataset_name+' m_rate'] = np.mean(rd['m_rate'])
            if self.output_nought:
                logs['val/'+dataset_name+' y2_error'] = np.mean(rd['y2_error'])
            else:
                logs['val/'+dataset_name+' mc_error'] = np.mean(rd['mc_error'])

        self.log_dict(logs)

        return None

    def test_step(self, batch, batch_idx):
        metrics_name = ['PSNR', 'Rate', 'Mo_Rate', 'MC-PSNR', 'MCrec-PSNR', 'MCerr-PSNR', 'BDQ-PSNR', 'QE-PSNR', 'back-PSNR',
                        'p1-PSNR', 'p1-BDQ-PSNR']
                
        metrics = {}
        for m in metrics_name:
            metrics[m] = []
        # PSNR: PSNR(gt, ADQ)
        # Rate
        # Mo_Rate: Motion Rate
        # MC-PSNR: PSNR(gt, mc_frame)
        # MCrec-PSNR: PSNR(gt, x_2)
        # MCerr-PSNR: PSNR(x_2, mc_frame)
        # BDQ-PSNR: PSNR(gt, BDQ)
        # QE-PSNR: PSNR(BDQ, ADQ)
        # back-PSNR: PSNR(mc_frame, BDQ)
        # p1-PSNR: PSNR(gt, ADQ) only when first P-frame in a GOP
        # p1-BDQ-PSNR: PSNR(gt, BDQ) only when first P-frame in a GOP
                
        dataset_name, seq_name, batch, frame_id_start = batch
        frame_id = int(frame_id_start)

        ref_frame = batch[:, 0] # Put reference frame in first position
        batch = batch[:, 1:] # GT
        seq_name = seq_name[0]
        dataset_name = dataset_name[0]

        gop_size = batch.size(1)

        height, width = ref_frame.size()[2:]
        estimate_bpp = partial(trc.estimate_bpp, num_pixels=height * width)


        psnr_list = []
        mc_psnr_list = []
        if not self.output_nought:
            mc_hat_psnr_list = []
        BDQ_psnr_list = []
        rate_list = []
        m_rate_list = []
        log_list = []
        align = trc.util.Alignment()

        for frame_idx in range(gop_size):
            ref_frame = ref_frame.clamp(0, 1)
            TO_VISUALIZE = (frame_id_start == 1 and frame_idx < 8 and seq_name in ['Beauty', 'Jockey', 'HoneyBee'])
            if frame_idx != 0:
                # reconstruced frame will be next ref_frame
                #if batch_idx % 100 == 0:
                if False and TO_VISUALIZE:
                    os.makedirs(os.path.join(self.args.save_dir, 'visualize_ANFIC', f'batch_{batch_idx}'), exist_ok=True)
                    if self.output_nought:
                        ref_frame, likelihoods, m_info, flow_hat, mc_frame, _, _, BDQ, y2 = self(
                                                                                                  align.align(ref_frame),
                                                                                                  align.align(batch[:, frame_idx]),
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
                        ref_frame, likelihoods, m_info, flow_hat, mc_frame, _, _, BDQ, mc_hat = self(
                                                                                                  align.align(ref_frame),
                                                                                                  align.align(batch[:, frame_idx]),
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
                    #continue
                    if self.output_nought:
                        ref_frame, likelihoods, m_info, flow_hat, mc_frame, _, _, BDQ, y2 = self(
                                                                                                  align.align(ref_frame),
                                                                                                  align.align(batch[:, frame_idx]),
                                                                                                  p_order=frame_idx
                                                                                                )
                    else:
                        ref_frame, likelihoods, m_info, flow_hat, mc_frame, _, _, BDQ, mc_hat = self(
                                                                                                  align.align(ref_frame),
                                                                                                  align.align(batch[:, frame_idx]),
                                                                                                  p_order=frame_idx
                                                                                                )
                    

                ref_frame = align.resume(ref_frame).clamp(0, 1)
                mc_frame = align.resume(mc_frame).clamp(0, 1)
                coding_frame = align.resume(batch[:, frame_idx]).clamp(0, 1)
                warp_frame = align.resume(m_info['warped']).clamp(0, 1)
                
                if self.output_nought:
                    y2 = align.resume(y2).clamp(0, 1)
                else:
                    mc_hat = align.resume(mc_hat).clamp(0, 1)
                
                BDQ = align.resume(BDQ).clamp(0, 1)

                os.makedirs(self.args.save_dir + f'/{seq_name}/flow', exist_ok=True)
                os.makedirs(self.args.save_dir + f'/{seq_name}/gt_frame', exist_ok=True)
                os.makedirs(self.args.save_dir + f'/{seq_name}/mc_frame', exist_ok=True)
                os.makedirs(self.args.save_dir + f'/{seq_name}/rec_frame', exist_ok=True)
                os.makedirs(self.args.save_dir + f'/{seq_name}/BDQ', exist_ok=True)
                os.makedirs(self.args.save_dir + f'/{seq_name}/warp', exist_ok=True)

                if self.output_nought:
                    os.makedirs(self.args.save_dir + f'/{seq_name}/y2', exist_ok=True)
                else:
                    os.makedirs(self.args.save_dir + f'/{seq_name}/mc_hat', exist_ok=True)

                if TO_VISUALIZE: 
                    flow_map = plot_flow(flow_hat)
                    save_image(flow_map, self.args.save_dir + f'/{seq_name}/flow/f{int(frame_idx)}_flow.png', nrow=1)

                    save_image(coding_frame[0], self.args.save_dir + f'/{seq_name}/gt_frame/'
                               f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(mc_frame[0], self.args.save_dir + f'/{seq_name}/mc_frame/'
                               f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(ref_frame[0], self.args.save_dir + f'/{seq_name}/rec_frame/'
                               f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(BDQ[0], self.args.save_dir + f'/{seq_name}/BDQ/'
                               f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(warp_frame[0], self.args.save_dir + f'/{seq_name}/warp/'
                               f'frame_{int(frame_id_start + frame_idx)}.png')
                    
                    if self.output_nought:
                        save_image(y2[0], self.args.save_dir + f'/{seq_name}/y2/'
                                    f'frame_{int(frame_id_start + frame_idx)}.png')
                    else:
                        save_image(mc_hat[0], self.args.save_dir + f'/{seq_name}/mc_hat/'
                                    f'frame_{int(frame_id_start + frame_idx)}.png')

                rate = trc.estimate_bpp(likelihoods, input=ref_frame).mean().item()
                mse = self.criterion(ref_frame, batch[:, frame_idx]).mean().item()
                if self.args.ssim:
                    psnr = mse
                else:
                    psnr = mse2psnr(mse)

                # likelihoods[0] & [1] are motion latent & hyper likelihood
                m_rate = trc.estimate_bpp(likelihoods[0], input=ref_frame).mean().item() + \
                         trc.estimate_bpp(likelihoods[1], input=ref_frame).mean().item()
                metrics['Mo_Rate'].append(m_rate)

                mc_psnr = mse2psnr(self.criterion(mc_frame, coding_frame).mean().item())
                metrics['MC-PSNR'].append(mc_psnr)
                
                if not self.output_nought:
                    mc_rec_psnr = mse2psnr(self.criterion(mc_hat, coding_frame).mean().item())
                    metrics['MCrec-PSNR'].append(mc_rec_psnr)
                    
                    mc_err_psnr = mse2psnr(self.criterion(mc_frame, mc_hat).mean().item())
                    metrics['MCerr-PSNR'].append(mc_err_psnr)
                
                BDQ_psnr = mse2psnr(self.criterion(BDQ, coding_frame).mean().item())
                metrics['BDQ-PSNR'].append(BDQ_psnr)

                QE_psnr = mse2psnr(self.criterion(BDQ, ref_frame).mean().item())
                metrics['QE-PSNR'].append(QE_psnr)
                
                back_psnr = mse2psnr(self.criterion(mc_frame, BDQ).mean().item())
                metrics['back-PSNR'].append(back_psnr)


                if frame_idx == 1:
                    metrics['p1-PSNR'].append(psnr)
                    metrics['p1-BDQ-PSNR'].append(BDQ_psnr)
                
                if self.output_nought:
                    log_list.append({'PSNR': psnr, 'Rate': rate, 'MC-PSNR': mc_psnr,
                                  'my': estimate_bpp(likelihoods[0]).item(), 'mz': estimate_bpp(likelihoods[1]).item(),
                                  'ry': estimate_bpp(likelihoods[2]).item(), 'rz': estimate_bpp(likelihoods[3]).item(),
                                  'BDQ-PSNR': BDQ_psnr})
                else:
                    log_list.append({'PSNR': psnr, 'Rate': rate, 'MC-PSNR': mc_psnr,
                                      'my': estimate_bpp(likelihoods[0]).item(), 'mz': estimate_bpp(likelihoods[1]).item(),
                                      'ry': estimate_bpp(likelihoods[2]).item(), 'rz': estimate_bpp(likelihoods[3]).item(),
                                      'MCerr-PSNR': mc_err_psnr, 'BDQ-PSNR': BDQ_psnr})
            else:
                with torch.no_grad():
                    rec_frame, likelihoods, _, _, _, _ = self.if_model(align.align(batch[:, frame_idx]))

                rec_frame = align.resume(rec_frame).clamp(0, 1)
                rate = trc.estimate_bpp(likelihoods, input=rec_frame).mean().item()

                mse = self.criterion(rec_frame, batch[:, frame_idx]).mean().item()
                psnr = mse2psnr(mse)

                ref_frame = rec_frame

                os.makedirs(self.args.save_dir + f'/{seq_name}/rec_frame', exist_ok=True)
                if TO_VISUALIZE:
                    save_image(rec_frame[0], self.args.save_dir + f'/{seq_name}/rec_frame/'
                               f'frame_{int(frame_id_start + frame_idx)}.png')

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
        current_epoch = self.trainer.current_epoch
        
        lr_step = []
        for k, v in phase.items():
            if 'RNN' in k and v > current_epoch: 
                lr_step.append(v-current_epoch)
        lr_gamma = 0.5

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

        dataset_root = os.getenv('DATAROOT')
        qp = {256: 37, 512: 32, 1024: 27, 2048: 22}[self.args.lmda]

        if stage == 'fit':
            transformer = transforms.Compose([
                transforms.RandomCrop((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])

            self.train_dataset = VideoDataIframe(dataset_root + "/vimeo_septuplet/", 'BPG_QP' + str(qp), 7,
                                                 transform=transformer)
            # self.val_dataset = VideoTestDataIframe(dataset_root + "TestVideo", self.args.lmda, first_gop=True)
            self.val_dataset = VideoTestDataIframe(dataset_root, self.args.lmda, first_gop=True)

        elif stage == 'test':
            # self.test_dataset = VideoTestDataIframe(dataset_root + "TestVideo", self.args.lmda, sequence=('U', 'B'))
            # self.test_dataset = VideoTestDataIframe(dataset_root, self.args.lmda, sequence=('U', 'B', 'M'))
            self.test_dataset = VideoTestDataIframe(dataset_root, self.args.lmda, sequence=('U', 'B'), GOP=32)
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
        parser.add_argument('--hidden_channels', default=8, type=int)
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

    #save_root = "/work/u4803414/torchDVC/"
    save_root = os.getenv('LOG', './') + '/torchDVC/'

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

    parser.add_argument('--restore', type=str, choices=['none', 'resume', 'load', 'custom', 'finetune'], default='none')
    parser.add_argument('--restore_exp_key', type=str, default=None)
    parser.add_argument('--restore_exp_epoch', type=int, default=49)
    parser.add_argument('--test', "-T", action="store_true")
    parser.add_argument('--experiment_name', type=str, default='basic')
    parser.add_argument('--project_name', type=str, default="CANFVC+")

    parser.add_argument('--MENet', type=str, choices=['PWC', 'SPy'], default='PWC')
    parser.add_argument('--motion_coder_conf', type=str, default=None)
    parser.add_argument('--residual_coder_conf', type=str, default=None)
    parser.add_argument('--prev_motion_coder_conf', type=str, default=None)
    parser.add_argument('--prev_residual_coder_conf', type=str, default=None)

    parser.set_defaults(gpus=1)

    # parse params
    args = parser.parse_args()

    experiment_name = args.experiment_name
    project_name = args.project_name

    torch.backends.cudnn.deterministic = True
 
    # I-frame coder ckpt
    ANFIC_code = {2048: '0821_0300', 1024: '0530_1212', 512: '0530_1213', 256: '0530_1215'}[args.lmda]

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
    
    assert not (args.residual_coder_conf is None)
    res_coder_cfg = yaml.safe_load(open(args.residual_coder_conf, 'r'))
    assert res_coder_cfg['model_architecture'] in trc.__CODER_TYPES__.keys()
    res_coder_arch = trc.__CODER_TYPES__[res_coder_cfg['model_architecture']]
    res_coder = res_coder_arch(**res_coder_cfg['model_params'])

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        save_last=True,
        #every_n_epochs=10,
        period=1,
        verbose=True,
        monitor='val/loss',
        mode='min',
        prefix=''
    )


    db = None
    if args.gpus > 1:
        db = 'ddp'

    comet_logger = CometLogger(
        api_key="sriOLxa6VvcxCPgGaKaaxAk0p",
        project_name=project_name,
        workspace="hongsheng416",
        experiment_name=experiment_name + "-" + str(args.lmda),
        experiment_key = args.restore_exp_key if args.restore == 'resume' else None,
        disabled=args.test or args.debug
    )
    
    args.save_dir = os.path.join(save_root, project_name, experiment_name + '-' + str(args.lmda))

    if args.restore == 'resume' or args.restore == 'finetune':
        # trainer = Trainer.from_argparse_args(args,
        #                                      checkpoint_callback=checkpoint_callback,
        #                                      gpus=args.gpus,
        #                                      distributed_backend=db,
        #                                      logger=comet_logger,
        #                                      default_root_dir=save_root,
        #                                      check_val_every_n_epoch=1,
        #                                      num_sanity_val_steps=1,
        #                                      terminate_on_nan=True)

        trainer = Trainer.from_argparse_args(args,
                                             checkpoint_callback=checkpoint_callback,
                                             gpus=args.gpus,
                                             distributed_backend=db,
                                             logger=comet_logger,
                                             default_root_dir=save_root,
                                             check_val_every_n_epoch=1,
                                             num_sanity_val_steps=1,
                                             terminate_on_nan=True,
                                            #  limit_train_batches=0.5
                                             )

        epoch_num = args.restore_exp_epoch
        if args.restore_exp_key is None:
            raise ValueError
        else:  # When prev_exp_key is specified in args
            checkpoint = torch.load(os.path.join(save_root, project_name, args.restore_exp_key, "checkpoints",
                                                 f"epoch={epoch_num}.ckpt"),
                                    map_location=(lambda storage, loc: storage))

        trainer.current_epoch = epoch_num + 1
        
        model = Pframe(args, mo_coder, res_coder).cuda()
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    
    elif args.restore == 'load':
        trainer = Trainer.from_argparse_args(args,
                                             checkpoint_callback=checkpoint_callback,
                                             gpus=args.gpus,
                                             distributed_backend=db,
                                             logger=comet_logger,
                                             default_root_dir=save_root,
                                             check_val_every_n_epoch=1,
                                             num_sanity_val_steps=0,
                                             terminate_on_nan=True)

        
        epoch_num = args.restore_exp_epoch
        if args.restore_exp_key is None:
            raise ValueError
        else:  # When prev_exp_key is specified in args
            checkpoint = torch.load(os.path.join(save_root, project_name, args.restore_exp_key, "checkpoints",
                                                 f"epoch={epoch_num}.ckpt"),
                                    map_location=(lambda storage, loc: storage))

        trainer.current_epoch = phase['trainAll_2frames'] - 2
        # trainer.current_epoch = phase['trainMV'] + 1
        #trainer.current_epoch = 29
        #trainer.current_epoch = epoch_num + 1

        coder_ckpt = torch.load(os.path.join(os.getenv('LOG', './'), f"ANFIC/ANFHyperPriorCoder_{ANFIC_code}/model.ckpt"),
                                map_location=(lambda storage, loc: storage))['coder']

        for k, v in coder_ckpt.items():
            key = 'if_model.' + k
            checkpoint['state_dict'][key] = v


        model = Pframe(args, mo_coder, res_coder).cuda()
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    elif args.restore == 'custom':
        trainer = Trainer.from_argparse_args(args,
                                             checkpoint_callback=checkpoint_callback,
                                             gpus=args.gpus,
                                             distributed_backend=db,
                                             logger=comet_logger,
                                             default_root_dir=save_root,
                                             check_val_every_n_epoch=1,
                                             num_sanity_val_steps=-1,
                                             #terminate_on_nan=True
                                             )
        
        epoch_num = args.restore_exp_epoch
        if args.restore_exp_key is None:
            raise ValueError
        else:  # When prev_exp_key is specified in args
            checkpoint = torch.load(os.path.join(save_root, project_name, args.restore_exp_key, "checkpoints",
                                                 f"epoch={epoch_num}.ckpt"),
                                    map_location=(lambda storage, loc: storage))

        # trainer.current_epoch = phase['trainMV'] + 1
        # trainer.current_epoch = phase['trainMC'] + 1
        # trainer.current_epoch = phase['trainRes_2frames'] + 1
        trainer.current_epoch = phase['trainRes_fullgop'] + 1
        

        # Previous coders
        #assert not (args.prev_motion_coder_conf is None)
        #prev_mo_coder_cfg = yaml.safe_load(open(args.prev_motion_coder_conf, 'r'))
        #assert prev_mo_coder_cfg['model_architecture'] in trc.__CODER_TYPES__.keys()
        #prev_mo_coder_arch = trc.__CODER_TYPES__[prev_mo_coder_cfg['model_architecture']]
        #prev_mo_coder = prev_mo_coder_arch(**prev_mo_coder_cfg['model_params'])
        #
        #assert not (args.prev_residual_coder_conf is None)
        #prev_res_coder_cfg = yaml.safe_load(open(args.prev_residual_coder_conf, 'r'))
        #assert prev_res_coder_cfg['model_architecture'] in trc.__CODER_TYPES__.keys()
        #prev_res_coder_arch = trc.__CODER_TYPES__[prev_res_coder_cfg['model_architecture']]
        #prev_res_coder = prev_res_coder_arch(**prev_res_coder_cfg['model_params'])
        
        from collections import OrderedDict
        new_ckpt = OrderedDict()
    
        for k, v in checkpoint['state_dict'].items():
            #if k[:5] == 'MENet':
            #    new_ckpt[k] = v
            if k[:6] == 'Motion':
                new_ckpt[k] = v
            elif k[:8] == 'Residual':
                new_ckpt[k] = v
            elif k[:5] == 'MCNet':
                new_ckpt[k] = v
            elif k[:9] == 'mc_decode':
                new_ckpt[k] = v
        
        # checkpoint = torch.load(os.path.join(save_root, project_name, 'a71d22faf87f45b2bc2216a8296b4bf4', "checkpoints",
        #                                          "epoch=22.ckpt"),
        #                             map_location=(lambda storage, loc: storage))
        
        # for k, v in checkpoint['state_dict'].items():
        #     if k[:5] == 'MCNet':
        #         new_ckpt[k] = v
        
 
        coder_ckpt = torch.load(os.path.join(os.getenv('LOG', './'), f"ANFIC/ANFHyperPriorCoder_{ANFIC_code}/model.ckpt"),
                                map_location=(lambda storage, loc: storage))['coder']

        for k, v in coder_ckpt.items():
            key = 'if_model.' + k
            new_ckpt[key] = v

  
        model = Pframe(args, mo_coder, res_coder).cuda()
        model.load_state_dict(new_ckpt, strict=False)
        
        #summary(model)
        #print(model.Motion)
        #summary(model.Motion)
        #summary(model.Residual)
        #print(model.Residual)
    
    else:
        trainer = Trainer.from_argparse_args(args,
                                             checkpoint_callback=checkpoint_callback,
                                             gpus=args.gpus,
                                             distributed_backend=db,
                                             logger=comet_logger,
                                             default_root_dir=save_root,
                                             check_val_every_n_epoch=1,
                                             num_sanity_val_steps=0,
                                             terminate_on_nan=True)
        
        model = Pframe(args, mo_coder, res_coder).cuda()

        from collections import OrderedDict
        new_ckpt = OrderedDict()
        coder_ckpt = torch.load(os.path.join(os.getenv('LOG', './'), f"ANFIC/ANFHyperPriorCoder_{ANFIC_code}/model.ckpt"),
                                map_location=(lambda storage, loc: storage))['coder']

        for k, v in coder_ckpt.items():
            key = 'if_model.' + k
            new_ckpt[key] = v

        model.load_state_dict(new_ckpt, strict=False)

        #summary(model.Residual)
        #print(model.Residual)
        #summary(model.Motion)
        #print(model.Motion)

    if args.test:
        trainer.test(model)
    else:
        trainer.fit(model)
