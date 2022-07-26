import argparse
import os
import csv
from functools import partial

import flowiz as fz
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
from torchvision.utils import make_grid

from dataloader import VideoDataIframe, VideoTestDataIframe
from flownets import PWCNet
from models import Refinement, MMC_Net, Feature_Extractor_RGB
from util.psnr import mse2psnr
from util.sampler import Resampler
from util.ssim import MS_SSIM
from util.vision import PlotFlow, save_image

# phase = {'init': 0, 'trainMV': 40000, 'trainMC': 100000, 'trainRNN': 200000, 'trainALL': 1000000}
# phase = {'init': 0, 'trainMV': 40000, 'trainMC': 100000, 'trainRNN': 4420000, 'trainALL': 10000000}

# This script should start from residual net training
phase = {'init': 0, 'trainMV': 10, 'trainMC': 20, 'trainRes': 30, 'trainALL_2frames': 50, 'trainALL_fullgop': 100}

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
    """torchDVC Pframe"""
    train_dataset: VideoDataIframe
    val_dataset: VideoTestDataIframe
    test_dataset: VideoTestDataIframe

    def __init__(self, args, coder, ar_coder):
        super(Pframe, self).__init__()
        self.args = args
        self.criterion = nn.MSELoss(reduction='none') if not self.args.ssim else MS_SSIM(data_range=1.).cuda()

        self.MENet = PWCNet()

        self.Motion = coder(in_channels=2, out_channels=2, kernel_size=3)

        self.Resampler = Resampler()
        self.MCNet = Refinement(6, 64)

        self.Residual = coder()

        ##################
        self.f_extractor = Feature_Extractor_RGB(3, 32)

        self.MMC_Net = MMC_Net(32, 64, num_MMC_frames=args.num_MMC_frames)
        self.num_MMC_frames = args.num_MMC_frames

    def load_args(self, args):
        self.args = args

    def motion_forward(self, ref_frame, coding_frame):
        flow = self.MENet(ref_frame, coding_frame)

        flow_hat, likelihood_m = self.Motion(flow)

        mc = self.mc_net_forward(ref_frame, flow_hat)
        return mc, {'m_likelihood': likelihood_m, 'flow_hat': flow_hat}

    def forward(self, ref_frame, coding_frame, ref_queue, flow_queue, p_order=0):
                
        mc, m_info = self.motion_forward(ref_frame, coding_frame)
        
        if p_order == 1: # Fill queues with ref_frame & flow_hat when frame_1
            for cnt in range(self.num_MMC_frames):
                ref_queue.append(self.f_extractor(ref_frame))
                flow_queue.append(m_info['flow_hat'])


        # Update queue
        ref_queue.pop(0) # head is the farthest frame
        flow_queue.pop(0)

        ref_queue.append(self.f_extractor(ref_frame))
        flow_queue.append(m_info['flow_hat'])

        predicted, intra_info, likelihood_i = mc, 0, ()

        # Estimate MMC frame
        mmc = self.MMC_Net(mc, ref_queue, flow_queue, current_order=p_order)
        
        # Residual coding
        res = coding_frame - mmc
        res_hat, likelihood_r = self.Residual(res)

        reconstructed = mmc + res_hat

        likelihoods = m_info['m_likelihood'] + likelihood_i + likelihood_r

        return reconstructed, likelihoods, m_info['flow_hat'], mmc, mc, predicted, intra_info, res, res_hat, ref_queue, flow_queue

    def training_step(self, batch, batch_idx):
        epoch = self.current_epoch

        batch = batch.cuda()
        ref_frame = batch[:, 0]

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
            mc = self.mc_net_forward(ref_frame, flow_hat)
            
            # Estimate MMC frame
            ref_queue = []
            flow_queue = []
            
            for cnt in range(self.num_MMC_frames):
                ref_queue.append(self.f_extractor(ref_frame))
                flow_queue.append(flow_hat.detach())

            mmc = self.MMC_Net(mc, ref_queue, flow_queue, warp_flows=True, current_order=1)
            
            reconstructed = mmc
            likelihoods = likelihood_m

            distortion = self.criterion(coding_frame, reconstructed)
            rate = trc.estimate_bpp(likelihoods, input=coding_frame)

            loss = self.args.lmda * distortion.mean() + rate.mean()
            logs = {'train/loss': loss.item(),
                    'train/distortion': distortion.mean().item(),
                    'train/rate': rate.mean().item()}
        else:
            if epoch <= phase['trainRes']:
                _phase = 'RES'
            else:
                _phase = 'ALL'

            reconstructed = ref_frame
            
            ref_queue = []
            flow_queue = []

            loss = torch.tensor(0., dtype=torch.float, device=reconstructed.device)
            dist_list = []
            rate_list = []
            mmc_psnr_list = []
            warping_loss_list = []

            for frame_idx in range(1, 7):
                ref_frame = reconstructed
                
                if frame_idx > 1:
                    if epoch <= phase['trainALL_2frames']: # 2-frame training stages
                        break

                ref_frame = ref_frame.detach() # Detach when RNN training
                coding_frame = batch[:, frame_idx]

                if _phase == 'RES': # Train res_coder only
                    with torch.no_grad():
                        flow = self.MENet(ref_frame, coding_frame)
                        flow_hat, likelihood_m = self.Motion(flow)
                        mc = self.mc_net_forward(ref_frame, flow_hat)

                        warping_loss = self.criterion(coding_frame, mc.detach())

                        if frame_idx == 1: # Fill queues with ref_frame & flow_hat when frame_1
                            for cnt in range(self.num_MMC_frames):
                                ref_queue.append(self.f_extractor(ref_frame))
                                flow_queue.append(flow_hat.detach())
                        
                        # Update queue
                        ref_queue.pop(0) # head is the farthest frame
                        flow_queue.pop(0)

                        # Put mc frame into last ; it's no need to warp it again
                        ref_queue.append(self.f_extractor(ref_frame))
                        flow_queue.append(flow_hat.detach())
                        
                        # Estimate MMC frame
                        mmc = self.MMC_Net(mc, ref_queue, flow_queue, warp_flows=True, current_order=frame_idx)


                else:
                    flow = self.MENet(ref_frame, coding_frame)
                    flow_hat, likelihood_m = self.Motion(flow)
                    mc = self.mc_net_forward(ref_frame, flow_hat)

                    warping_loss = self.criterion(coding_frame, mc.detach())

                    if frame_idx == 1: # Fill queues with ref_frame & flow_hat when frame_1
                        for cnt in range(self.num_MMC_frames):
                            ref_queue.append(self.f_extractor(ref_frame))
                            flow_queue.append(flow_hat.detach())
                    
                    # Update queue
                    ref_queue.pop(0) # head is the farthest frame
                    flow_queue.pop(0)

                    # Put mc frame into last ; it's no need to warp it again
                    ref_queue.append(self.f_extractor(ref_frame))
                    flow_queue.append(flow_hat.detach())
                    
                    # Estimate MMC frame
                    mmc = self.MMC_Net(mc, ref_queue, flow_queue, warp_flows=True, current_order=frame_idx)

                # Residual coding
                res = coding_frame - mmc
                res_hat, likelihood_r = self.Residual(res)

                reconstructed = mmc + res_hat
                likelihoods = likelihood_m + likelihood_r

                distortion = self.criterion(coding_frame, reconstructed)
                rate = trc.estimate_bpp(likelihoods, input=coding_frame)

                loss += self.args.lmda * distortion.mean() + rate.mean()
                dist_list.append(distortion.mean())
                warping_loss_list.append(warping_loss.mean())
                rate_list.append(rate.mean())
                
            distortion = torch.mean(torch.tensor(dist_list))
            warping_loss = torch.mean(torch.tensor(warping_loss_list))
            rate = torch.mean(torch.tensor(rate_list))
            logs = {
                    'train/loss': loss.item(), 
                    'train/distortion': distortion.item(), 
                    'train/PSNR': mse2psnr(distortion.item()), 
                    'train/rate': rate.item(), 
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

        seq_name, batch, frame_id_start = batch

        ref_frame = batch[:, 0]
        batch = batch[:, 1:]
        seq_name = seq_name[0]

        gop_size = batch.size(1)

        height, width = ref_frame.size()[2:]

        psnr_list = []
        mc_psnr_list = []
        mse_list = []
        rate_list = []
        m_rate_list = []
        loss_list = []
        align = trc.util.Alignment()

        # For MMC_Net
        ref_queue = []
        flow_queue = []

        for frame_idx in range(gop_size):
            if frame_idx != 0:
                rec_frame, likelihoods, flow_hat, mmc, mc_frame, _, _, res, res_hat,ref_queue, flow_queue = self(align.align(ref_frame),
                                                                                                                    align.align(batch[:, frame_idx]),
                                                                                                                    ref_queue,
                                                                                                                    flow_queue,
                                                                                                                    p_order=frame_idx,
                                                                                                               )
                                        
                ref_frame = align.resume(ref_frame).clamp(0, 1)
                rec_frame = align.resume(rec_frame).clamp(0, 1)
                coding_frame = align.resume(batch[:, frame_idx]).clamp(0, 1)
                mmc = align.resume(mmc).clamp(0, 1)
                mc_frame = align.resume(mc_frame).clamp(0, 1)
                res = align.resume(res).clamp(0, 1)
                res_hat = align.resume(res_hat).clamp(0, 1)

                rate = trc.estimate_bpp(likelihoods, input=ref_frame).mean().item()
                m_rate = trc.estimate_bpp(likelihoods[0], input=ref_frame).mean().item() + \
                         trc.estimate_bpp(likelihoods[1], input=ref_frame).mean().item()


                if frame_idx == 5:
                    mse = torch.mean((rec_frame - coding_frame).pow(2))
                    psnr = get_psnr(mse).cpu().item()

                    #flow_map = plot_flow(flow_hat.to('cuda:0'))
                    upload_img(ref_frame.cpu().numpy()[0], f'{seq_name}_ref_frame.png', grid=False)
                    upload_img(coding_frame.cpu().numpy()[0], f'{seq_name}_gt_frame.png', grid=False)
                    upload_img(mmc.cpu().numpy()[0], f'{seq_name}_MMC_frame.png', grid=False)
                    upload_img(mc_frame.cpu().numpy()[0], f'{seq_name}_MC_frame.png', grid=False)
                    upload_img(rec_frame.cpu().numpy()[0],
                               seq_name + '_rec_frame_{:.3f}.png'.format(psnr),
                               grid=False)
                    upload_img(res.cpu().numpy()[0], f'{seq_name}_residual_frame.png', grid=False)
                    upload_img(res_hat.cpu().numpy()[0], f'{seq_name}_residual_frame_Q.png', grid=False)
                   
                    flow_rgb = torch.from_numpy(
                                    fz.convert_from_flow(flow_hat[0].permute(1, 2, 0).cpu().numpy()) / 255).permute(2, 0, 1)
                    upload_img(flow_rgb.cpu().numpy(), f'{seq_name}_flow_map.png', grid=False)


                ref_frame = rec_frame

                mse = self.criterion(ref_frame, batch[:, frame_idx]).mean().item()
                psnr = mse2psnr(mse)
                mc_mse = self.criterion(mc_frame, batch[:, frame_idx]).mean().item()
                mc_psnr = mse2psnr(mc_mse)
                loss = self.args.lmda * mse + rate

                mc_psnr_list.append(mc_psnr)
                m_rate_list.append(m_rate)

            else:
                intra_index = {256: 3, 512: 2, 1024: 1, 2048: 0}[self.args.lmda]

                rate = iframe_byte[seq_name][intra_index] * 8 / height / width

                mse = self.criterion(ref_frame, batch[:, frame_idx]).mean().item()
                psnr = mse2psnr(mse)
                loss = self.args.lmda * mse + rate
            

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
        metrics_name = ['PSNR', 'Rate', 'Mo_Rate', 'MC-PSNR', 'MMC-PSNR']
        metrics = {}
        for m in metrics_name:
            metrics[m] = []
        # PSNR: PSNR(gt, ADQ)
        # Rate
        # Mo_Rate: Motion Rate
        # MMC-PSNR: PSNR(gt, mmc)
        # MC-PSNR: PSNR(gt, mc_frame)
                
        seq_name, batch, frame_id_start = batch
        frame_id = int(frame_id_start)

        ref_frame = batch[:, 0] # Put reference frame in first position
        batch = batch[:, 1:] # GT
        seq_name = seq_name[0]

        gop_size = batch.size(1)

        height, width = ref_frame.size()[2:]
        estimate_bpp = partial(trc.estimate_bpp, num_pixels=height * width)


        psnr_list = []
        mc_psnr_list = []
        rate_list = []
        m_rate_list = []
        mmc_psnr_list = []
        log_list = []
        align = trc.util.Alignment()


        # For MMC_Net
        ref_queue = []
        flow_queue = []

        for frame_idx in range(gop_size):
            if frame_idx != 0:
                # reconstruced frame will be next ref_frame
                ref_frame, likelihoods, flow_hat, mmc, mc_frame, _, _, res, res_hat,ref_queue, flow_queue = self(align.align(ref_frame),
                                                                                                                 align.align(batch[:, frame_idx]),
                                                                                                                 ref_queue,
                                                                                                                 flow_queue,
                                                                                                                 p_order=frame_idx,
                                                                                                                )

                ref_frame = align.resume(ref_frame)
                mmc = align.resume(mmc)
                mc_frame = align.resume(mc_frame)
                coding_frame = align.resume(batch[:, frame_idx])

                os.makedirs(self.args.save_dir + f'/{seq_name}/flow', exist_ok=True)
                os.makedirs(self.args.save_dir + f'/{seq_name}/gt_frame', exist_ok=True)
                os.makedirs(self.args.save_dir + f'/{seq_name}/mmc', exist_ok=True)
                os.makedirs(self.args.save_dir + f'/{seq_name}/mc_frame', exist_ok=True)
                os.makedirs(self.args.save_dir + f'/{seq_name}/rec_frame', exist_ok=True)

                if frame_id_start < 8 and seq_name in ['BasketballDrive', 'Kimono1', 'HoneyBee', 'Jockey']:
                    flow_map = plot_flow(flow_hat)
                    save_image(flow_map, self.args.save_dir + f'/{seq_name}/flow/f{int(frame_idx)}_flow.png', nrow=1)

                    save_image(coding_frame[0], self.args.save_dir + f'/{seq_name}/gt_frame/'
                               f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(mmc[0], self.args.save_dir + f'/{seq_name}/mmc/'
                               f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(mc_frame[0], self.args.save_dir + f'/{seq_name}/mc_frame/'
                               f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(ref_frame[0], self.args.save_dir + f'/{seq_name}/rec_frame/'
                               f'frame_{int(frame_id_start + frame_idx)}.png')

                rate = trc.estimate_bpp(likelihoods, input=ref_frame).mean().item()
                mse = self.criterion(ref_frame, batch[:, frame_idx]).mean().item()
                psnr = mse2psnr(mse)

                # likelihoods[0] & [1] are motion latent & hyper likelihood
                m_rate = trc.estimate_bpp(likelihoods[0], input=ref_frame).mean().item() + \
                         trc.estimate_bpp(likelihoods[1], input=ref_frame).mean().item()
                metrics['Mo_Rate'].append(m_rate)

                mc_psnr = mse2psnr(self.criterion(mc_frame, coding_frame).mean().item())
                metrics['MC-PSNR'].append(mc_psnr)
               
                mmc_psnr = mse2psnr(self.criterion(mmc, coding_frame).mean().item())
                metrics['MMC-PSNR'].append(mmc_psnr)


                log_list.append({'PSNR': psnr, 'Rate': rate, 'MMC-PSNR': mmc_psnr, 'MC-PSNR': mc_psnr,
                                 'my': estimate_bpp(likelihoods[0]).item(), 'mz': estimate_bpp(likelihoods[1]).item(),
                                 'ry': estimate_bpp(likelihoods[2]).item(), 'rz': estimate_bpp(likelihoods[3]).item(),
                                })

            else:
                qp = {256: 37, 512: 32, 1024: 27, 2048: 22}[self.args.lmda]
                dataset_root = os.getenv('DATASET')

                # Read the binary files directly for accurate bpp estimate.
                size_byte = os.path.getsize(f'{dataset_root}/TestVideo/bpg/{qp}/bin/{seq_name}/frame_{frame_id}.bin')
                rate = size_byte * 8 / height / width
                mse = self.criterion(ref_frame, batch[:, frame_idx]).mean().item()
                psnr = mse2psnr(mse)

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

                #writer.writerow(['frame', 'PSNR', 'total bits', 'MC-PSNR', 'my', 'mz', 'ry', 'rz', 'MCerr-PSNR', 'BDQ-PSNR'])

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

    def mc_net_forward(self, ref_frame, coding_frame, return_warped=False):
        warped = self.Resampler(ref_frame, coding_frame)
        if self.MCNet is not None:
            mc_net_input = [ref_frame, warped]

            mc_frame = self.MCNet(*mc_net_input)
        else:
            mc_frame = warped

        if return_warped:
            return mc_frame, warped
        else:
            return mc_frame

    def compress(self, ref_frame, coding_frame, p_order):
        # TODO: Modify to match operation as forward()
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
        # TODO: Modify to match operation as forward()
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
            self.val_dataset = VideoTestDataIframe(dataset_root + "TestVideo", self.args.lmda, first_gop=True)

        elif stage == 'test':
            self.test_dataset = VideoTestDataIframe(dataset_root + "TestVideo", self.args.lmda)

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

        parser.add_argument('--num_MMC_frames', default=3, type=int)

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
    parser.add_argument('--restore', type=str, choices=['none', 'resume', 'load', 'load_from_DVC'], default='none')
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
        period=3, # Save at least every 5 epochs
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
            experiment_key=args.restore_exp_key,
            disabled=args.test or args.debug
        )
        args.save_dir = os.path.join(save_root, project_name, experiment_name + '-' + str(args.lmda))

        trainer = Trainer.from_argparse_args(args,
                                             checkpoint_callback=checkpoint_callback,
                                             gpus=args.gpus,
                                             distributed_backend=db,
                                             logger=comet_logger,
                                             default_root_dir=save_root,
                                             check_val_every_n_epoch=3,
                                             num_sanity_val_steps=0,
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
        trainer.current_epoch = epoch_num + 1

        model = Pframe(args, res_coder, pred_coder).cuda()
        
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
                                             check_val_every_n_epoch=3,
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
        trainer.current_epoch = phase['trainRes'] + 1
        
        model = Pframe(args, res_coder, pred_coder).cuda()
        model.load_state_dict(checkpoint['state_dict'], strict=True)


    elif args.restore == 'load_from_DVC':
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
                                             check_val_every_n_epoch=3,
                                             num_sanity_val_steps=-1,
                                             terminate_on_nan=True,
                                             #automatic_optimization=False # For manual backward
                                             )

        checkpoint = torch.load(os.path.join(save_root, 'DVC_baseline', 'base_model_{}.ckpt'.format(args.lmda)),
                                map_location=(lambda storage, loc: storage))

        #trainer.global_step = 4400000
        trainer.current_epoch = phase['trainMV'] + 1

        model = Pframe(args, res_coder, pred_coder).cuda()
        model.load_state_dict(checkpoint['model'], strict=False)

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
                                             num_sanity_val_steps=-1,
                                             terminate_on_nan=True,
                                             #automatic_optimization=False # For manual backward 
                                             )

        model = Pframe(args, res_coder, pred_coder).cuda()

    # comet_logger.experiment.log_code(file_name='torchDVC.py')

    if args.test:
        trainer.test(model)
    else:
        trainer.fit(model)
