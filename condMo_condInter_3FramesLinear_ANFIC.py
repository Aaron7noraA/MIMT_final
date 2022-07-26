import argparse
import os
import csv
from functools import partial

import yaml
import comet_ml
import flowiz as fz
import numpy as np
import torch
import torch_compression as trc

from torchinfo import summary
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from torch import nn, optim
from torch.utils.data import DataLoader
from torch_compression.modules.entropy_models import EntropyBottleneck
from torch_compression.hub import AugmentedNormalizedFlowHyperPriorCoder
from torchvision import transforms
from torchvision.utils import make_grid

from dataloader import VideoDataIframe, VideoTestDataIframe
from flownets import PWCNet, SPyNet
from SDCNet import SDCNet_3M
from models import ShortCutRefinement
from util.psnr import mse2psnr
from util.sampler import Resampler
from util.ssim import MS_SSIM
from util.vision import PlotFlow, PlotHeatMap, save_image

plot_flow = PlotFlow().cuda()
plot_bitalloc = PlotHeatMap("RB").cuda()

phase = {'trainMC': 20, 
         #'trainRes_2frames': 25, 
         'trainRes_2frames': 22, 
         #'trainAll_2frames': 30, 
         'trainAll_2frames': 25, 
         'trainAll_fullgop': 30, 
         'trainAll_RNN_1': 33, 
         'trainAll_RNN_2': 36, 
         'trainAll_RNN_3': 38}


# Custom pytorch-lightning trainer ; provide feature that configuring trainer.current_epoch
## Useless currently
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
    def __init__(self, args, mo_coder, cond_mo_coder, res_coder):
        super(Pframe, self).__init__()
        self.args = args
        self.criterion = nn.MSELoss(reduction='none') if not self.args.ssim else MS_SSIM(data_range=1.).cuda()

        self.if_model = AugmentedNormalizedFlowHyperPriorCoder(128, 320, 192, num_layers=2, use_DQ=True, use_code=False,
                                                               use_context=True, condition='GaussianMixtureModel',
                                                               quant_mode='RUN')
        if self.args.MENet == 'PWC':
            self.MENet = PWCNet(trainable=False)
        elif self.args.MENet == 'SPy':
            self.MENet = SPyNet(trainable=False)

        self.MWNet = SDCNet_3M(sequence_length=3) # Motion warping network
        self.MWNet.__delattr__('flownet')

        self.Motion = mo_coder
        self.CondMotion = cond_mo_coder

        self.Resampler = Resampler()
        #self.MCNet = ShortCutRefinement(15, 3, 64, 6)

        self.Residual = res_coder
        self.frame_buffer = list()
        self.flow_buffer = list()

        self.update = True

    def load_args(self, args):
        self.args = args

    def warp_multiple_frames(self, ref_queue, flow_queue, warp_flows=True):
        '''
            args: 
                ref_queue: previous frames
                flow_queue: previous flows of adjacent frames
                warp_flow: to warp flows before warp frames with them or not
        '''

        # Warp the flows, and warp the frames, and put them into another queue
        ## Tail frame is the closest, so it it warped first
        warped_frame_queue = []
        accumulated_flows = torch.zeros_like(flow_queue[-1])

        ## Warp in reverse order
        for i in range(len(ref_queue)):
            # Warp the flow
            if warp_flows:
                current_flow = self.Resampler(flow_queue[-(i+1)], accumulated_flows)

                # Warp the frame
                warped_frame = self.Resampler(ref_queue[-(i+1)], current_flow + accumulated_flows)

                # For P-frame index < num_MMC_frames, some flows need not to be accumulated
                #if current_order - i > 1:
                accumulated_flows += current_flow
            else:
                current_flow = flow_queue[-(i+1)]
                accumulated_flows = current_flow
            
                # Warp the frame
                warped_frame = self.Resampler(ref_queue[-(i+1)], accumulated_flows)

            warped_frame_queue.append(warped_frame)
        
        # Make the order be consistent ; that is, keep warped_frame_queue as reverse orfer
        warped_frame_queue = warped_frame_queue[::-1]

        return warped_frame_queue

    def lstsq(self, b, A):
        try:
            inv = torch.inverse(torch.mm(A.T, A))
            solution = torch.mm(torch.mm(inv, A.T), b)
            return solution
        except:
            print(A)
            return torch.zeros((A.shape[0]))

    def lstsq_multi_reference(self, reference_frame_list, target_frame):
        batch_size = target_frame.size(0)
        solutions = []
        for bs in range(batch_size):
            A = []
            # detect zero image
            ## calculate least-sequare solution only if the reference frame is not zero img
            is_zero_img = []
            for idx, frame in enumerate(reference_frame_list):
                #is_zero_img.append(len(torch.nonzero(frame[bs, :])) == 0)
                is_zero_img.append(torch.sum(frame[bs, :]) < 1e-15)
                if not is_zero_img[-1]:
                   A.append(frame[bs, :].view(-1))

            if len(A) > 0:
                A = torch.stack(A).T
                b = target_frame[bs, :].reshape(-1, 1)
                #solutions.append(torch.lstsq(b, A).solution[:A.size(1)].detach())
                solution = self.lstsq(b, A).detach()
            
            solution_idx = 0
            weight = torch.zeros((1, len(reference_frame_list))).cuda().T
            for idx in range(len(reference_frame_list)):
                if not is_zero_img[idx]:
                    weight[idx] = solution[solution_idx]
                    solution_idx += 1
            if len(A) == 0:
                weight[-1] += 1
                print('\nall reference images are zero!')
                print('weight =', weight)
                print(is_zero_img)
                for idx, frame in enumerate(reference_frame_list):
                    print(f'nonzeros in ref_frame_{idx} = ', len(torch.nonzero(frame[bs, :])))
                    print('\ttorch.sum(frame_{idx}) =', torch.sum(frame[bs, :]))
            solutions.append(weight)
        return solutions

    def motion_forward(self, ref_frame, coding_frame, visual=False, visual_prefix='', p_order=1):
        predict = p_order != 1

        if predict:
            assert len(self.frame_buffer) == 3 or len(self.frame_buffer) == 2

            if len(self.frame_buffer) == 2: # should equal to p_order==2
                self.frame_buffer = [self.frame_buffer[0], self.frame_buffer[0], self.frame_buffer[1]]
                self.flow_buffer = [torch.zeros_like(self.flow_buffer[0]), self.flow_buffer[0]]
            

            pred_frame, pred_flow = self.MWNet(self.frame_buffer, self.flow_buffer, True)

            flow = self.MENet(ref_frame, coding_frame)
            flow_hat, likelihood_m, pred_flow_hat, _, _, _ = self.CondMotion(flow, 
                                                                             output=pred_flow, 
                                                                             cond_coupling_input=pred_flow, 
                                                                             pred_prior_input=pred_frame,#)
                                                                             visual=visual, figname=visual_prefix.replace('visualize_ANFIC', 'visualize_motion_ANFIC'))
            self.flow_buffer.append(flow_hat)

            warped_frame_queue = self.warp_multiple_frames(self.frame_buffer, self.flow_buffer, True)

            if p_order == 2:
                weights = self.lstsq_multi_reference(warped_frame_queue[1:], coding_frame)
            else:
                weights = self.lstsq_multi_reference(warped_frame_queue, coding_frame)

            batch_size, _, H, W = coding_frame.shape
            fused_frame = []
            #weight_frame = []
            for bs in range(batch_size):
                fused_frame.append(sum([c*frame[bs, :] for c, frame in zip(weights[bs], warped_frame_queue)]))
                #weight_frame.append(torch.cat([c*torch.ones(1, H, W).cuda() for c in weights[bs]]))
            fused_frame = torch.stack(fused_frame)
            #weight_frame = torch.stack(weight_frame)

            #mc_frame = self.MCNet(torch.cat(warped_frame_queue + [fused_frame, weight_frame], dim=1))
            mc_frame = fused_frame
            
            #if _loss > 1e5:
            #    print("inf encountered ; distortion of mc = ", _dis.mean().item(), "; loss = ", _loss.item(), 
            #          "len(self.frame_buffer) = ", len(self.frame_buffer),
            #          "len(self.flow_buffer) = ", len(self.flow_buffer),
            #          "; SKIPPED")
            #    with open('./dump_info/weight.txt', 'w') as fp:
            #        for idx in range(batch_size):
            #            save_image(coding_frame[idx], './dump_info/'+f'coding_frame_{idx}.png')
            #            save_image(mc_frame[idx], './dump_info/'+f'mc_frame_{idx}.png')
            #            save_image(fused_frame[idx], './dump_info/'+f'fused_frame_{idx}.png')
            #            for i in range(2):
            #                save_image(self.flow_buffer[i][idx], './dump_info/'+f'flow_{idx}_#{i}.png')
            #            for i in range(3):
            #                save_image(self.frame_buffer[i][idx], './dump_info/'+f'ref_frame_{idx}_#{i}.png')
            #                save_image(warped_frame_queue[i][idx], './dump_info/'+f'warped_frame_{idx}_#{i}.png')
            #                fp.write(f"weight_batch_{idx}_#{i} = "+str(weights[idx][i].item()))
            #            fp.write(f"path_{idx}_#{i} = "+str(path[idx]))
            #    fp.close()
            #    
            #    fused_frame = sum(warped_frame_queue)/3
            #    weight_frame = torch.ones_like(coding_frame)/3
            #    mc_frame = self.MCNet(torch.cat(warped_frame_queue + [fused_frame, weight_frame], dim=1))
                
            likelihoods = likelihood_m
            data = {'likelihood_m': likelihood_m, 
                    'flow': flow, 'flow_hat': flow_hat, 'mc_frame': mc_frame, 
                    'pred_frame': pred_frame, 'pred_flow': pred_flow, 
                    'pred_flow_hat': pred_flow_hat, 'warped_frames': warped_frame_queue,
                    'fused_coef': weights}

        else:
            flow = self.MENet(ref_frame, coding_frame)
            flow_hat, likelihood_m = self.Motion(flow)

            warped_frame = self.Resampler(ref_frame, flow_hat)
            #weight_frame = torch.ones_like(coding_frame)/3
            #mc_frame = self.MCNet(torch.cat([warped_frame]*4 + [weight_frame], dim=1))
            mc_frame = warped_frame

            self.flow_buffer.append(flow_hat)

            likelihoods = likelihood_m
            data = {'likelihood_m': likelihood_m, 
                    'flow': flow, 'flow_hat': flow_hat, 
                    'mc_frame': mc_frame, 'warped_frames': [warped_frame]*3}
        
        if len(self.flow_buffer) == 3:
            self.flow_buffer.pop(0)
            assert len(self.flow_buffer) == 2, str(len(self.flow_buffer))

        return mc_frame, likelihoods, data


    def forward(self, ref_frame, coding_frame, p_order, visual=False, visual_prefix=''):
        mc, likelihood_m, m_info = self.motion_forward(ref_frame, coding_frame, visual=visual, visual_prefix=visual_prefix, p_order=p_order)

        reconstructed, likelihood_r, mc_hat, _, _, BDQ = self.Residual(coding_frame, output=mc, cond_coupling_input=mc,
                                                                       visual=visual, figname=visual_prefix)

        likelihoods = likelihood_m + likelihood_r

        #self.frame_buffer.append(reconstructed)
        self.frame_buffer.append(reconstructed.clamp(0, 1))

        if len(self.frame_buffer) == 4:
            self.frame_buffer.pop(0)
            assert len(self.frame_buffer) == 3, str(len(self.frame_buffer))
        
        return reconstructed, likelihoods, m_info, mc, BDQ, mc_hat

    def training_step(self, batch, batch_idx):
        epoch = self.current_epoch

        batch = batch.cuda()

        ref_frame = batch[:, 0]

        # I-frame
        with torch.no_grad():
            ref_frame, _, _, _, _, _ = self.if_model(ref_frame)

        self.frame_buffer = list()
        self.flow_buffer = list()

        if epoch < phase['trainMC']:
            self.MWNet.requires_grad_(False)
            #self.Motion.requires_grad_(False)

            _phase = 'MC'
            # First P-frame
            coding_frame = batch[:, 1]
            mc_frame, likelihood_m, data = self.motion_forward(ref_frame, coding_frame, p_order=1)

            distortion = self.criterion(coding_frame, mc_frame)
            rate = trc.estimate_bpp(likelihood_m, input=coding_frame)

            loss = (self.args.lmda * distortion.mean() + rate.mean())

            # One the other P-frame
            self.frame_buffer = [ref_frame, batch[:, 1], batch[:, 2]]
            self.flow_buffer = [
                                data['flow_hat'],
                                self.MENet(batch[:, 1], batch[:, 2]).detach()
                               ]
            
            ref_frame = batch[:, 2]
            coding_frame = batch[:, 3]

            mc_frame_1, likelihood_m_1, data_1 = self.motion_forward(ref_frame, coding_frame, p_order=3)


            distortion_1 = self.criterion(coding_frame, mc_frame_1)
            rate_1 = trc.estimate_bpp(likelihood_m_1, input=coding_frame)
            pred_frame_hat = self.Resampler(ref_frame, data_1['pred_flow_hat'])
            pred_frame_error_1 = nn.MSELoss(reduction='none')(data_1['pred_frame'], pred_frame_hat)

            loss += (self.args.lmda * distortion_1.mean() + rate_1.mean() + 0.01 * self.args.lmda * pred_frame_error_1.mean())

            loss /= 2

            logs = {'train/loss': loss.item(),
                    'train/distortion': np.mean([distortion.mean().item(), distortion_1.mean().item()]),
                    'train/rate': np.mean([rate.mean().item(), rate_1.mean().item()]),
                    'train/pred_frame_error': pred_frame_error_1.mean().item()
                   }

        elif epoch < phase['trainAll_2frames']:
            self.MWNet.requires_grad_(False)
            self.Motion.requires_grad_(False)

            if epoch < phase['trainRes_2frames']:
                _phase = 'RES'
            else:
                _phase = 'ALL'
            # First P-frame
            coding_frame = batch[:, 1]
            if _phase == 'RES':  # Train res_coder only
                with torch.no_grad():
                    mc_frame, likelihood_m, data = self.motion_forward(ref_frame, coding_frame, p_order=1)

            else:
                mc_frame, likelihood_m, data = self.motion_forward(ref_frame, coding_frame, p_order=1)

            rec_frame, likelihood_r, mc_hat, _, _, _ = self.Residual(coding_frame, cond_coupling_input=mc_frame,
                                                                     output=mc_frame)

            likelihoods = likelihood_m + likelihood_r

            distortion = self.criterion(coding_frame, rec_frame)
            rate = trc.estimate_bpp(likelihoods, input=coding_frame)
            mc_error = nn.MSELoss(reduction='none')(mc_frame, mc_hat)

            loss = self.args.lmda * distortion.mean() + rate.mean() + 0.01 * self.args.lmda * mc_error.mean()


            # One the other P-frame
            self.frame_buffer = [ref_frame, batch[:, 1], batch[:, 2]]
            self.flow_buffer = [
                                data['flow_hat'],
                                self.MENet(batch[:, 1], batch[:, 2])
                               ]
            ref_frame = batch[:, 2]
            coding_frame = batch[:, 3]

            if _phase == 'RES':  # Train res_coder only
                with torch.no_grad():
                    mc_frame, likelihood_m, data = self.motion_forward(ref_frame, coding_frame, p_order=3)

            else:
                mc_frame, likelihood_m, data = self.motion_forward(ref_frame, coding_frame, p_order=3)

            rec_frame, likelihood_r, mc_hat, _, _, _ = self.Residual(coding_frame, cond_coupling_input=mc_frame,
                                                                     output=mc_frame)

            likelihoods_1 = likelihood_m + likelihood_r

            distortion_1 = self.criterion(coding_frame, rec_frame)
            rate_1 = trc.estimate_bpp(likelihoods_1, input=coding_frame)
            mc_error_1 = nn.MSELoss(reduction='none')(mc_frame, mc_hat)
            pred_frame_hat = self.Resampler(ref_frame, data['pred_flow_hat'])
            pred_frame_error_1 = nn.MSELoss(reduction='none')(data['pred_frame'], pred_frame_hat)

            loss += self.args.lmda * distortion_1.mean() + rate_1.mean() + 0.01 * self.args.lmda * (mc_error_1.mean() + pred_frame_error_1.mean())
            
            loss /= 2

            logs = {
                'train/loss': loss.item(),
                'train/distortion': np.mean([distortion.mean().item(), distortion_1.mean().item()]),
                'train/rate': np.mean([rate.mean().item(), rate_1.mean().item()]),
                'train/PSNR': mse2psnr(np.mean([distortion.mean().item(), distortion_1.mean().item()])),
                'train/mc_error': np.mean([mc_error.mean().item(), mc_error_1.mean().item()]),
                'train/pred_frame_error': pred_frame_error_1.mean().item()
            }

        else:
            self.requires_grad_(True)
            #if epoch < phase['trainAll_fullgop']:
            #    self.MWNet.requires_grad_(False)
            
            ref_frame = batch[:, 0]
            reconstructed = ref_frame

            loss = torch.tensor(0., dtype=torch.float, device=reconstructed.device)
            dist_list = []
            rate_list = []
            mc_error_list = []
            pred_frame_error_list = []
            self.frame_buffer = []
            frame_count = 0

            for frame_idx in range(1, 5):
                frame_count += 1
                ref_frame = reconstructed
                
                coding_frame = batch[:, frame_idx]

                if frame_idx == 1:
                    self.frame_buffer = [ref_frame]

                reconstructed, likelihoods, m_info, mc, BDQ, mc_hat = self(ref_frame, coding_frame, p_order=frame_idx)
                
                distortion = self.criterion(coding_frame, reconstructed)
                rate = trc.estimate_bpp(likelihoods, input=coding_frame)
                mc_error = nn.MSELoss(reduction='none')(mc, mc_hat)
                
                dist_list.append(distortion.mean())
                rate_list.append(rate.mean())
                mc_error_list.append(mc_error.mean())
                if frame_idx == 1:
                    loss += self.args.lmda * distortion.mean() + rate.mean() + 0.01 * self.args.lmda * mc_error.mean()
                else:
                    pred_frame_hat = self.Resampler(ref_frame, m_info['pred_flow_hat'])
                    pred_frame_error = nn.MSELoss(reduction='none')(m_info['pred_frame'], pred_frame_hat)

                    loss += self.args.lmda * distortion.mean() + rate.mean() + 0.01 * self.args.lmda * (mc_error.mean() + pred_frame_error.mean())

                    pred_frame_error_list.append(pred_frame_error.mean())

                if epoch < phase['trainAll_fullgop']:
                    reconstructed = reconstructed.detach()
                    self.frame_buffer[-1] = self.frame_buffer[-1].detach()
                    self.flow_buffer[-1] = self.flow_buffer[-1].detach()

            loss = loss / frame_count
            distortion = torch.mean(torch.tensor(dist_list))
            rate = torch.mean(torch.tensor(rate_list))
            mc_error = torch.mean(torch.tensor(mc_error_list))
            pred_frame_error = torch.mean(torch.tensor(pred_frame_error_list))

            logs = {
                    'train/loss': loss.item(),
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
                                             image_channels=ch, overwrite=True)

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
        rate_my_list = []
        rate_mz_list = []
        m_rate_list = []
        loss_list = []
        align = trc.util.Alignment()

        epoch = int(self.current_epoch)

        self.frame_buffer = list()
        self.flow_buffer = list()

        for frame_idx in range(gop_size):
            ref_frame = ref_frame.clamp(0, 1)
            if frame_idx != 0:
                coding_frame = batch[:, frame_idx]

                if frame_idx == 1:
                    self.frame_buffer = [align.align(ref_frame)]
                    self.flow_buffer = list()

                rec_frame, likelihoods, m_info, mc_frame, BDQ, mc_hat = self(
                                                                               align.align(ref_frame),
                                                                               align.align(batch[:, frame_idx]),
                                                                               p_order=frame_idx
                                                                              )

                ref_frame = align.resume(ref_frame).clamp(0, 1)
                rec_frame = align.resume(rec_frame).clamp(0, 1)
                # coding_frame = align.resume(batch[:, frame_idx]).clamp(0, 1)
                mc_frame = align.resume(mc_frame).clamp(0, 1)
                mc_hat = align.resume(mc_hat).clamp(0, 1)
                BDQ = align.resume(BDQ).clamp(0, 1)

                rate = trc.estimate_bpp(likelihoods, input=ref_frame).mean().item()
                m_rate = trc.estimate_bpp(likelihoods[:2], input=ref_frame).mean().item()

                if frame_idx < 3:
                    mse = torch.mean((rec_frame - coding_frame).pow(2))
                    mc_mse = torch.mean((mc_frame - coding_frame).pow(2))
                    psnr = get_psnr(mse).cpu().item()
                    mc_psnr = get_psnr(mc_mse).cpu().item()

                    flow_hat = align.resume(m_info['flow'])
                    flow_rgb = torch.from_numpy(
                        fz.convert_from_flow(flow_hat[0].permute(1, 2, 0).cpu().numpy()) / 255).permute(2, 0, 1)
                    upload_img(flow_rgb.cpu().numpy(), f'{seq_name}_{epoch}_{frame_idx}_ori_flow.png', grid=False)

                    flow_hat = align.resume(m_info['flow_hat'])
                    flow_rgb = torch.from_numpy(
                        fz.convert_from_flow(flow_hat[0].permute(1, 2, 0).cpu().numpy()) / 255).permute(2, 0, 1)
                    upload_img(flow_rgb.cpu().numpy(), f'{seq_name}_{epoch}_{frame_idx}_dec_flow.png', grid=False)
                    
                    if frame_idx == 2:
 
                        flow_hat = align.resume(m_info['pred_flow'])
                        flow_rgb = torch.from_numpy(
                            fz.convert_from_flow(flow_hat[0].permute(1, 2, 0).cpu().numpy()) / 255).permute(2, 0, 1)
                        upload_img(flow_rgb.cpu().numpy(), f'{seq_name}_{epoch}_pred_flow.png', grid=False)
                        
                        flow_hat = align.resume(m_info['pred_flow_hat'])
                        flow_rgb = torch.from_numpy(
                            fz.convert_from_flow(flow_hat[0].permute(1, 2, 0).cpu().numpy()) / 255).permute(2, 0, 1)
                        upload_img(flow_rgb.cpu().numpy(), f'{seq_name}_{epoch}_pred_flow_hat.png', grid=False)
                        print(m_rate)

                    upload_img(ref_frame.cpu().numpy()[0], f'{seq_name}_{epoch}_ref_frame_{frame_idx}.png', grid=False)
                    upload_img(coding_frame.cpu().numpy()[0], f'{seq_name}_{epoch}_gt_frame_{frame_idx}.png', grid=False)
                    upload_img(mc_frame.cpu().numpy()[0], seq_name + '_{:d}_mc_frame_{:d}_{:.3f}.png'.format(epoch, frame_idx, mc_psnr),
                               grid=False)
                    upload_img(rec_frame.cpu().numpy()[0], seq_name + '_{:d}_rec_frame_{:d}_{:.3f}.png'.format(epoch, frame_idx, psnr),
                               grid=False)

                ref_frame = rec_frame

                mse = self.criterion(ref_frame, batch[:, frame_idx]).mean().item()
                psnr = mse2psnr(mse)
                mc_mse = self.criterion(mc_frame, batch[:, frame_idx]).mean().item()
                mc_psnr = mse2psnr(mc_mse)
                loss = self.args.lmda * mse + rate

                mc_psnr_list.append(mc_psnr)
                m_rate_list.append(m_rate)

            else:
                with torch.no_grad():
                    rec_frame, likelihoods, _, _, _, _ = self.if_model(align.align(batch[:, frame_idx]))

                rec_frame = align.resume(rec_frame).clamp(0, 1)
                rate = trc.estimate_bpp(likelihoods, input=rec_frame).mean().item()

                mse = self.criterion(rec_frame, batch[:, frame_idx]).mean().item()
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
            #print(logs['val/'+dataset_name+' m_rate'])

        self.log_dict(logs)

        return None

    def test_step(self, batch, batch_idx):
        metrics_name = ['PSNR', 'Rate', 'Mo_Rate', 'MC-PSNR', 'MCrec-PSNR', 'MCerr-PSNR', 'BDQ-PSNR', 'QE-PSNR',
                        'back-PSNR', 'p1-PSNR', 'p1-BDQ-PSNR']
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

        ref_frame = batch[:, 0]  # Put reference frame in first position
        batch = batch[:, 1:]  # GT
        seq_name = seq_name[0]
        dataset_name = dataset_name[0]

        gop_size = batch.size(1)

        height, width = ref_frame.size()[2:]
        estimate_bpp = partial(trc.estimate_bpp, num_pixels=height * width)

        psnr_list = []
        mc_psnr_list = []
        mc_hat_psnr_list = []
        BDQ_psnr_list = []
        rate_list = []
        m_rate_list = []
        log_list = []
        align = trc.util.Alignment()

        self.frame_buffer = list()
        self.flow_buffer = list()

        for frame_idx in range(gop_size):
            ref_frame = ref_frame.clamp(0, 1)
            #TO_VISUALIZE = frame_id_start == 1 and frame_idx < 8 and seq_name in ['Jockey', 'HoneyBee', 'Bosphorus']
            TO_VISUALIZE = seq_name in ['Jockey', 'HoneyBee']
            if frame_idx != 0:
                coding_frame = batch[:, frame_idx]

                if frame_idx == 1:
                    self.frame_buffer = [align.align(ref_frame)]

                # reconstruced frame will be next ref_frame
                if TO_VISUALIZE:
                    os.makedirs(os.path.join(self.args.save_dir, 'visualize_ANFIC', f'{seq_name}'), exist_ok=True)
                    os.makedirs(os.path.join(self.args.save_dir, 'visualize_motion_ANFIC', f'{seq_name}'), exist_ok=True)

                    rec_frame, likelihoods, m_info, mc_frame, BDQ, mc_hat = self(
                        align.align(ref_frame),
                        align.align(coding_frame),
                        p_order=frame_idx,
                        visual=True,
                        visual_prefix=os.path.join(
                            self.args.save_dir,
                            'visualize_ANFIC',
                            f'{seq_name}',
                            f'frame_{frame_idx}',
                        )
                    )

                else:
                    rec_frame, likelihoods, m_info, mc_frame, BDQ, mc_hat = self(
                                                                                   align.align(ref_frame),
                                                                                   align.align(batch[:, frame_idx]),
                                                                                   p_order=frame_idx
                                                                              )
                ref_frame = rec_frame.detach()
                ref_frame = align.resume(ref_frame).clamp(0,1)
                mc_frame = align.resume(mc_frame)
                warped_frames = [align.resume(frame).clamp(0,1) for frame in m_info['warped_frames']]
                coding_frame = align.resume(coding_frame)
                mc_hat = align.resume(mc_hat)
                BDQ = align.resume(BDQ)

                res_frame = coding_frame - mc_frame
                res_frame = torch.abs(res_frame)
                res_frame = res_frame.clamp(0., 1.)

                os.makedirs(self.args.save_dir + f'/{seq_name}/flow', exist_ok=True)
                os.makedirs(self.args.save_dir + f'/{seq_name}/gt_frame', exist_ok=True)
                os.makedirs(self.args.save_dir + f'/{seq_name}/mc_frame', exist_ok=True)
                os.makedirs(self.args.save_dir + f'/{seq_name}/rec_frame', exist_ok=True)
                os.makedirs(self.args.save_dir + f'/{seq_name}/res_frame', exist_ok=True)
                os.makedirs(self.args.save_dir + f'/{seq_name}/pred_frame', exist_ok=True)
                os.makedirs(self.args.save_dir + f'/{seq_name}/BDQ', exist_ok=True)
                os.makedirs(self.args.save_dir + f'/{seq_name}/mc_hat', exist_ok=True)
                os.makedirs(self.args.save_dir + f'/{seq_name}/fusion_coef', exist_ok=True)

                if TO_VISUALIZE:
                    flow_map = plot_flow(m_info['flow'])
                    save_image(flow_map,
                               self.args.save_dir + f'/{seq_name}/flow/'
                                                    f'frame_{int(frame_id_start + frame_idx)}_ori_flow.png',
                               nrow=1)

                    flow_map = plot_flow(m_info['flow_hat'])
                    save_image(flow_map,
                               self.args.save_dir + f'/{seq_name}/flow/'
                                                    f'frame_{int(frame_id_start + frame_idx)}_flow.png',
                               nrow=1)

                    if frame_idx > 1:
                        flow_map = plot_flow(m_info['pred_flow'])
                        save_image(flow_map,
                                   self.args.save_dir + f'/{seq_name}/flow/'
                                                        f'frame_{int(frame_id_start + frame_idx)}_flow_pred.png',
                                   nrow=1)

                        flow_map = plot_flow(m_info['pred_flow_hat'])
                        save_image(flow_map,
                                   self.args.save_dir + f'/{seq_name}/flow/'
                                                        f'frame_{int(frame_id_start + frame_idx)}_flow_pred_hat.png',
                                   nrow=1)

                        pred_frame = align.resume(m_info['pred_frame'])
                        save_image(pred_frame, self.args.save_dir + f'/{seq_name}/pred_frame/'
                                                                    f'frame_{int(frame_id_start + frame_idx)}.png')

                        with open(self.args.save_dir + f'/{seq_name}/fusion_coef/'+f'coef_{int(frame_id_start + frame_idx)}.txt', 'w') as f:
                            f.write('alpha = ' + str(m_info['fused_coef'][0][0].item()) + '\n')
                            f.write('beta = ' + str(m_info['fused_coef'][0][1].item()) + '\n')
                            if frame_idx > 2:
                                f.write('gamma = ' + str(m_info['fused_coef'][0][2].item()) + '\n')
                        #print('my =', estimate_bpp(likelihoods[0]).item(), ' ; mz =', estimate_bpp(likelihoods[1]).item())

                    save_image(coding_frame[0], self.args.save_dir + f'/{seq_name}/gt_frame/'
                                                                     f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(mc_frame[0], self.args.save_dir + f'/{seq_name}/mc_frame/'
                                                                 f'frame_{int(frame_id_start + frame_idx)}.png')
                    for idx, warped_frame in enumerate(warped_frames):
                        save_image(warped_frame[0], self.args.save_dir + f'/{seq_name}/mc_frame/'
                                                                  f'frame_{int(frame_id_start + frame_idx)}_bmc_{idx}.png')
                    save_image(ref_frame[0], self.args.save_dir + f'/{seq_name}/rec_frame/'
                                                                  f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(res_frame[0], self.args.save_dir + f'/{seq_name}/res_frame/'
                                                                  f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(BDQ[0], self.args.save_dir + f'/{seq_name}/BDQ/'
                                                            f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(mc_hat[0], self.args.save_dir + f'/{seq_name}/mc_hat/'
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
        print('lr decay =', lr_gamma, 'lr milestones =', lr_step)

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
            self.val_dataset = VideoTestDataIframe(dataset_root, self.args.lmda, first_gop=True)

        elif stage == 'test':
            #self.test_dataset = VideoTestDataIframe(dataset_root, self.args.lmda, sequence=('U', 'B', 'M'))
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
    parser.add_argument('--experiment_name', type=str, default='basic')
    parser.add_argument('--project_name', type=str, default="CANFVC+")

    parser.add_argument('--MENet', type=str, choices=['PWC', 'SPy'], default='PWC')
    parser.add_argument('--motion_coder_conf', type=str, default=None)
    parser.add_argument('--cond_motion_coder_conf', type=str, default=None)
    parser.add_argument('--residual_coder_conf', type=str, default=None)
    parser.add_argument('--prev_motion_coder_conf', type=str, default=None)
    parser.add_argument('--prev_cond_motion_coder_conf', type=str, default=None)
    parser.add_argument('--prev_residual_coder_conf', type=str, default=None)

    parser.set_defaults(gpus=1)

    # parse params
    args = parser.parse_args()

    experiment_name = args.experiment_name
    project_name = args.project_name

    # I-frame coder ckpt
    ANFIC_code = {2048: '0821_0300', 1024: '0530_1212', 512: '0530_1213', 256: '0530_1215'}[args.lmda]

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


    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        save_last=True,
        #every_n_epochs=10, # Save at least every 10 epochs
        period=1, # Save at least every 3 epochs
        verbose=True,
        monitor='val/loss',
        mode='min',
        prefix=''
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

        trainer.current_epoch = epoch_num + 1
        
        model = Pframe(args, mo_coder, cond_mo_coder, res_coder).cuda()
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

        trainer.current_epoch = phase['trainAll_RNN_1'] - 1
        #trainer.current_epoch = phase['trainMV']

        coder_ckpt = torch.load(os.path.join(os.getenv('LOG', './'), f"ANFIC/ANFHyperPriorCoder_{ANFIC_code}/model.ckpt"),
                                map_location=(lambda storage, loc: storage))['coder']

        for k, v in coder_ckpt.items():
            key = 'if_model.' + k
            checkpoint['state_dict'][key] = v

        model = Pframe(args, mo_coder, cond_mo_coder, res_coder).cuda()
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    
    elif args.restore == 'custom':
        trainer = Trainer.from_argparse_args(args,
                                             checkpoint_callback=checkpoint_callback,
                                             gpus=args.gpus,
                                             distributed_backend=db,
                                             logger=comet_logger,
                                             default_root_dir=save_root,
                                             check_val_every_n_epoch=1,
                                             num_sanity_val_steps=-1,
                                             terminate_on_nan=True)
        
        epoch_num = args.restore_exp_epoch
        if args.restore_exp_key is None:
            raise ValueError
        else:  # When prev_exp_key is specified in args
            checkpoint = torch.load(os.path.join(save_root, project_name, args.restore_exp_key, "checkpoints",
                                                 f"epoch={epoch_num}.ckpt"),
                                    map_location=(lambda storage, loc: storage))

        trainer.current_epoch = phase['trainMC'] - 3
        # Previous coders
        #assert not (args.prev_motion_coder_conf is None)
        #prev_mo_coder_cfg = yaml.safe_load(open(args.prev_motion_coder_conf, 'r'))
        #assert prev_mo_coder_cfg['model_architecture'] in trc.__CODER_TYPES__.keys()
        #prev_mo_coder_arch = trc.__CODER_TYPES__[prev_mo_coder_cfg['model_architecture']]
        #prev_mo_coder = prev_mo_coder_arch(**prev_mo_coder_cfg['model_params'])
        #
        #assert not (args.prev_cond_motion_coder_conf is None)
        #prev_cond_mo_coder_cfg = yaml.safe_load(open(args.prev_cond_motion_coder_conf, 'r'))
        #assert prev_cond_mo_coder_cfg['model_architecture'] in trc.__CODER_TYPES__.keys()
        #prev_cond_mo_coder_arch = trc.__CODER_TYPES__[prev_cond_mo_coder_cfg['model_architecture']]
        #prev_cond_mo_coder = prev_cond_mo_coder_arch(**prev_cond_mo_coder_cfg['model_params'])

        #assert not (args.prev_residual_coder_conf is None)
        #prev_res_coder_cfg = yaml.safe_load(open(args.prev_residual_coder_conf, 'r'))
        #assert prev_res_coder_cfg['model_architecture'] in trc.__CODER_TYPES__.keys()
        #prev_res_coder_arch = trc.__CODER_TYPES__[prev_res_coder_cfg['model_architecture']]
        #prev_res_coder = prev_res_coder_arch(**prev_res_coder_cfg['model_params'])
   
        coder_ckpt = torch.load(os.path.join(os.getenv('LOG', './'), f"ANFIC/ANFHyperPriorCoder_{ANFIC_code}/model.ckpt"),
                                map_location=(lambda storage, loc: storage))['coder']

        from collections import OrderedDict
        new_ckpt = OrderedDict()

        for k, v in coder_ckpt.items():
            key = 'if_model.' + k
            new_ckpt[key] = v

        for k, v in checkpoint['state_dict'].items():
            if k[:6] == 'Motion':
                new_ckpt[k] = v
            elif k[:10] == 'CondMotion':
                new_ckpt[k] = v
            elif k[:8] == 'Residual':
                new_ckpt[k] = v
 
        model = Pframe(args, mo_coder, cond_mo_coder, res_coder).cuda()
        model.load_state_dict(new_ckpt, strict=False)
 
        #model = Pframe(args, prev_mo_coder, prev_cond_mo_coder, prev_res_coder).cuda()

    else:
        trainer = Trainer.from_argparse_args(args,
                                             checkpoint_callback=checkpoint_callback,
                                             gpus=args.gpus,
                                             distributed_backend=db,
                                             logger=comet_logger,
                                             default_root_dir=save_root,
                                             check_val_every_n_epoch=3,
                                             num_sanity_val_steps=-1,
                                             terminate_on_nan=True)
    
        coder_ckpt = torch.load(os.path.join(os.getenv('LOG', './'), f"ANFIC/ANFHyperPriorCoder_{ANFIC_code}/model.ckpt"),
                                map_location=(lambda storage, loc: storage))['coder']

        for k, v in coder_ckpt.items():
            key = 'if_model.' + k
            checkpoint['state_dict'][key] = v

     
        model = Pframe(args, mo_coder, cond_mo_coder, res_coder).cuda()
        #summary(model.Motion)
        #summary(model.CondMotion)
        #summary(model.Residual)
        #summary(model)

    if args.test:
        trainer.test(model)
    else:
        trainer.fit(model)
