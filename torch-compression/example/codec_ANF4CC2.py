# PyTorch tools
import argparse
import functools
import os
import random
import shutil
import sys
import time
from functools import partial
from glob import glob

import matplotlib.pyplot as plt
# Python tools
import numpy as np
import torch
import torch_compression as trc
import torchvision
from absl import app
from absl.flags import argparse_flags
# - Metrics
from skimage import io
from torch import mean, nn
from torch.nn.parallel import data_parallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_compression.modules.conditional_module import (
    ConditionalLayer, conditional_warping, gen_condition, gen_discrete_condition, gen_random_condition,
    set_condition)
from torch_compression.util.toolbox import torchseed
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm, trange

# - Tools
try:
    from util.auto_helper import cptree, ffilter, mkdir
    from util.datasets import MSCOCO, CLICTrain, CustomData, Kodak, Vimeo90K
    from util.log_manage import (AvgMeter, dump_args, gen_log_folder_name,
                                 load_args)
    from util.loss import MS_SSIM, PSNR, huber_loss
    from util.metric import MultiScaleSSIM, PSNR_np
    from util.optim import RAdam
except:
    from .util.auto_helper import cptree, ffilter, mkdir
    from .util.datasets import MSCOCO, CLICTrain, CustomData, Kodak, Vimeo90K
    from .util.log_manage import (AvgMeter, dump_args, gen_log_folder_name,
                                  load_args)
    from .util.loss import MS_SSIM, PSNR, huber_loss
    from .util.metric import MultiScaleSSIM, PSNR_np
    from .util.optim import RAdam


# Enable CUDA computation
use_cuda = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if use_cuda else 'cpu')

Dataset_dir = os.getenv('DATASET')
LogRoot = os.getenv('LOG', './') + "torch_compression/"


def switch_mode(coder, args, iter, mode):
    if iter % args.alt_step == 0:
        coder.num_layers = mode+1
        if mode > 1:
            coder['analysis'+str(mode-1)].requires_grad_(False)
            coder['synthesis'+str(mode-1)].requires_grad_(False)
        else:
            coder.requires_grad_(True)
        mode = (mode + 1) % args.num_layers

    return mode


class Queue():
    def __init__(self, max_size):
        self.max_size = max_size
        self._values = []

    def put(self, item):
        self._values.append(item)
        if len(self._values) > self.max_size:
            self._values.pop(0)

    def mean(self):
        return np.mean(self._values) if len(self._values) else np.zeros(1).mean()


def train(args):
    model_ID = os.path.basename(
        args.checkpoint_dir if args.checkpoint_dir[-1] is not '/' else args.checkpoint_dir[:-1])

    # Create input data pipeline.
    data_transforms = transforms.Compose([
        transforms.RandomCrop(args.patch_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    train_dataset = {"MSCOCO": MSCOCO(Dataset_dir + "COCO/train2014/", data_transforms),
                     "CLIC": CLICTrain(Dataset_dir + "CLIC_train/images/", data_transforms),
                     "VIMEO": Vimeo90K(Dataset_dir + "vimeo_septuplet/", data_transforms)}[args.dataset]
    train_dataloader = DataLoader(
        train_dataset, args.batch_size, True, num_workers=16, drop_last=True)
    validate_dataset = Kodak(Dataset_dir + "Kodak/", transforms.ToTensor())
    validate_dataloader = DataLoader(
        validate_dataset, 1, False, num_workers=16)

    coder = trc.hub.AugmentedNormalizedFlowHyperPriorCoder(
        args.var_filters, args.num_features, args.num_hyperpriors, gdn_mode=args.gdn_mode,
        num_layers=args.num_layers, use_DQ=args.use_DQ, share_wei=args.share_weight,
        init_code=args.init_code, use_code=not args.disable_code, dec_add=args.alt_mode != "None", use_attn=args.use_attn,
        use_mean=args.Mean, use_context=args.use_context, condition=args.condition, quant_mode=args.quant_mode)

    args.lmda = sorted(args.lmda, reverse=True)
    ckpt_dirs = [args.checkpoint_dir]
    for _ in args.lmda[:-1]:
        ckpt_dirs.append(gen_log_folder_name(
            root=LogRoot, prefix=args.architecture+"_"))
        os.makedirs(ckpt_dirs[-1], exist_ok=False)
    print(ckpt_dirs)
    log_writers = dict(
        zip(args.lmda, [SummaryWriter(log_dir=d) for d in ckpt_dirs]))
    log_writer = log_writers[args.lmda[0]]
    ft = ffilter(keep=['*.py', '*.txt'],
                 ignore=['tmp/', 'results/', 'models/',
                         '.vscode/', '.ipynb_checkpoints/', '.git/', 'src/'])
    cptree('./', mkdir(args.checkpoint_dir+'/src'), ffilter=ft, param='-va')
    if args.detach_enc:
        args.rec_code = True
    print(coder)
    print(args)
    log_writer.add_text('Config', str(vars(args)))

    psnr = PSNR(reduction='mean', data_range=255.)
    ms_ssim = MS_SSIM(reduction='mean', data_range=255.).to(DEVICE)

    if args.metric == 'SSIM':
        ms_ssim_criterion = MS_SSIM(data_range=1.).to(DEVICE)

    best_eval_value = 1e9

    @torch.no_grad()
    def evaluate(epoch, eval_lmda, is_resume=False):
        coder.eval()
        if eval_lmda in log_writers:
            eval_log_writer = log_writers[eval_lmda]
            eval_dir = eval_log_writer.get_logdir()
        else:
            eval_log_writer = None
            eval_dir = args.checkpoint_dir

        split_rates = [
            f"rate_z{str(n+1)}" for n in range(coder.num_bitstreams)]
        logger = AvgMeter(
            ['psnr', "bdq_psnr", 'ssim', "bdq_ssim", 'rate', 'aux_loss', 'rd_loss'] + split_rates)
        if is_resume:
            visual = -2
        elif epoch == 0:
            visual = -1
        elif epoch % 3 == 0:
            visual = epoch
        else:
            visual = 0
        if eval_log_writer is None:
            visual = 0
        if args.enc_optim:
            coder.requires_grad_(False)
        if args.enc_noise:
            torchseed(666)
        step = 200
        steps = np.arange(step)
        # coder.train()
        lmda = torch.Tensor([eval_lmda]).view(1, 1).to(DEVICE)
        for idx, eval_imgs in enumerate(validate_dataloader):
            if args.enc_noise:
                eval_imgs = torch.rand(1, 3, 512, 768).to(DEVICE)
            else:
                eval_imgs = eval_imgs.to(DEVICE)

            if args.enc_optim:
                input, code, jac = eval_imgs, None, None
                input, code, jac = coder.encode(input, code, jac)
                init_X, feature = input, code
                optim2 = torch.optim.Adam(
                    [feature.requires_grad_(True)], lr=2e-4)
                with torch.enable_grad():
                    print("optim idx", idx)
                    optim_logger = AvgMeter(["D", "R", "T", "P"])
                    op_bar = tqdm(range(step))
                    for i in op_bar:
                        optim2.zero_grad()
                        set_condition(coder, lmda)
                        Y_error, y_tilde, z_tilde, y_likelihood, z_likelihood = coder.entropy_model(
                            init_X, feature)
                        input, code, hyper_code = torch.zeros_like(
                            input), y_tilde, z_tilde
                        input, code, jac = coder.decode(input, code, jac)
                        if coder.DQ is not None:
                            BDQ = input
                            input = coder.DQ(input)
                        rec = input
                        distortion = torch.nn.functional.mse_loss(
                            rec, eval_imgs)
                        Y_dis = Y_error.pow(2).mean() * \
                            (255**2 * eval_lmda * 0.01)
                        rate = trc.estimate_bpp(
                            (y_likelihood, z_likelihood), input=rec)
                        rd_loss = distortion * \
                            (255**2 * eval_lmda) + Y_dis + rate.mean()

                        rd_loss.backward()
                        optim2.step()
                        op_bar.set_description_str("{:.3e}, {:.4f}, {:.4f}".format(
                            distortion.item(), rate.item(), rd_loss.item()))

                        optim_logger.append("D", distortion)
                        optim_logger.append("R", rate)
                        optim_logger.append("T", rd_loss)
                        optim_logger.append(
                            "P", distortion.reciprocal().log10()*10)

                    fig, ax = plt.subplots(4, sharex=True)
                    ax[0].plot(steps, optim_logger['D'], label="distortion")
                    ax[1].plot(steps, optim_logger['R'], label="rate")
                    ax[2].plot(steps, optim_logger['T'], label="total_rd")
                    ax[3].plot(steps, optim_logger['P'], label="psnr")

                    for canvas in ax:
                        canvas.legend()
                    plt.savefig(eval_dir +
                                "/kodak_%02d_optim.png" % (idx+1))
                    plt.close(fig)

                eval_imgs_tilde, eval_likelihoods = rec, (
                    y_likelihood, z_likelihood)
            else:
                set_condition(coder, lmda)
                eval_imgs_tilde, eval_likelihoods, Y_error, jac, _, BDQ = coder(
                    eval_imgs, visual=visual, figname=os.path.join(eval_dir, "kodak_%02d" % (idx+1)))
                # disY = Y_error.pow(2).flatten(1).mean(1)
                # jac = torch.stack(jac, 1) if len(jac) else imgs.new_zeros(
                #     imgs.size(0), 1)

            rates = [trc.estimate_bpp(
                eval_likelihoods[n], input=eval_imgs) for n in range(coder.num_bitstreams)]
            eval_rate = sum(rates)

            eval_aux_loss = coder.aux_loss()

            if args.metric == 'PSNR':
                distortion = (
                    eval_imgs - eval_imgs_tilde).pow(2).flatten(1).mean(1)
                eval_rd_loss = torch.mean(
                    (lmda * 255**2) * distortion + eval_rate)
            elif args.metric == 'SSIM':
                distortion = 1 - ms_ssim_criterion(eval_imgs_tilde, eval_imgs)
                eval_rd_loss = torch.mean(
                    lmda * 2000 * distortion + eval_rate)

            eval_imgs = eval_imgs.mul(255).clamp(0, 255).round()
            eval_imgs_tilde = eval_imgs_tilde.mul(255).clamp(0, 255).round()

            eval_psnr = psnr(eval_imgs, eval_imgs_tilde)
            eval_msssim = ms_ssim(eval_imgs, eval_imgs_tilde)

            BDQ = BDQ.mul(255).clamp(0, 255).round()
            bdq_psnr = psnr(eval_imgs, BDQ).item()
            bdq_msssim = ms_ssim(eval_imgs, BDQ)

            logger.append('psnr', eval_psnr)
            logger.append('bdq_psnr', bdq_psnr)
            logger.append('ssim', eval_msssim)
            logger.append('bdq_ssim', bdq_msssim)
            logger.append('rate', eval_rate)
            logger.append('aux_loss', eval_aux_loss)
            logger.append('rd_loss', eval_rd_loss)
            for n in range(len(rates)):
                logger.append(split_rates[n], rates[n])

            # if not is_resume and idx == 6:
            if eval_log_writer is not None:
                save_image(eval_imgs_tilde.div(255),
                           os.path.join(eval_dir, "kodak_%02d.png" % (idx+1)))

        eval = logger.mean()

        if eval_log_writer is not None:
            with open(os.path.join(eval_dir, "eval.txt"), 'w') as fp:
                fp.write(
                    'name\t\tpsnr\tbdq_psnr\tssim\tbdq_ssim\trate\t{}\t{}\n'.format("\t".join(split_rates), "\t".join(["%"]*len(rates))))
                for idx in range(len(logger)):
                    eval_items = logger[idx]
                    line = "kodak_{idx:02d}:\t{psnr:.4f}\t{bdq_psnr:.4f}\t{ssim:.6f}\t{bdq_ssim:.4f}\t{rate:.4f}".format(
                        idx=idx+1, **eval_items)
                    for n in range(len(rates)):
                        line += "\t{:.4f}".format(eval_items[split_rates[n]])
                    for n in range(len(rates)):
                        line += "\t{:.4f}".format(
                            eval_items[split_rates[n]]/eval_items['rate'])
                    fp.write(line+"\n")
                fp.write("AVG.\n")
                avg = "kodak:\t\t{psnr:.4f}\t{bdq_psnr:.4f}\t{ssim:.6f}\t{bdq_ssim:.4f}\t{rate:.4f}".format(
                    **eval)
                for n in range(len(rates)):
                    avg += "\t{:.4f}".format(eval[split_rates[n]])
                for n in range(len(rates)):
                    avg += "\t{:.4f}".format(eval[split_rates[n]]/eval['rate'])
                fp.write(avg+"\n")

        print("lmda={:.1e}::".format(lmda.item()) if eval_lmda is not None else "",
              "PSNR: {psnr:.4f}, MS-SSIM: {ssim:.6f}, rate: {rate:.4f}, aux: {aux_loss:.4f}, rd_loss: {rd_loss:.4f}".format(**eval))

        if not is_resume and eval_log_writer is not None:
            eval_log_writer.add_scalar('Loss/aux', eval['aux_loss'], epoch)
            eval_log_writer.add_scalar('Evaluation/PSNR', eval["psnr"], epoch)
            eval_log_writer.add_scalar(
                'Evaluation/MS-SSIM', eval["ssim"], epoch)
            eval_log_writer.add_scalar(
                'Evaluation/est-rate', eval["rate"], epoch)
            eval_log_writer.add_scalar(
                'Evaluation/total', eval['rd_loss'], epoch)

        if args.enc_optim or args.enc_noise:
            exit(0)
        return eval['rd_loss']

    optim_type = {"Adam": torch.optim.Adam,
                  "AdamW": torch.optim.AdamW, "RAdam": RAdam}[args.optim]
    cond_warping = partial(conditional_warping,
                           conditions=1, ver=args.cond_ver)
    # load checkpoint if needed/ wanted
    grad_log = os.path.join(args.checkpoint_dir, "grad_log")
    if args.check_grad:
        os.makedirs(grad_log, exist_ok=True)
    trc.util.toolbox._check_grad = args.check_grad
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(os.path.join(args.checkpoint_dir, "model.ckpt"),
                          map_location='cpu')
        print("========================================================================")

        start_epoch = ckpt['epoch'] + 1

        try:
            if args.pretrained:
                cond_warping(coder)
            coder.load_state_dict(ckpt['coder'])
            if not args.pretrained:
                cond_warping(coder)
            coder = coder.to(DEVICE)

            optim = optim_type([dict(params=coder.main_parameters(), lr=args.lr),
                                dict(params=coder.aux_parameters(), lr=args.lr*10)])  # OPTIM
            sched = torch.optim.lr_scheduler.MultiStepLR(
                optim, milestones=[500, 800], gamma=0.5)  # OPTIM

            # optim.load_state_dict(ckpt['optim'])
        except RuntimeError as e:
            # Warning(e)
            print(e)
            ans = input("Sure to ignore these module(s)?(Y/n)>>>")
            if ans.strip().lower() not in ["", "y"]:
                exit(0)
            print(coder)
            coder.load_state_dict(ckpt['coder'], strict=False)
            cond_warping(coder)
            coder = coder.to(DEVICE)

            optim = optim_type([dict(params=coder.main_parameters(), lr=args.lr),
                                dict(params=coder.aux_parameters(), lr=args.lr*10)])  # OPTIM
            sched = torch.optim.lr_scheduler.MultiStepLR(
                optim, milestones=[500, 800], gamma=0.5)  # OPTIM

        if 'sched' in ckpt:
            sched.load_state_dict(ckpt['sched'])

        if not args.verbose:
            best_eval_value = np.mean(
                [evaluate(0, lmda, is_resume=True) for lmda in args.lmda])

        print("Latest checkpoint restored, start training at step {}.".format(start_epoch))

    else:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        coder = coder.to(DEVICE)

    log_writer.add_text('Architecture', str(coder))

    if args.fix_ANCHOR:
        for n, mm in coder.named_modules():
            if isinstance(mm, ConditionalLayer):
                mm.requires_grad_(True)
                # print(n, "C")
            else:
                mm.requires_grad_(False)
                # print(n)

    if args.fix_ANF:
        for n, m in coder.named_children():
            if n not in ["DQ", "syntax"]:
                print("FIX", n)
                m.requires_grad_(False)

    model_forward = partial(data_parallel, coder) if args.parallel else coder
    main_parameters = coder.main_parameters()
    if args.alt_mode == "Force":
        switch_mode(coder, args, 0, args.alt_step)

    condition_func = gen_random_condition if args.cond_mode == 'rand' else gen_condition

    mode = 0
    iter_count = 0
    loss_queue = Queue(1000)
    for epoch in range(start_epoch, args.max_epochs):
        coder.train()

        logger = AvgMeter(
            ['rd_loss', 'distortion', 'rate', 'disY', 'jac'])

        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        # for loop going through dataset
        if args.alt_mode == "outloop":
            mode = switch_mode(coder, args, epoch, mode)

        for iter, imgs in enumerate(pbar):
            imgs = imgs.to(DEVICE)
            lmda = condition_func(args.lmda, args.batch_size, device=DEVICE)
            if iter == 0:
                print(lmda.flatten().cpu().numpy())
            # print(lmda)
            set_condition(coder, lmda)
            lmda = lmda.flatten()
            weight = torch.nn.functional.normalize(
                lmda.reciprocal().pow(args.weight_gamma), 1., dim=0)

            if args.alt_mode == 'inloop':
                mode = switch_mode(coder, args, iter, mode)

            optim.zero_grad()
            if args.parallel:
                imgs_tilde, likelihoods, Y_error, jac, y_err, _ = model_forward(
                    imgs, module_kwargs=dict(jac=args.cal_jac, rev_ng=args.rev_ng, rec_code=args.rec_code, IDQ=args.IDQ, detach_enc=args.detach_enc))
            else:
                imgs_tilde, likelihoods, Y_error, jac, y_err, _ = model_forward(
                    imgs, jac=args.cal_jac, rev_ng=args.rev_ng, rec_code=args.rec_code, IDQ=args.IDQ, detach_enc=args.detach_enc)
            imgs_tilde = trc.util.bound(imgs_tilde, 0, 1)
            disY = Y_error.pow(2).flatten(1).mean(1)
            if not args.cal_jac:
                jac = imgs.new_zeros(imgs.size(0), 1)

            if args.rec_code:
                code_err = y_err.pow(2).flatten(1).mean(1)
            else:
                code_err = imgs.new_zeros(imgs.size(0))

            rate = trc.estimate_bpp(likelihoods, input=imgs)

            flag = False
            if args.metric == 'PSNR':
                if args.use_huber:
                    distortion = huber_loss(
                        imgs_tilde * 255, imgs * 255, 25) / 255 ** 2.
                else:
                    distortion = (imgs - imgs_tilde).pow(2).flatten(1).mean(1)
                if torch.isnan(distortion).any() or distortion.gt(100).any():
                    print("dis", distortion)
                    print(imgs_tilde.min(), imgs_tilde.max())
                    flag = True
                if torch.isnan(disY).any():
                    print('Y_dis', disY)
                    flag = True
                if torch.isnan(rate).any():
                    print('rate', rate)
                    flag = True
                mask = distortion.lt(100)  # OPTIM
                if mask.sum().item() == 0:
                    rd_loss = distortion.mul(0).mean()
                else:
                    rd_loss = (lmda * 255**2) * distortion + rate
                    if args.cal_yerr:
                        rd_loss += (lmda * 255**2 * 0.01) * disY
                    if args.cal_jac:
                        rd_loss -= (lmda * 255**2 * 1e-5) * jac.sum(1)
                    if args.rec_code:
                        rd_loss += code_err
                    rd_loss *= weight
                    rd_loss = (rd_loss * mask).sum() / mask.sum()
            elif args.metric == 'SSIM':
                distortion = 1 - ms_ssim_criterion(imgs_tilde, imgs)
                rd_loss = lmda * 2000 * distortion + rate
                if args.cal_yerr:
                    rd_loss += (lmda * 255**2 * 0.01) * disY
                rd_loss *= weight
                rd_loss = rd_loss.mean()

            aux_loss = coder.aux_loss()

            rd_loss.add(aux_loss).backward()

            if iter % 500 == 0:
                trc.util.vision.plot_grad_flow(coder.named_main_parameters(
                ), os.path.join(args.checkpoint_dir, "gradflow.png"))
            if args.clip_max_norm > 0:
                nn.utils.clip_grad_norm_(main_parameters, args.clip_max_norm)
            if args.check_grad:
                trc.AugmentedNormalizedFlows.dump_grad(
                    os.path.join(grad_log, f"grad_log_{iter}.txt"))

            # if iter % 500 == 0:
            #     save_image(torch.cat(
            #         [imgs[:5], imgs_tilde[:5]]), args.checkpoint_dir+"/check_imgs.png", nrow=5)

            if iter_count > 2000 and rd_loss.item() > loss_queue.mean() * 2:
                # save_image(torch.cat(
                # [imgs, imgs_tilde]), args.checkpoint_dir+"/err_imgs.png", nrow=args.batch_size)
                # coder(imgs, jac=args.cal_jac, rev_ng=args.rev_ng, rec_code=args.rec_code, IDQ=args.IDQ,
                #       detach_enc=args.detach_enc, visual=iter_count, figname=args.checkpoint_dir+"/err_img")

                ckpt = torch.load(os.path.join(args.checkpoint_dir, "model_back.ckpt"),
                                  map_location=DEVICE)
                coder.load_state_dict(ckpt['coder'])
                print("ERROR at", iter_count)
            else:
                loss_queue.put(rd_loss.item())

                if iter_count % 100 == 0:
                    torch.save({
                        'epoch': epoch,
                        'coder': coder.state_dict(),
                        'optim': optim.state_dict()
                    }, os.path.join(args.checkpoint_dir, "model_back.ckpt"))
            optim.step()

            desc = " D: {:.3e}, R: {:.3e}, DY: {:.2e}, L: {:.2e}".format(
                distortion.mean(), rate.mean(), disY.mean(), rd_loss)
            if args.rec_code:
                desc += ", NE: {:.2e}".format(code_err.mean())
            pbar.set_description(desc)

            logger.append('rd_loss', rd_loss)
            logger.append('rate', rate.mean())
            logger.append('distortion', distortion.mean())
            logger.append('disY', disY.mean())
            # logger.append('jac', jac.sum(1).mean())

            log_writer.add_scalar(
                'Trian/total', logger['rd_loss'][-1], iter_count)
            log_writer.add_scalar('Trian/rate', logger['rate'][-1], iter_count)
            log_writer.add_scalar('Trian/distortion',
                                  logger['distortion'][-1], iter_count)
            log_writer.add_scalar('Trian/Y_distortion',
                                  logger['disY'][-1], iter_count)
            iter_count += 1

            if flag:
                exit(0)

        losses = logger.mean()

        log_writer.add_scalar('Loss/total', losses['rd_loss'], epoch)
        log_writer.add_scalar('Loss/rate', losses['rate'], epoch)
        log_writer.add_scalar('Loss/distortion', losses['distortion'], epoch)
        log_writer.add_scalar('Loss/Y_distortion', losses['disY'], epoch)

        if epoch % 100 == 0:
            torch.save({
                'epoch': epoch,
                'coder': coder.state_dict(),
                'optim': optim.state_dict()
            }, os.path.join(args.checkpoint_dir, f"model_{epoch}.ckpt"))

        # Testing
        if not args.verbose:
            print("Model ID:: {}, Epoch: {}/{}:".format(
                model_ID, epoch, args.max_epochs))
            if epoch % 15 == 0:
                lmdas = args.lmda
                lmdas = lmdas + np.exp(np.random.uniform(np.log(np.min(lmdas)), np.log(np.max(
                    lmdas)), len(lmdas)*2)).tolist()
                eval_value = np.mean(
                    [evaluate(epoch, lmda) for lmda in sorted(lmdas, reverse=True)])
            else:
                eval_value = np.mean(
                    [evaluate(epoch, lmda) for lmda in args.lmda])

        if args.use_lr_scheduler:
            sched.step(eval_value)

        # Save model
        if eval_value < best_eval_value:
            best_eval_value = eval_value

            torch.save({
                'epoch': epoch,
                'coder': coder.state_dict(),
                'optim': optim.state_dict()
            }, os.path.join(args.checkpoint_dir, "model.ckpt"))
        elif eval_value > best_eval_value*1.5:
            ckpt = torch.load(os.path.join(args.checkpoint_dir, "model.ckpt"),
                              map_location=DEVICE)
            coder.load_state_dict(ckpt['coder'])


def compress(args):
    model_ID = os.path.basename(args.checkpoint_dir[:-1])

    coder = trc.get_coder_from_args(args)()
    align = trc.util.Alignment(coder.divisor)

    # custom method for loading last checkpoint
    ckpt = torch.load(os.path.join(args.checkpoint_dir,
                                   "model.ckpt"), map_location='cpu')
    print("========================================================================\n"
          "Loading model checkpoint at directory: ", args.checkpoint_dir,
          "\n========================================================================")

    try:
        coder.load_state_dict(ckpt['coder'])
    except RuntimeError as e:
        # Warning(e)
        print(e)
        coder.load_state_dict(ckpt['coder'], strict=False)

    coder = coder.to(DEVICE)

    print("Model ID:: {}".format(model_ID))

    # Create input data pipeline.
    test_dataset = CustomData(args.source_dir, transforms.ToTensor())
    test_dataloader = DataLoader(test_dataset, 1, False, num_workers=16)

    coder.eval()

    os.makedirs(args.target_dir, exist_ok=True)

    est_rate_list = []
    rate_list = []

    with torch.no_grad():
        for eval_img, img_path in test_dataloader:
            img_name = os.path.basename(img_path[0])
            file_name = os.path.join(args.target_dir, img_name + ".ifc")
            t0 = time.perf_counter()
            eval_img = eval_img.to(DEVICE)

            with trc.util.BitStreamIO(file_name, 'w') as fp:
                stream_list, shape_list = coder.compress(align.align(eval_img))
                fp.write(stream_list, [eval_img.size()]+shape_list)

            encode_time = time.perf_counter() - t0

            if args.eval:
                _, eval_est_likelihoods = coder(eval_img)
                eval_est_rate = trc.estimate_bpp(
                    eval_est_likelihoods, input=eval_img)

                eval_rate = os.path.getsize(
                    file_name) * 8 / (eval_img.size(2) * eval_img.size(3))

                print("{}:: est.rate: {:.4f} rate: {:.4f}/{:.3f}(s)".format(
                    img_name, eval_est_rate.item(), eval_rate, encode_time))

                est_rate_list.append(eval_est_rate.item())
                rate_list.append(eval_rate)

        if args.eval:
            print("==========avg. performance==========")
            print("est.rate: {:.4f} rate: {:.4f}".format(
                np.mean(est_rate_list),
                np.mean(rate_list)
            ))


def decompress(args):
    model_ID = os.path.basename(args.checkpoint_dir[:-1])

    coder = trc.get_coder_from_args(args)()
    align = trc.util.Alignment(coder.divisor)

    # custom method for loading last checkpoint
    ckpt = torch.load(os.path.join(args.checkpoint_dir,
                                   "model.ckpt"), map_location='cpu')
    print("========================================================================\n"
          "Loading model checkpoint at directory: ", args.checkpoint_dir,
          "\n========================================================================")

    coder.load_state_dict(ckpt['coder'])
    coder = coder.to(DEVICE)

    print("Model ID:: {}".format(model_ID))

    coder.eval()

    os.makedirs(args.target_dir, exist_ok=True)
    file_name_list = sorted(glob(os.path.join(args.source_dir, "*.ifc")))
    if len(file_name_list) == 0:
        print('compressed file not found in', args.source_dir)
        return

    with torch.no_grad():
        eval_psnr_list = []
        eval_msssim_list = []
        eval_rate_list = []

        for file_name in file_name_list:
            img_name = os.path.basename(file_name)[:-4]
            save_name = os.path.join(args.target_dir, img_name)
            t0 = time.perf_counter()

            with trc.util.BitStreamIO(file_name, 'r') as fp:
                stream_list, shape_list = fp.read_file()
                eval_img_tilde = coder.decompress(stream_list, shape_list[-1])
                eval_img_tilde = align.resume(
                    eval_img_tilde, shape=shape_list[0])

            decode_time = time.perf_counter() - t0
            save_image(eval_img_tilde, save_name)

            if args.eval:
                eval_img = io.imread(os.path.join(args.original_dir, img_name))
                eval_img_np = io.imread(save_name)

                eval_psnr = PSNR_np(eval_img, eval_img_np, data_range=255.)
                eval_msssim = MultiScaleSSIM(eval_img[None], eval_img_np[None])
                eval_rate = os.path.getsize(
                    file_name) * 8 / (eval_img.shape[0] * eval_img.shape[1])

                print("{}:: PSNR: {:2.4f}, MS-SSIM: {:.4f}, rate: {:.4f}/{:.3f}(s)".format(
                    img_name, eval_psnr, eval_msssim, eval_rate, decode_time
                ))

                eval_psnr_list.append(eval_psnr)
                eval_msssim_list.append(eval_msssim)
                eval_rate_list.append(eval_rate)

        if args.eval:
            print("==========avg. performance==========")
            print("PSNR: {:.4f}, MS-SSIM: {:.4f}, rate: {:.4f}".format(
                np.mean(eval_psnr_list),
                np.mean(eval_msssim_list),
                np.mean(eval_rate_list),
            ))


def parse_args(argv):
    """Parses command line arguments."""
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # High-level options.
    trc.add_coder_args(parser)
    parser.add_argument(
        "--var_filters", "-VNFL", type=int, nargs='+',
        help="variable filters.")
    parser.add_argument(
        "--num_layers", "-L", type=int, default=2,
        help="Layers of ANF.")
    parser.add_argument(
        "--use_DQ", "-DQ", action="store_true",
        help="Use DequantizationModule.")
    parser.add_argument(
        "--use_attn", "-ATTN", action="store_true",
        help="Use NonLocalAttentionModule.")
    parser.add_argument(
        "--input_norm", "-IN", type=str, default="None", choices=["None", "shift", "scale"],
        help="Use InputNormalize.")
    parser.add_argument(
        "--use_syntax", "-US", action="store_true",
        help="Use Syntax coder.")
    parser.add_argument(
        "--use_AQ", "-UA", action="store_true",
        help="Use AdaptiveQuant.")
    parser.add_argument(
        "--syntax_prior", "-SP", type=str, default="None", choices=["None", "Factorized"],
        help="Prior of syntax coder.")
    parser.add_argument(
        "--ch_wise", "-CW", action="store_true",
        help="Use AdaptiveQuant.")
    parser.add_argument(
        "--cond_ver", "-CV", type=int, default=2,
        help="Version of CConv.")
    parser.add_argument(
        "--share_weight", "-SW", action="store_true",
        help="Share weight of layers.")
    parser.add_argument(
        "--init_code", type=str, default='gaussian',
        help="init code distribution.")
    parser.add_argument(
        "--disable_code", "-DC", action="store_true",
        help="Disable code(multiply).")
    parser.add_argument(
        "--verbose", "-V", action="store_true",
        help="Report bitrate and distortion when training or compressing.")
    parser.add_argument(
        "--checkpoint_dir", "-ckpt", default=None,
        help="Directory where to save/load model checkpoints.")

    subparsers = parser.add_subparsers(
        title="commands", dest="command",
        help="What to do: \n"
             "'train' loads training data and trains (or continues to train) a new model.\n"
             "'compress' reads an image file (lossless PNG format) and writes a compressed binary file.\n"
             "'decompress' reads a binary file and reconstructs the image (in PNG format).\n"
             "input and output filenames need to be provided for the latter two options.\n\n"
             "Invoke '<command> -h' for more information.")

    # 'train' sub-command.
    train_cmd = subparsers.add_parser(
        "train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Trains (or continues to train) a new model.")
    train_cmd.add_argument(
        "--batch_size", type=int, default=32,  # OPTIM
        help="Batch size for training.")
    train_cmd.add_argument(
        "--patch_size", type=int, default=256,  # OPTIM
        help="Size of image patches for training.")
    train_cmd.add_argument(
        "--metric", type=str, default='PSNR',
        choices=['PSNR', 'SSIM'], help="Training dataset.")
    train_cmd.add_argument(
        "--use_huber", "-HU", action="store_true",
        help="Use huber_loss or not.")
    train_cmd.add_argument(
        "--lambda", type=float, default=0.01, dest="lmda", nargs='+',
        help="Lambda for rate-distortion trade-off.")
    train_cmd.add_argument(
        "--lr", type=float, default=1e-4, dest="lr",  # OPTIM
        help="Learning rate.")
    train_cmd.add_argument(
        "--max_epochs", type=int, default=2000,
        help="Train up to this number of steps.")
    train_cmd.add_argument(
        '--clip_max_norm', default=0.1, type=float,  # OPTIM
        help='gradient clipping max norm')
    train_cmd.add_argument(
        "--dataset", type=str, default='VIMEO',
        help="Training dataset.")
    train_cmd.add_argument(
        "--optim", type=str, default="Adam", choices=["Adam", "AdamW", "RAdam"],
        help="Optimizer type.")
    train_cmd.add_argument(
        "--use_lr_scheduler", "-LRS", action="store_true",
        help="Use lr_scheduler or not.")
    train_cmd.add_argument(
        "--alt_mode", type=str, default="None", choices=["None", "outloop", "inloop", "Force"],
        help="ALT mode.")
    train_cmd.add_argument(
        "--alt_step", type=int, default=1,
        help="Train up to this number of steps then change mode.")
    train_cmd.add_argument(
        '--weight_gamma', default=1, type=float,
        help='gradient clipping max norm')
    train_cmd.add_argument(
        "--cal_yerr", action="store_true",
        help="Calculate Y error or not.")
    train_cmd.add_argument(
        "--cal_jac", action="store_true",
        help="Calculate jacobian or not.")
    train_cmd.add_argument(
        "--enc_optim", action="store_true",
        help="Encoding optimize or not.")
    train_cmd.add_argument(
        "--enc_noise", action="store_true",
        help="Encoding white noise or not.")
    train_cmd.add_argument(
        "--fix_ANCHOR", action="store_true",
        help="fix base ANF coder.")
    train_cmd.add_argument(
        "--fix_ANF", action="store_true",
        help="fix main ANF coder.")
    train_cmd.add_argument(
        "--IDQ", action="store_true",
        help="Input dequant noise.")
    train_cmd.add_argument(
        "--rec_code", action="store_true",
        help="Use latent error or not.")
    train_cmd.add_argument(
        "--rev_ng", action="store_true",
        help="reverse with nograd or not.")
    train_cmd.add_argument(
        "--detach_enc", action="store_true",
        help="reverse with nograd or not.")
    train_cmd.add_argument(
        "--pretrained", action="store_true",
        help="Use pretrained CConv.")
    train_cmd.add_argument(
        "--cond_mode", type=str, default="normal", choices=["normal", "rand"],
        help="CConv mode.")
    train_cmd.add_argument(
        "--parallel", action="store_true",
        help="Use DataParallel or not.")
    train_cmd.add_argument(
        "--check_grad", action="store_true",
        help="Check gradient or not.")
    train_cmd.add_argument(
        "--resume", action="store_true",
        help="Whether to resume on previous checkpoint")
    train_cmd.add_argument(
        "--reuse_ckpt", action="store_true",
        help="Whether to reuse the specified checkpoint, if not, "
             "we'll resume the current model in the new created directory.")
    train_cmd.add_argument(
        "--reuse_args", action="store_true",
        help="Whether to reuse the original arguments")

    # 'compress' sub-command.
    compress_cmd = subparsers.add_parser(
        "compress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compress images with a trained model.")
    compress_cmd.add_argument(
        "--source_dir", "-SD",
        help="The directory of the images that are expected to compress.")
    compress_cmd.add_argument(
        "--target_dir", "-TD",
        help="The directory where the compressed files are expected to store at.")
    compress_cmd.add_argument(
        "--eval", action="store_true",
        help="Evaluate compressed images with original ones.")

    # 'decompress' sub-command.
    decompress_cmd = subparsers.add_parser(
        "decompress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Decompress bitstreams with a trained model.")
    decompress_cmd.add_argument(
        "--source_dir", "-SD",
        help="The directory of the compressed files that are expected to decompress.")
    decompress_cmd.add_argument(
        "--target_dir", "-TD",
        help="The directory where the images are expected to store at.")
    decompress_cmd.add_argument(
        "--eval", action="store_true",
        help="Evaluate decompressed images with original ones.")
    decompress_cmd.add_argument(
        "--original_dir", "-OD", nargs="?",
        help="The directory where the original images are expected to store at.")

    # Parse arguments.
    args = parser.parse_args(argv[1:])
    if args.command is None:
        parser.print_usage()
        sys.exit(2)
    return args


def main(args):
    np.set_printoptions(threshold=sys.maxsize)

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # Invoke subcommand.
    if args.command == "train":
        torch.backends.cudnn.benchmark = True
        if args.checkpoint_dir is None:
            args.checkpoint_dir = gen_log_folder_name(
                root=LogRoot, prefix=args.architecture+"_")

        if not args.resume:
            assert args.checkpoint_dir is not None
            print("========================================================================\n"
                  "Creating model checkpoint at directory: ", args.checkpoint_dir,
                  "\n========================================================================")
            os.makedirs(args.checkpoint_dir, exist_ok=False)

        else:
            assert args.checkpoint_dir is not None, "Checkpoint directory must be specified. [-ckpt=path/to/your/model]"
            if args.reuse_args:
                old_args = load_args(args, args.checkpoint_dir)
                old_args.resume = True
                old_args.reuse_ckpt = args.reuse_ckpt
                args = old_args

            if not args.reuse_ckpt:
                old_ckpt_dir = args.checkpoint_dir
                args.checkpoint_dir = gen_log_folder_name(
                    root=LogRoot, prefix=args.architecture+"_")
                os.makedirs(args.checkpoint_dir, exist_ok=False)

                print("========================================================================\n"
                      "Creating model checkpoint at directory: ", args.checkpoint_dir,
                      "\nCopying model checkpoint from ", old_ckpt_dir)

                assert os.path.exists(os.path.join(old_ckpt_dir, "model.ckpt")), \
                    os.path.join(old_ckpt_dir, "model.ckpt") + " not exist"
                shutil.copy(os.path.join(old_ckpt_dir, "model.ckpt"),
                            os.path.join(args.checkpoint_dir, "model.ckpt"))
            else:
                print(
                    "========================================================================\n")

        # Config dump
        dump_args(args, args.checkpoint_dir)
        train(args)
    else:
        torch.backends.cudnn.deterministic = True
        if args.command == "compress":
            compress(args)
        elif args.command == "decompress":
            decompress(args)


if __name__ == '__main__':
    app.run(main, flags_parser=parse_args)
