# PyTorch tools
import argparse
import os
import shutil
import sys
import time
from functools import partial
from glob import glob

# Python tools
import numpy as np
import torch
import torch_compression as trc
import torchvision
from absl import app
from absl.flags import argparse_flags
# - Metrics
from skimage import io
from torch import nn
from torch.nn.parallel import data_parallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

# - Tools
try:
    from util.auto_helper import cptree, ffilter, mkdir
    from util.datasets import MSCOCO, CLICTrain, CustomData, Kodak, Vimeo90K
    from util.log_manage import (AvgMeter, dump_args, gen_log_folder_name,
                                 load_args)
    from util.msssim import MultiScaleSSIM
    from util.psnr import PSNR, PSNR_np
    from util.ssim import MS_SSIM
except:
    from .util.auto_helper import cptree, ffilter, mkdir
    from .util.datasets import MSCOCO, CLICTrain, CustomData, Kodak, Vimeo90K
    from .util.log_manage import (AvgMeter, dump_args, gen_log_folder_name,
                                  load_args)
    from .util.msssim import MultiScaleSSIM
    from .util.psnr import PSNR, PSNR_np
    from .util.ssim import MS_SSIM


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

    archi = trc.hub.AugmentedNormalizedHyperPriorCoder2 if args.anf_archi == "HL1" else trc.AugmentedNormalizedHyperPriorCoder
    coder = archi(
        args.var_filters, args.num_features, args.num_hyperpriors,
        num_layers=args.num_layers, use_DQ=args.use_DQ, share_wei=args.share_weight,
        init_code=args.init_code, use_code=not args.disable_code, dec_add=args.alt_mode != "None", use_context=args.use_context, condition=args.condition, quant_mode=args.quant_mode)

    for n, m in coder.named_modules():
        if hasattr(m, "clamp"):
            # print(n, m.clamp)
            if "hyper" in n:
                m.clamp = args.hscale_clamp
            else:
                m.clamp = args.scale_clamp
    log_writer = SummaryWriter(log_dir=args.checkpoint_dir)
    ft = ffilter(keep=['*.py', '*.txt'],
                 ignore=['tmp/', 'results/', 'models/',
                         '.vscode/', '.ipynb_checkpoints/', '.git/', 'src/'])
    cptree('./', mkdir(args.checkpoint_dir+'/src'), ffilter=ft, param='-va')
    print(coder)
    print(args)
    log_writer.add_text('Config', str(vars(args)))
    log_writer.add_text('Architecture', str(coder))

    optim = torch.optim.Adam([dict(params=coder.main_parameters(), lr=args.lr),
                              dict(params=coder.aux_parameters(), lr=args.lr*10)])  # OPTIM
    sched = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=[700, 1300], gamma=0.5)  # OPTIM

    psnr = PSNR(reduction='mean', data_range=255.)
    ms_ssim = MS_SSIM(reduction='mean', data_range=255.).to(DEVICE)

    @torch.no_grad()
    def evaluate(epoch, eval_lmda=None, is_resume=False):
        coder.eval()

        logger = AvgMeter(
            ['psnr', 'ssim', 'rate', 'aux_loss', 'rate_y', 'rate_z', 'rd_loss'])

        for idx, eval_imgs in enumerate(validate_dataloader):
            eval_imgs = eval_imgs.to(DEVICE)

            eval_imgs_tilde, eval_likelihoods, Y_error, jac, _, _ = coder(
                eval_imgs, visual=epoch % 3 == 0, figname=os.path.join(args.checkpoint_dir, "kodak_%02d" % (idx+1)))
            # disY = Y_error.pow(2).flatten(1).mean(1)
            # jac = torch.stack(jac, 1) if len(jac) else imgs.new_zeros(
            #     imgs.size(0), 1)

            eval_rate_y = trc.estimate_bpp(
                eval_likelihoods[0], input=eval_imgs)
            eval_rate_z = trc.estimate_bpp(
                eval_likelihoods[1], input=eval_imgs)
            eval_rate = eval_rate_y + eval_rate_z

            eval_aux_loss = coder.aux_loss()

            if args.metric == 'PSNR':
                distortion = (
                    eval_imgs - eval_imgs_tilde).pow(2).flatten(1).mean(1)
                eval_rd_loss = torch.mean(
                    (args.lmda * 255**2) * distortion + eval_rate)
            elif args.metric == 'SSIM':
                distortion = 1 - ms_ssim_criterion(eval_imgs_tilde, eval_imgs)
                eval_rd_loss = torch.mean(args.lmda * distortion + eval_rate)

            eval_imgs = eval_imgs.mul(255).clamp(0, 255).round()
            eval_imgs_tilde = eval_imgs_tilde.mul(255).clamp(0, 255).round()

            eval_psnr = psnr(eval_imgs, eval_imgs_tilde)
            eval_msssim = ms_ssim(eval_imgs, eval_imgs_tilde)

            logger.append('psnr', eval_psnr)
            logger.append('ssim', eval_msssim)
            logger.append('rate', eval_rate)
            logger.append('aux_loss', eval_aux_loss)
            logger.append('rate_y', eval_rate_y)
            logger.append('rate_z', eval_rate_z)
            logger.append('rd_loss', eval_rd_loss)

            # if not is_resume and idx == 6:
            save_image(eval_imgs_tilde.div(255),
                       os.path.join(args.checkpoint_dir, "kodak_%02d.png" % (idx+1)))

        eval = logger.mean()

        with open(os.path.join(args.checkpoint_dir, "eval.txt"), 'w') as fp:
            fp.write(
                'name\tpsnr\tssim\trate\trate_y\trate_z\trate_y_per\trate_z_per\n')
            for idx in range(len(logger)):
                eval_items = logger[idx]
                fp.write("kodak_{idx:02d}:\t\t{psnr:.2f}\t{ssim:.4f}\t{rate:.2f}\t{rate_y:.2f}\t{rate_z:.2f}\t{perc_y:.2f}\t{perc_z:.2f}\n".format(
                    idx=idx+1, **eval_items, perc_y=eval_items['rate_y']/eval_items['rate'], perc_z=eval_items['rate_z']/eval_items['rate']))
            fp.write("AVG.\n")
            fp.write("kodak:\t\t{psnr:.2f}\t{ssim:.4f}\t{rate:.2f}\t{rate_y:.2f}\t{rate_z:.2f}\t{perc_y:.2f}\t{perc_z:.2f}\n".format(
                **eval, perc_y=eval['rate_y']/eval['rate'], perc_z=eval['rate_z']/eval['rate']))

        print("lmda={:.1e}::".format(eval_lmda) if eval_lmda is not None else "",
              "PSNR: {psnr:.4f}, MS-SSIM: {ssim:.4f}, rate: {rate:.4f}, aux: {aux_loss:.4f}".format(**eval))

        if not is_resume:
            log_writer.add_scalar('Loss/aux', eval['aux_loss'], epoch)
            log_writer.add_scalar('Evaluation/PSNR', eval["psnr"], epoch)
            log_writer.add_scalar('Evaluation/MS-SSIM', eval["ssim"], epoch)
            log_writer.add_scalar('Evaluation/est-rate', eval["rate"], epoch)

        return eval['rd_loss']

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
            coder.load_state_dict(ckpt['coder'])
        except RuntimeError as e:
            # Warning(e)
            print(e)
            coder.load_state_dict(ckpt['coder'], strict=False)

        coder = coder.to(DEVICE)

        optim.load_state_dict(ckpt['optim'])
        if 'sched' in ckpt:
            sched.load_state_dict(ckpt['sched'])

        if not args.verbose:
            evaluate(0, is_resume=True)

        print("Latest checkpoint restored, start training at step {}.".format(start_epoch))

    else:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        coder = coder.to(DEVICE)

    if args.metric == 'SSIM':
        ms_ssim_criterion = MS_SSIM(data_range=1.).to(DEVICE)

    model_forward = partial(data_parallel, coder) if args.parallel else coder
    main_parameters = coder.main_parameters()
    if args.alt_mode == "Force":
        switch_mode(coder, args, 0, args.alt_step)

    best_eval_value, mode = 1e9, 0
    for epoch in range(start_epoch, args.max_epochs):
        coder.train()

        logger = AvgMeter(
            ['rd_loss', 'distortion', 'rate', 'disY', 'jac'])

        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        # for loop going through dataset
        if args.alt_mode == "outloop":
            mode = switch_mode(coder, args, epoch, mode)
        iter = 0
        for imgs in pbar:
            imgs = imgs.to(DEVICE)

            if args.alt_mode == 'inloop':
                mode = switch_mode(coder, args, iter, mode)

            optim.zero_grad()
            if args.parallel:
                imgs_tilde, likelihoods, Y_error, jac, y_err, z_err = model_forward(
                    imgs, module_kwargs=dict(jac=args.cal_jac, rev_ng=args.rev_ng, rec_code=args.rec_code, IDQ=args.IDQ))
            else:
                imgs_tilde, likelihoods, Y_error, jac, y_err, z_err = model_forward(
                    imgs, jac=args.cal_jac, rev_ng=args.rev_ng, rec_code=args.rec_code, IDQ=args.IDQ)
            imgs_tilde = trc.util.bound(imgs_tilde, 0, 1)
            disY = Y_error.pow(2).flatten(1).mean(1)
            if not args.cal_jac:
                jac = imgs.new_zeros(imgs.size(0), 1)

            if args.rec_code:
                y_err = y_err.pow(2).flatten(1).mean(1)
                z_err = z_err.pow(2).flatten(1).mean(1)
                code_err = y_err + z_err
            else:
                code_err = imgs.new_zeros(imgs.size(0))

            rate = trc.estimate_bpp(likelihoods, input=imgs)

            if args.metric == 'PSNR':
                distortion = (imgs - imgs_tilde).pow(2).flatten(1).mean(1)
                flag = False
                if torch.isnan(distortion).any() or (epoch > 3 and distortion.gt(100).any()):
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
                    rd_loss = torch.zeros_like(distortion).sum()
                else:
                    rd_loss = (args.lmda * 255**2) * distortion + rate
                    if args.cal_yerr:
                        rd_loss += (255**2 * 0.01) * disY
                    if args.cal_jac:
                        rd_loss -= (args.lmda * 255**2 * 1e-5) * jac.sum(1)
                    if args.rec_code:
                        rd_loss += code_err
                    rd_loss = (rd_loss * mask).sum() / mask.sum()
            elif args.metric == 'SSIM':
                distortion = 1 - ms_ssim_criterion(imgs_tilde, imgs)
                rd_loss = torch.mean(args.lmda * distortion + rate)

            aux_loss = coder.aux_loss()

            rd_loss.add(aux_loss).backward()

            if iter % 500 == 0:
                trc.vision.plot_grad_flow(coder.named_main_parameters(
                ), os.path.join(args.checkpoint_dir, "gradflow.png"))
            if args.clip_max_norm > 0:
                nn.utils.clip_grad_norm_(main_parameters, args.clip_max_norm)
            if iter % 500 == 0:
                trc.vision.plot_grad_flow(coder.named_main_parameters(
                ), os.path.join(args.checkpoint_dir, "gradflow_clip.png"))
            if args.check_grad:
                trc.AugmentedNormalizedFlows.dump_grad(
                    os.path.join(grad_log, f"grad_log_{iter}.txt"))
            optim.step()

            desc = " dis: {:.3e}, rate: {:.3e}".format(
                distortion.mean(), rate.mean())
            if args.rec_code:
                desc += ", code_err: {:.2e}".format(code_err.mean())
            pbar.set_description(desc)

            logger.append('rd_loss', rd_loss)
            logger.append('rate', rate.mean())
            logger.append('distortion', distortion.mean())
            logger.append('disY', disY.mean())
            logger.append('jac', jac.sum(1).mean())

            if flag:
                exit(0)

            if iter % 500 == 0:
                desc = "disY: {:.2e}, jac: {:.2e}".format(
                    disY.mean(), jac.sum(1).mean())
                if args.rec_code:
                    desc += ", y_err: {:.2e}, z_err: {:.2e}".format(
                        y_err.mean(), z_err.mean())
                print(desc)
                if args.cal_jac:
                    print(jac.mean(0).detach().cpu().numpy())

            iter += 1

        losses = logger.mean()

        log_writer.add_scalar('Loss/total', losses['rd_loss'], epoch)
        log_writer.add_scalar('Loss/rate', losses['rate'], epoch)
        log_writer.add_scalar('Loss/distortion', losses['distortion'], epoch)
        log_writer.add_scalar('Loss/Y_distortion', losses['disY'], epoch)

        # sched.step()

        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'coder': coder.state_dict(),
                'optim': optim.state_dict()
            }, os.path.join(args.checkpoint_dir, f"model_{epoch}.ckpt"))

        # Testing
        if not args.verbose:
            print("Model ID:: {}, Epoch: {}/{}:".format(
                model_ID, epoch, args.max_epochs))
            eval_value = evaluate(epoch)

        # Save model
        if eval_value < best_eval_value:
            best_eval_value = eval_value

            torch.save({
                'epoch': epoch,
                'coder': coder.state_dict(),
                'optim': optim.state_dict()
            }, os.path.join(args.checkpoint_dir, "model.ckpt"))


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
        "--anf_archi", type=str, default="default", choices=['default', "HL1"],
        help="Layers of ANF.")
    parser.add_argument(
        "--var_filters", "-VNFL", type=int, nargs='+',
        help="variable filters.")
    parser.add_argument(
        "--scale_clamp", "-SC", type=int, default=1,
        help="Scale clamp of ANF.")
    parser.add_argument(
        "--hscale_clamp", "-HSC", type=int, default=1,
        help="Scale clamp of ANF.")
    parser.add_argument(
        "--num_layers", "-L", type=int, default=2,
        help="Layers of ANF.")
    parser.add_argument(
        "--use_DQ", "-DQ", action="store_true",
        help="Use DequantizationModule.")
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
        "--lambda", type=float, default=0.01, dest="lmda",
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
        "--alt_mode", type=str, default="None", choices=["None", "outloop", "inloop", "Force"],
        help="ALT mode.")
    train_cmd.add_argument(
        "--alt_step", type=int, default=1,
        help="Train up to this number of steps then change mode.")
    train_cmd.add_argument(
        "--cal_yerr", action="store_true",
        help="Calculate Y error or not.")
    train_cmd.add_argument(
        "--cal_jac", action="store_true",
        help="Calculate jacobian or not.")
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
        "--parallel", action="store_true",
        help="Use DataParallel or not.")
    train_cmd.add_argument(
        "--check_grad", action="store_true",
        help="Check gradient or not.")
    train_cmd.add_argument(
        "--use_radam", action="store_true",
        help="Use RAdam or not.")
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
