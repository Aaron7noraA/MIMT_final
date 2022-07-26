import argparse
import os

import torch_compression as trc

name_list = ['BasketballDrive', 'BQTerrace', 'Cactus', 'Kimono1', 'ParkScene',
             'Beauty', 'Bosphorus', 'HoneyBee', 'Jockey', 'ReadySteadyGo', 'ShakeNDry', 'YachtRide']
frame_list = [100, 100, 100, 100, 100, 600, 600, 600, 600, 600, 300, 600]
GOP_list = [10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 12, 12]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("load_dir")
    parser.add_argument("load_file")
    trc.add_coder_args(parser)
    parser.add_argument("--lmda", type=int, default=-1)
    parser.add_argument("--name", default='BasketballPass')
    parser.add_argument("--frame", type=int, default=-1)
    parser.add_argument("--GOP", type=int, default=-1)
    parser.add_argument("--mode", default='PSNR', choices=['PSNR', 'MS-SSIM'])
    parser.add_argument("--metric", default='PSNR',
                        choices=['PSNR', 'MS-SSIM'])
    parser.add_argument("--visual", action="store_true",
                        help="Save Flow etc. or not.")

    parser.add_argument("--ME", type=str, default='PWC',
                        choices=['SPy', 'PWC', 'PWC3d'])
    parser.add_argument("--use_occlution", type=int, default=0,
                        choices=[0, 1])
    parser.add_argument("--MC", type=str, default='MC',
                        choices=['MC', 'woMC'])
    parser.add_argument("--scale_space", type=int, default=0,
                        choices=[0, 1])
    parser.add_argument("--use_flow", type=int, default=0,
                        choices=[0, 1])
    parser.add_argument("--stop_flow", type=int, default=0,
                        choices=[0, 1])
    parser.add_argument("--use_Intra", type=int, default=0,
                        choices=[0, 1, 2])
    parser.add_argument("--Icodec", type=str, default='BPG',
                        choices=['BPG', '265'])
    parser.add_argument("--scale_factor", "-SF", type=float, default=1.)
    args = parser.parse_args()
    kwargs = vars(args)

    frames = [args.frame]
    if args.name == 'HEVC':
        names = name_list[:5]
        if args.frame < 0:
            frames = frame_list[:5]
        GOPs = GOP_list[:5]
    elif args.name == 'UVG':
        names = name_list[5:14]
        if args.frame < 0:
            frames = frame_list[5:14]
        GOPs = GOP_list[5:14]
    elif args.name == 'all':
        names = name_list
        if args.frame < 0:
            frames = frame_list
        GOPs = GOP_list
    else:
        names = [args.name]
        GOPs = [dict(zip(name_list, GOP_list))[args.name]]
        frames = [dict(zip(name_list, frame_list))[args.name]]

    command = "python3 torchDVC_test.py {load_dir} {load_file} ".format(
        **kwargs)
    command += "-ARCHI {architecture} -NF {num_features} -NFL {num_filters} -NHP {num_hyperpriors} ".format(
        **kwargs)
    if args.Mean:
        command += "-M "
    command += "--mode {mode} --metric {metric} ".format(
        **kwargs)
    if args.visual:
        command += "--visual "
    command += "--ME {ME} --use_occlution {use_occlution} --MC {MC} --scale_space {scale_space} --use_flow {use_flow} --stop_flow {stop_flow} ".format(
        **kwargs)
    command += "--use_Intra {use_Intra} --Icodec {Icodec} -SF {scale_factor} ".format(**kwargs)
    command += "--name {name} --frame {frame} --GOP {GOP}"

    for name, frame, GOP in zip(names, frames, GOPs):
        command_ = command.format(name=name, frame=frame, GOP=GOP)
        print(command_)
        os.system(command_)
