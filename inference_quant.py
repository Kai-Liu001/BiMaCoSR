import os
import sys
import argparse
from pathlib import Path

from omegaconf import OmegaConf
from sampler import Sampler

from utils.util_opts import str2bool
from basicsr.utils.download_util import load_file_from_url
import torch

# Reset GPU memory statistics at the start of the program
torch.cuda.reset_peak_memory_stats()

def get_configs(args):
    if args.config is None:
        if args.task == "SinSR":
            configs = OmegaConf.load('./configs/SinSR.yaml')
        elif args.task == 'realsrx4':
            configs = OmegaConf.load('./configs/realsr_swinunet_realesrgan256.yaml')
    else:
        configs = OmegaConf.load(args.config)
    # prepare the checkpoint
    ckpt_dir = Path('./weights')
    if args.ckpt is None:
        if not ckpt_dir.exists():
            ckpt_dir.mkdir()
        if args.task == "SinSR":
            ckpt_path = ckpt_dir / f'SinSR_v1.pth'
        elif args.task == 'realsrx4':
            ckpt_path = ckpt_dir / f'resshift_{args.task}_s{args.steps}_v1.pth'
    else:
        ckpt_path = Path(args.ckpt)
    print(f"[INFO] Using the checkpoint {ckpt_path}")
    
    if not ckpt_path.exists():
        if args.task == "SinSR":
            load_file_from_url(
                url=f"https://github.com/wyf0912/SinSR/releases/download/v1.0/{ckpt_path.name}",
                model_dir=ckpt_dir,
                progress=True,
                file_name=ckpt_path.name,
                )
        else:
            load_file_from_url(
                url=f"https://github.com/zsyOAOA/ResShift/releases/download/v2.0/{ckpt_path.name}",
                model_dir=ckpt_dir,
                progress=True,
                file_name=ckpt_path.name,
                )
    vqgan_path = ckpt_dir / f'autoencoder_vq_f4.pth'
    if not vqgan_path.exists():
         load_file_from_url(
            url="https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth",
            model_dir=ckpt_dir,
            progress=True,
            file_name=vqgan_path.name,
            )

    configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.timestep_respacing = args.infer_steps
    configs.diffusion.params.sf = args.scale
    configs.autoencoder.ckpt_path = str(vqgan_path)

    # save folder
    if not Path(args.out_path).exists():
        Path(args.out_path).mkdir(parents=True)

    if args.chop_size == 512:
        chop_stride = 448
    elif args.chop_size == 256:
        chop_stride = 224
    else:
        raise ValueError("Chop size must be in [512, 384, 256]")

    return configs, chop_stride

def main():
    parser = argparse.ArgumentParser(description="Inference script for BiMaCoSR")
    parser.add_argument('--in_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--out_path', type=str, required=True, help='Path to save the output image')
    parser.add_argument('--config', type=str, default=None, help='Path to the config file')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to the checkpoint file')
    parser.add_argument('--task', type=str, default="SinSR", help='Task name')
    parser.add_argument('--steps', type=int, default=15, help='Number of steps')
    parser.add_argument('--infer_steps', type=int, default=15, help='Number of inference steps')
    parser.add_argument('--scale', type=int, default=4, help='Scale factor')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--one_step', type=str2bool, default=True, help='Use one-step inference')
    parser.add_argument('--chop_size', type=int, default=256, help='Chop size')
    parser.add_argument('--ddim', type=str2bool, default=False, help='Use DDIM')
    parser.add_argument('--ref_path', type=str, default=None, help='ref path')

    args = parser.parse_args()

    configs, chop_stride = get_configs(args)

    resshift_sampler = Sampler(
            configs,
            chop_size=args.chop_size,
            chop_stride=chop_stride,
            chop_bs=1,
            use_fp16=False,
            seed=args.seed,
            ddim=args.ddim
            )

    resshift_sampler.inference(args.in_path, args.out_path, bs=1, noise_repeat=False, one_step=args.one_step)
    import evaluate
    evaluate.evaluate(args.out_path, args.ref_path, None)
    
    
if __name__ == '__main__':
    main()
    # Print peak memory usage (allocated memory)
    print(f"Max allocated memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    # Print peak memory usage (total memory, including allocated but unused memory)
    print(f"Max reserved memory: {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB")