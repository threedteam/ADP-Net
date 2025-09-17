import argparse
import os
import random
from typing import Dict, Any, Tuple

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ADP.data.dataset import ADPDataset
from ADP.data.parse_config import ConfigParser
from ADP.networks.generators import ADPNet


def setup_environment(seed: int, gpu_id: int) -> torch.device:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.set_device(gpu_id)
    cudnn.benchmark = True
    print(f"Environment set up: Using GPU:{gpu_id}, Global random seed: {seed}")
    return torch.device(f"cuda:{gpu_id}")


def prepare_dataloader(config: Dict[str, Any]) -> DataLoader:
    print("Preparing data loader...")
    val_dataset = ADPDataset(
        data_args=config['dataset'],
        flist_path=config['val_flist'],
        mask_path=None,
        batch_size=config['batch_size'],
        augment=False,
        training=False,
        test_mask_path=config['test_mask_flist'],
        world_size=1,
        trainer=config['trainer']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )
    print("Data loader is ready.")
    return val_loader


def load_model_and_checkpoint(config: Dict[str, Any], args: argparse.Namespace, device: torch.device) -> ADPNet:
    print("Loading model and checkpoint...")
    model = ADPNet(config['g_args']).to(device).eval()

    if args.load_pl:
        print(f"Loading PyTorch Lightning checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')['state_dict']
        weights = {k.replace('adp_ema.', ''): v for k, v in checkpoint.items() if k.startswith('adp_ema.')}
        if not weights:
            print("Warning: No EMA weights with 'ema.' prefix found in the checkpoint.")
            weights = {k.replace('adp.', ''): v for k, v in checkpoint.items() if k.startswith('adp.')}
        model.load_state_dict(weights, strict=False)
    else:
        g_ckpt_name = args.g_ckpt or 'G_last.pth'
        resume_path = os.path.join(str(config.resume), g_ckpt_name)
        print(f"Loading standard generator checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location='cpu')
        model.G.load_state_dict(checkpoint.get('generator', {}), strict=False)

    print("Model and checkpoint loaded.")
    return model.requires_grad_(False)


def run_inference(model: ADPNet, dataloader: DataLoader, output_path: str, device: torch.device):
    os.makedirs(output_path, exist_ok=True)
    print(f"Starting inference, results will be saved to: {output_path}")

    with torch.no_grad():
        for items in tqdm(dataloader, desc="Processing images"):
            items_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in items.items()}

            gen_output = model(items_on_device)

            image = items_on_device['image']
            mask = items_on_device['mask']
            final_img = image * (1 - mask) + gen_output * mask

            final_img = torch.clamp(final_img * 255.0, 0, 255)
            final_img_np = final_img.permute(0, 2, 3, 1).byte().cpu().numpy()

            for i in range(final_img_np.shape[0]):
                img_name = items['name'][i]
                save_path = os.path.join(output_path, img_name)
                cv2.imwrite(save_path, final_img_np[i, :, :, ::-1])

    print("Inference complete!")


def parse_arguments() -> Tuple[argparse.Namespace, Dict[str, Any]]:
    parser = argparse.ArgumentParser(description='ADP Inference Script')
    parser.add_argument('-c', '--config', type=str, required=True, help='Config file path.')
    parser.add_argument('-e', '--exp_name', type=str, help='Experiment name.')
    parser.add_argument('-r', '--resume', type=str, help='Path to latest checkpoint.')
    parser.add_argument('--mae_ckpt', type=str, help='MAE checkpoint path.')
    parser.add_argument('--g_ckpt', type=str, help='Generator checkpoint name (e.g., G_last.pth).')
    parser.add_argument('--output_path', type=str, default='./outputs', help='Output directory path.')
    parser.add_argument('--load_pl', action='store_true', help='Load a PyTorch Lightning checkpoint.')
    parser.add_argument('--image_size', type=int, default=256, help='Test image size.')

    args = parser.parse_args()
    config = ConfigParser.from_args(args, mkdir=False)
    return args, config


def main():
    args, config = parse_arguments()
    device = setup_environment(seed=3407, gpu_id=0)
    val_loader = prepare_dataloader(config)
    model = load_model_and_checkpoint(config, args, device)
    eval_path = config.get('test_outputs', args.output_path)
    run_inference(model, val_loader, eval_path, device)


if __name__ == '__main__':
    main()