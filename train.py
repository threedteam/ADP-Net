import argparse
import gc
import multiprocessing
import random
from typing import Tuple, Dict, Any


import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger

from ADP.data.parse_config import ConfigParser
from ADP.networks.discriminators import NLayerDiscriminator
from ADP.networks.generators import ADPNet
from ADP.trainer import PLTrainer
from MAE.util.misc import get_mae_model

def setup_environment(seed: int):

    multiprocessing.freeze_support()

    gc.collect()
    torch.cuda.empty_cache()

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Global random seed set to {seed}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='ADP Training Script')
    parser.add_argument('-c', '--config', required=True, type=str, help='Path to the configuration file.')
    parser.add_argument('-e', '--exp_name', required=True, type=str, help='Name of the experiment.')
    parser.add_argument('-r', '--resume', default=None, type=str, help='Path to the checkpoint to resume training from.')
    parser.add_argument('--use_ema', action='store_true', help='Enable Exponential Moving Average for model weights.')
    parser.add_argument('--resume_mae', default=None, type=str, help='Path to a pretrained MAE checkpoint to load.')
    return parser.parse_args()


def setup_logging_and_callbacks(exp_name: str) -> Tuple[TestTubeLogger, ModelCheckpoint]:

    save_dir = "ckpts"
    model_save_path = f'{save_dir}/{exp_name}/models'

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_path,
        monitor='val/F1',
        mode='max',
        save_last=True
    )

    logger = TestTubeLogger(
        save_dir=save_dir,
        name=exp_name,
        version=0,
        debug=False,
        create_git_tag=False
    )
    return logger, checkpoint_callback


def run_training(model: PLTrainer, config: Dict[str, Any], args: argparse.Namespace, logger: TestTubeLogger,
                 checkpoint_callback: ModelCheckpoint):

    trainer = Trainer(
        max_steps=config['trainer']['total_step'],
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=args.resume,
        logger=logger,
        log_every_n_steps=config['trainer']['logging_every'],
        val_check_interval=config['trainer']['eval_period'],
        gpus=-1,
        accelerator='ddp',
        precision=32,
        benchmark=True,
        num_sanity_val_steps=-1,
        terminate_on_nan=False,
    )
    trainer.fit(model)


def main():

    args = parse_arguments()
    config = ConfigParser.from_args(args, mkdir=True)


    seed = config.get('seed', 3407)
    setup_environment(seed)


    logger, checkpoint_callback = setup_logging_and_callbacks(args.exp_name)


    mae_model = get_mae_model('mae_vit_base_patch16') if args.resume_mae else None
    adp_model = ADPNet(config['g_args'])
    discriminator_model = NLayerDiscriminator(config['d_args'])


    num_gpus = torch.cuda.device_count()
    pl_model = PLTrainer(
        mae_model,
        adp_model,
        discriminator_model,
        config,
        f'ckpts/{args.exp_name}',
        num_gpus,
        args.use_ema
    )


    if args.resume_mae and not args.resume:
        print(f"Loading MAE-only checkpoint for new training: {args.resume_mae}")
        checkpoint = torch.load(args.resume_mae, map_location='cpu')
        pl_model.mae.load_state_dict(checkpoint['model'], strict=False)

    if args.use_ema:
        pl_model.reset_ema()

    run_training(pl_model, config, args, logger, checkpoint_callback)

if __name__ == '__main__':
    main()