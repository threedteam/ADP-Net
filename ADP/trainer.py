import copy
import os
import cv2
import pytorch_lightning as ptl
import torch.distributed as dist
from torch.utils.data import DataLoader
from ADP.data.dataset import ADPDataset
from ADP.inpainting_metric import get_inpainting_metrics
from ADP.networks.losses import *
from ADP.networks.pcp import ResNetPL
from ADP.utils import stitch_images, get_lr_milestone_decay_with_warmup

class PLTrainer(ptl.LightningModule):
    def __init__(self, mae, adp, D, config, out_path, num_gpus=1, use_ema=False):
        super().__init__()
        self.mae = mae
        if self.mae:
            self.mae.requires_grad_(False).eval()
        self.adp = adp
        self.ema = None
        self.D = D
        self.pin_memory = config['pin_memory']
        self.num_gpus = num_gpus
        self.args = config
        self.use_ema = use_ema
        self.config = config['trainer']
        self.data_args = config['dataset']
        self.D_reg_interval = self.config.get('D_reg_interval', 1)
        self.sample_period = config['trainer']['sample_period']
        self.sample_path = os.path.join(out_path, 'samples')
        self.eval_path = os.path.join(out_path, 'validation')
        self.model_path = os.path.join(out_path, 'models')
        os.makedirs(self.sample_path, exist_ok=True)
        os.makedirs(self.eval_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        self.test_path = config['test_path']

        self.adv_args = self.config['adversarial']
        if self.config.get("resnet_pl", {"weight": 0})['weight'] > 0:
            self.loss_resnet_pl = ResNetPL(**self.config['resnet_pl']).to(self.device)
        else:
            self.loss_resnet_pl = None
        self.g_opt_state = None
        self.d_opt_state = None
        self.gen_img_for_train = None
        self.img_size = 256

    def reset_ema(self):
        self.ema = copy.deepcopy(self.adp)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        if hasattr(self, 'g_lr'):
            items['g_lr'] = self.g_lr
        if hasattr(self, 'd_lr'):
            items['d_lr'] = self.d_lr
        items['size'] = self.img_size
        return items

    def run_G(self, items):
        out = self.adp.forward(items)
        combined_out = items['image'] * (1 - items['mask']) + out * items['mask']
        return combined_out, out

    def run_G_ema(self, items):
        out = self.ema.forward(items)
        combined_out = items['image'] * (1 - items['mask']) + out * items['mask']
        return combined_out, out

    def postprocess(self, img, range=[0, 1]):
        img = (img - range[0]) / (range[1] - range[0])
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def _preprocess_for_mae(self, img_tensor):
        self.mae_mean = torch.tensor( [0.485, 0.456, 0.406], device=self.device).view(
            1, -1, 1, 1)
        self.mae_std = torch.tensor( [0.229, 0.224, 0.225], device=self.device).view(1,-1,1,1)
        img_resized = F.interpolate(img_tensor,
                                    size=(256, 256),
                                    mode='bilinear',
                                    align_corners=False)
        img_normalized = (img_resized - self.mae_mean.to(img_resized.device)) / self.mae_std.to(
            img_resized.device)
        return img_normalized

    def train_G(self, items):
        loss_dict = dict()
        loss_G_all = torch.tensor(0.0, device=items['image'].device)
        _, gen_img = self.run_G(items)
        self.gen_img_for_train = gen_img

        gen_logits, gen_feats = self.D.forward(gen_img)

        adv_gen_loss = generator_loss(discr_fake_pred=gen_logits, mask=items['mask'], args=self.adv_args)
        loss_G_all += adv_gen_loss
        loss_dict['g_fake'] = adv_gen_loss

        kernel_size = self.config['kernel_size']
        padding = (kernel_size - 1) // 2
        dilated_mask = F.max_pool2d(
            items['instance_mask'],
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        difference_mask = torch.clamp(dilated_mask - items['instance_mask'], 0, 1)

        if self.loss_resnet_pl is not None:
            pcp_loss = self.loss_resnet_pl(gen_img, items['image'])
            loss_G_all += pcp_loss
            loss_dict['pcp'] = pcp_loss

        if self.config['feature_matching']['weight'] > 0:
            _, real_feats = self.D(items['image'])
            fm_loss = feature_matching_loss(gen_feats, real_feats, mask=None) * self.config['feature_matching']['weight']
            loss_G_all += fm_loss
            loss_dict['fm'] = fm_loss

        if self.config['l1']['use_l1']:
            per_pixel_l1 = F.l1_loss(gen_img, items['image'], reduction='none')
            l1_mask = items['mask'] * self.config['l1']['weight_missing'] + (1 - items['mask']) * self.config['l1']['weight_known']
            l1_loss = (per_pixel_l1 * l1_mask).mean()
            loss_G_all += l1_loss
            loss_dict['l1'] = l1_loss

        if self.mae is not None and self.config['mae_encoder_perceptual']['weight'] > 0:
            x_resized = F.interpolate(items['image'], size=(256, 256), mode="bilinear", align_corners=False)
            real_img_for_mae = self._preprocess_for_mae(x_resized)
            x_resized = F.interpolate(gen_img, size=(256, 256), mode="bilinear", align_corners=False)
            comp_img_for_mae = self._preprocess_for_mae(x_resized)
            loss_mae_enc_p = boundary_mae_perceptual_loss(
                self.mae,
                comp_img_for_mae,
                real_img_for_mae,
                difference_mask,
                focused_region_weight_multiplier=self.config['mae_encoder_perceptual']['region_weight'],
                layer_weights=None,
                device=self.device
            )
            loss_mae =  self.config['mae_encoder_perceptual']['weight'] * loss_mae_enc_p
            loss_G_all += loss_mae
            loss_dict['mae_perceptual'] = loss_mae

        return loss_G_all, loss_dict

    def train_D(self, items, do_GP=True):
        real_img_tmp = items['image'].requires_grad_(do_GP)
        real_logits, _ = self.D.forward(real_img_tmp)

        gen_img = self.gen_img_for_train
        gen_logits, _ = self.D.forward(gen_img.detach())

        dis_real_loss, grad_penalty = discriminator_real_loss(real_batch=real_img_tmp, discr_real_pred=real_logits,
                                                              gp_coef=self.adv_args['gp_coef'], do_GP=do_GP)

        dis_fake_loss = discriminator_fake_loss(discr_fake_pred=gen_logits, mask=items['mask'], args=self.adv_args)
        loss_dict = {'d_real': dis_real_loss, 'd_fake': dis_fake_loss}

        if do_GP:
            grad_penalty = grad_penalty * self.D_reg_interval
            loss_dict['gp'] = grad_penalty
        loss_D_all = dis_real_loss + dis_fake_loss + grad_penalty

        return loss_D_all, loss_dict

    def train_dataloader(self):

        self.train_dataset = ADPDataset(config=self.data_args, flist=self.args['train_flist'], mask_path=self.args['train_mask_flist'],
                                        batch_size=self.args['batch_size'] // self.num_gpus, augment=True,
                                        training=True, test_mask_path=None, world_size=self.num_gpus, trainer = self.args['trainer'])

        return DataLoader(self.train_dataset, pin_memory=self.pin_memory, batch_size=self.args['batch_size'] // self.num_gpus, shuffle=False, num_workers=8)


    def val_dataloader(self):
        self.val_dataset = ADPDataset(config=self.data_args,flist=self.args['val_flist'], mask_path=None,
                                      batch_size=1, augment=False,
                                      training=False,
                                      test_mask_path=self.args['test_mask_flist'], trainer=self.args['trainer'])
        return DataLoader(self.val_dataset, pin_memory=self.pin_memory,
                          batch_size=1, shuffle=False, num_workers=4)

    def configure_optimizers(self):
        opt_args = self.args['optimizer']
        g_optimizer = torch.optim.Adam(self.adp.parameters(), lr=opt_args['g_opt']['lr'],
                                       betas=(opt_args['g_opt']['beta1'], opt_args['g_opt']['beta2']), eps=1e-8)
        d_optimizer = torch.optim.Adam(self.D.parameters(), lr=opt_args['d_opt']['lr'],
                                       betas=(opt_args['d_opt']['beta1'], opt_args['d_opt']['beta2']), eps=1e-8)
        if self.g_opt_state is not None:
            g_optimizer.load_state_dict(self.g_opt_state)

        if self.d_opt_state is not None:
            d_optimizer.load_state_dict(self.d_opt_state)

        g_sche = get_lr_milestone_decay_with_warmup(g_optimizer, num_warmup_steps=opt_args['warmup_steps'],
                                                    milestone_steps=opt_args['decay_steps'],
                                                    gamma=opt_args['decay_rate'])
        g_sche = {'scheduler': g_sche, 'interval': 'step'}

        d_sche = get_lr_milestone_decay_with_warmup(d_optimizer, num_warmup_steps=opt_args['warmup_steps'],
                                                    milestone_steps=opt_args['decay_steps'],
                                                    gamma=opt_args['decay_rate'])
        d_sche = {'scheduler': d_sche, 'interval': 'step'}

        return [g_optimizer, d_optimizer], [g_sche, d_sche]

    def on_train_start(self) -> None:
        if self.get_ddp_rank() is not None and (self.g_opt_state is not None or self.d_opt_state is not None):
            for opt in self.trainer.optimizers:
                if 'state' in opt.state_dict():
                    for k in opt.state_dict()['state']:
                        for k_ in opt.state_dict()['state'][k]:
                            if isinstance(opt.state_dict()['state'][k][k_], torch.Tensor):
                                opt.state_dict()['state'][k][k_] = opt.state_dict()['state'][k][k_].to(device=self.get_ddp_rank())


    def training_step(self, batch, batch_idx, optimizer_idx=None):
        if self.global_step % self.sample_period == 0 and optimizer_idx == 0:
            self._sample(batch)

        self.img_size = batch['image'].shape[-1]
        if optimizer_idx == 0:
            self.adp.requires_grad_(True)
            self.D.requires_grad_(False)
            loss, loss_dict = self.train_G(batch)

        elif optimizer_idx == 1:
            self.adp.requires_grad_(False)
            self.D.requires_grad_(True)
            loss, loss_dict = self.train_D(batch)
        else:
            raise NotImplementedError

        log = {'train/'+ k: v for k, v in loss_dict.items()}
        return dict(loss=loss, log_info=log)

    def training_step_end(self, batch_parts_outputs):
        if self.use_ema:
            with torch.no_grad():
                for p_ema, p in zip(self.ema.parameters(), self.adp.parameters()):
                    p_ema.copy_(p.lerp(p_ema, self.config['ema_beta']))
                for b_ema, b in zip(self.ema.buffers(), self.adp.buffers()):
                    b_ema.copy_(b)

        full_loss = (batch_parts_outputs['loss'].mean()
                     if torch.is_tensor(batch_parts_outputs['loss'])
                     else torch.tensor(batch_parts_outputs['loss']).float().requires_grad_(True))
        log_info = {k: v.mean() for k, v in batch_parts_outputs['log_info'].items()}
        self.log_dict(log_info, on_step=True, on_epoch=False)

        sche = self.trainer.lr_schedulers
        self.g_lr = sche[0]['scheduler'].get_lr()[0]
        self.d_lr = sche[1]['scheduler'].get_lr()[0]

        return full_loss

    def validation_step(self, batch, batch_idx):
        if self.use_ema:
            self.ema.eval()
        self.adp.eval()
        if self.use_ema:
            gen_ema_img, _ = self.run_G_ema(batch)
        else:
            gen_ema_img, _ = self.run_G(batch)
        gen_ema_img = torch.clamp(gen_ema_img, 0, 1)
        gen_ema_img = gen_ema_img * 255.0
        gen_ema_img = gen_ema_img.permute(0, 2, 3, 1).int().cpu().numpy()
        for img_num in range(batch['image'].shape[0]):
            cv2.imwrite(self.eval_path + '/' + batch['name'][img_num], gen_ema_img[img_num, :, :, ::-1])
        self.adp.train()

    def validation_epoch_end(self, outputs):

        if self.trainer.world_size > 1:
            dist.barrier()
        if self.trainer.is_global_zero:
            self.metric = get_inpainting_metrics(self.eval_path, self.test_path)

            self.print(f'Steps:{self.global_step}')
            for m in self.metric:
                if m in ['PSNR', 'SSIM', 'FID', 'LPIPS']:
                    if m in self.metric:
                        self.print(m, self.metric[m])
                self.metric[m] *= self.num_gpus

        else:
            self.metric = {'PSNR': 0, 'SSIM': 0, 'LPIPS': 0, 'FID': 0}

        fid_rel = max(0, 100 - self.metric['FID']) / 100
        f1 = 2 * self.metric['SSIM'] * fid_rel / (self.metric['SSIM'] + fid_rel + 1e-3)

        self.log('val/PSNR', self.metric['PSNR'], sync_dist=True)
        self.log('val/SSIM', self.metric['SSIM'], sync_dist=True)
        self.log('val/LPIPS', self.metric['LPIPS'], sync_dist=True)
        self.log('val/FID', self.metric['FID'], sync_dist=True)
        self.log('val/F1', f1, sync_dist=True)

    def _sample(self, items):
        self.adp.eval()
        if self.use_ema:
            self.ema.eval()

            combined_gen_img, _ = self.run_G(items)
            combined_gen_img = torch.clamp(combined_gen_img, 0, 1)
            if self.use_ema:
                combined_gen_ema_img, _ = self.run_G_ema(items)
                combined_gen_ema_img = torch.clamp(combined_gen_ema_img, 0, 1)
            else:
                combined_gen_ema_img = torch.zeros_like(combined_gen_img)
            if self.args['batch_size'] // self.num_gpus > 1:
                image_per_row = 2
            else:
                image_per_row = 1
            kernel_size = self.config['kernel_size']
            padding = (kernel_size - 1) // 2
            dilated_mask = F.max_pool2d(
                items['instance_mask'],
                kernel_size=kernel_size,
                stride=1,
                padding=padding
            )
            images = stitch_images(
                self.postprocess((items['image']).cpu()),
                self.postprocess((items['image'] * (1 - items['mask'])).cpu()),
                self.postprocess((items['instance_mask']).cpu()),
                self.postprocess((dilated_mask).cpu()),
                self.postprocess(combined_gen_img.cpu()),
                self.postprocess(combined_gen_ema_img.cpu()),
                img_per_row=image_per_row
            )
            if self.get_ddp_rank() in (None, 0):
                name = os.path.join(self.sample_path, str(self.global_step).zfill(6) + ".jpg")
                self.print('saving sample ' + name)
                images.save(name)
        self.adp.train()

    def get_ddp_rank(self):
        return self.trainer.global_rank if (self.trainer.num_nodes * self.trainer.num_processes) > 1 else None


