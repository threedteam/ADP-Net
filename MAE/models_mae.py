# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import numpy as np
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed

from MAE.util.pos_embed import get_2d_sincos_pos_embed
from MAE.vision_transformer import Block


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    Since [cls] is useless in inpainting, we remove it.
    """

    def __init__(self, img_size=256, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, decoder_embed_dim=512,
                 decoder_depth=8, decoder_num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm,
                 norm_pix_loss=False, init=True, random_mask=False, mask_decoder=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                      requires_grad=False)


        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)


        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))


        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim),
                                              requires_grad=False)


        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, mask_decoder=mask_decoder)
            for i in range(decoder_depth)])


        self.decoder_norm = norm_layer(decoder_embed_dim)


        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # encoder to decoder

        self.norm_pix_loss = norm_pix_loss
        self.random_mask = random_mask
        self.mask_decoder = mask_decoder

        if init:
            self.initialize_weights()

    def initialize_weights(self):

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))


        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def adaptive_random_masking(self, x, mask, mask_ratio):

        N, L, D = x.shape  # batch, length, dim
        s = int(np.sqrt(L))
        mask = F.interpolate(mask, size=[s, s], mode='area')
        mask[mask > 0] = 1  # [N,1,S,S]
        mask = mask.reshape(N, L)  # [N,L]

        len_keep = int(L * (1 - mask_ratio))


        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        noise = torch.clamp(noise + mask, 0.0, 1.0)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]


        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask, mask_ratio):

        # embed patches
        x = self.patch_embed(x)

        x = x + self.pos_embed

        # masking: length -> length * mask_ratio
        if self.random_mask:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            x, mask, ids_restore = self.adaptive_random_masking(x, mask, mask_ratio)

        # apply Transformer blocks
        for blk in self.blocks:
            x, _ = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):

        # embed tokens
        x = self.decoder_embed(x)


        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)

        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token

        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        x = x_

        # add pos embed
        x = x + self.decoder_pos_embed


        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x, _ = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward_encoder_with_mask(self, x, mask):

        # embed patches
        x = self.patch_embed(x) #
        x = x + self.pos_embed
        N, L, D = x.shape  # batch, length, dim
        s = int(np.sqrt(L))
        # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking(x, mask_ratio)
        mask = F.interpolate(mask, size=[s, s], mode='area')
        mask_small = mask.clone()
        mask[mask > 0] = 1  # [N,1,S,S]
        mask_small[mask_small < 1] = 0
        mask = mask.reshape(N, L).unsqueeze(1).unsqueeze(1)  # [N,1,1,L]

        # apply Transformer blocks
        for blk in self.blocks:
            x, _ = blk(x, mask)
        x = self.norm(x)  # N,L,D

        mask = mask.squeeze(1).squeeze(1)  # N, L
        mask_small = mask_small.reshape(N, L).unsqueeze(1).unsqueeze(1) # [N,1,1,L]

        return x, mask, mask_small

    def forward_encoder_all_features(self, x):
        x_embed = self.patch_embed(x)
        if hasattr(self, 'cls_token') and self.cls_token is not None:
            x_processed = x_embed + self.pos_embed[:, 1:, :] if self.pos_embed.shape[1] == (
                        x_embed.shape[1] + 1) else x_embed + self.pos_embed
        else:
            x_processed = x_embed + self.pos_embed

        all_features_list = []
        current_x = x_processed
        i = 1
        for blk in self.blocks:
            if i == 9:
                break
            block_output = blk(current_x)
            if isinstance(block_output, tuple):
                current_x = block_output[0]
            else:
                current_x = block_output
            if i < 5:
              pass
            else:
                all_features_list.append(current_x.clone())
            i += 1
        final_normalized_output = self.norm(current_x)

        return final_normalized_output, all_features_list

    def forward(self, imgs, mask, mask_ratio=0.75):
        """
        return loss, pred img, mask. Used during training.
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask



