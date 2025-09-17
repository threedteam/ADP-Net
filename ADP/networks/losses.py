
import torch
import torch.nn.functional as F

def boundary_mae_perceptual_loss(
        mae_model,
        gen_img,
        real_img,
        region_mask=None,
        patch_size=16,
        focused_region_weight_multiplier=2.0,
        loss_type='l1',
        layer_weights=None,
        device='cpu'
):

    gen_img = gen_img.float().to(device)
    real_img = real_img.float().to(device)
    if region_mask is not None:
        region_mask = region_mask.to(device)

    _, gen_features_list = mae_model.forward_encoder_all_features(gen_img)
    with torch.no_grad():
        _, real_features_list = mae_model.forward_encoder_all_features(real_img)

    total_averaged_layer_loss = torch.tensor(0.0, device=device)
    num_layers = len(gen_features_list)

    if num_layers == 0:
        return total_averaged_layer_loss

    effective_encoder_layer_weights = layer_weights if layer_weights is not None else [1.0] * num_layers
    if len(effective_encoder_layer_weights) != num_layers:
        raise ValueError(
            f"layer_weights len ({len(effective_encoder_layer_weights)}) !!! ({num_layers})"
        )

    N, C, H, W = gen_img.shape
    patch_level_spatial_weights = None
    if region_mask is not None:
        if region_mask.ndim == 3:
            region_mask = region_mask.unsqueeze(1)

        if region_mask.shape[2] != H or region_mask.shape[3] != W:
            region_mask = F.interpolate(region_mask.float(), size=(H, W), mode='nearest').byte()

        patch_level_avg = F.avg_pool2d(region_mask.float(), kernel_size=patch_size, stride=patch_size)
        patch_level_region_indicator_flat = (patch_level_avg.view(N, -1) >= 0.5).float()


        patch_level_spatial_weights = torch.ones_like(patch_level_region_indicator_flat)
        patch_level_spatial_weights += (focused_region_weight_multiplier - 1.0) * patch_level_region_indicator_flat

        patch_level_spatial_weights = patch_level_spatial_weights.unsqueeze(-1)

    for i in range(num_layers):
        gen_feat = gen_features_list[i]
        real_feat = real_features_list[i]

        if loss_type == 'l1':
            pointwise_loss = torch.abs(gen_feat - real_feat)
        elif loss_type == 'l2':
            pointwise_loss = torch.pow(gen_feat - real_feat, 2)
        else:
            raise ValueError(f" loss_type: {loss_type} error")

        if patch_level_spatial_weights is not None:
            weighted_pointwise_loss = pointwise_loss * patch_level_spatial_weights
            layer_loss_val = weighted_pointwise_loss.mean()
        else:
            layer_loss_val = pointwise_loss.mean()

        encoder_layer_weight = effective_encoder_layer_weights[i]
        total_averaged_layer_loss += encoder_layer_weight * layer_loss_val

    if num_layers > 0:
        final_loss = total_averaged_layer_loss / num_layers
    else:
        final_loss = total_averaged_layer_loss

    return final_loss


def interpolate_mask(mask, shape, allow_scale_mask=False, mask_scale_mode='nearest'):
    assert mask is not None
    assert allow_scale_mask or shape == mask.shape[-2:]
    if shape != mask.shape[-2:] and allow_scale_mask:
        if mask_scale_mode == 'maxpool':
            mask = F.adaptive_max_pool2d(mask, shape)
        else:
            mask = F.interpolate(mask, size=shape, mode=mask_scale_mode)
    return mask


def generator_loss(discr_fake_pred: torch.Tensor, mask=None, args=None):

    fake_loss = F.softplus(-discr_fake_pred)
    if (args['mask_as_fake_target'] and args['extra_mask_weight_for_gen'] > 0) or not args['use_unmasked_for_gen']:
        mask = interpolate_mask(mask, discr_fake_pred.shape[-2:], args['allow_scale_mask'], args['mask_scale_mode'])
        if not args['use_unmasked_for_gen']:
            fake_loss = fake_loss * mask
        else:
            pixel_weights = 1 + mask * args['extra_mask_weight_for_gen']
            fake_loss = fake_loss * pixel_weights

    return fake_loss.mean() * args['weight']

def feature_matching_loss(fake_features, target_features, mask=None):
    if mask is None:
        res = torch.stack([F.mse_loss(fake_feat, target_feat)
                           for fake_feat, target_feat in zip(fake_features, target_features)]).mean()
    else:
        res = 0
        norm = 0
        for fake_feat, target_feat in zip(fake_features, target_features):
            cur_mask = F.interpolate(mask, size=fake_feat.shape[-2:], mode='bilinear', align_corners=False)
            error_weights = 1 - cur_mask
            cur_val = ((fake_feat - target_feat).pow(2) * error_weights).mean()
            res = res + cur_val
            norm += 1
        res = res / norm
    return res

def make_r1_gp(discr_real_pred, real_batch):
    if torch.is_grad_enabled():
        grad_real = torch.autograd.grad(outputs=discr_real_pred.sum(), inputs=real_batch, create_graph=True)[0]
        grad_penalty = (grad_real.view(grad_real.shape[0], -1).norm(2, dim=1) ** 2).mean()
    else:
        grad_penalty = 0
    real_batch.requires_grad = False

    return grad_penalty

def discriminator_real_loss(real_batch, discr_real_pred, gp_coef, do_GP=True):
    real_loss = F.softplus(-discr_real_pred).mean()
    if do_GP:
        grad_penalty = (make_r1_gp(discr_real_pred, real_batch) * gp_coef).mean()
    else:
        grad_penalty = 0

    return real_loss, grad_penalty


def discriminator_fake_loss(discr_fake_pred: torch.Tensor, mask=None, args=None):

    fake_loss = F.softplus(discr_fake_pred)

    if not args['use_unmasked_for_discr'] or args['mask_as_fake_target']:

        mask = interpolate_mask(mask, discr_fake_pred.shape[-2:], args['allow_scale_mask'], args['mask_scale_mode'])
        fake_loss = fake_loss * mask
        if args['mask_as_fake_target']:
            fake_loss = fake_loss + (1 - mask) * F.softplus(-discr_fake_pred)

    sum_discr_loss = fake_loss
    return sum_discr_loss.mean()
