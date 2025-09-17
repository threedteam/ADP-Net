from ADP.networks.adu import ADU
from ADP.networks.lamadownsamper import *
import torch.nn
from functools import partial

class ADPGenerator(nn.Module):
    def __init__(self, config,act ='relu',Norm = 'BN'):
        super().__init__()
        if act == 'relu':
            self.act = nn.ReLU(True)
            self.act_mid = nn.ReLU
            self.act_up = nn.ReLU(True)
        elif act == 'swish':
            self.act = nn.SiLU(True)
            self.act_mid = nn.SiLU
            self.act_up = nn.SiLU(True)
        elif act == 'leakyrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            self.act_mid = partial(nn.LeakyReLU , negative_slope=0.2, inplace=True)
            self.act_up = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if Norm == 'IN':
            NormLayer = partial(nn.InstanceNorm2d, affine=True, track_running_stats=False)
        elif Norm == 'BN':
            NormLayer = partial(nn.BatchNorm2d)

        self.config = config
        self.downsampler = LamaDownsampler()

        blocks = []
        for i in range(9):
            blocks.append(
                FFCResnetBlock(
                    dim=512,
                    padding_type='reflect',
                    norm_layer=NormLayer,
                    activation_layer=self.act_mid,
                    dilation=1,
                    inline=False,
                    ratio_gin=0.75,
                    ratio_gout=0.75,
                    enable_lfu=False,
                    gated=False,
                    spatial_scale_factor=None,
                ))
        blocks.append(ConcatTupleLayer())
        self.middle = nn.Sequential(*blocks)

        self.up_block1 = ADU(
            in_channels=512,
            out_channels=256,
            kernel_size=3,
            stride=2,
            padding=1,
            transpose=True,
            output_padding=1,
            activation=self.act_up,
            NormLayer = NormLayer,
            first = True
        )

        self.up_block2 = ADU(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1,
            transpose=True,
            output_padding=1,
            activation=self.act_up,
            NormLayer = NormLayer,
            first = True
        )

        self.up_block3 = ADU(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            transpose=True,
            output_padding=1,
            activation=self.act_up,
            NormLayer = NormLayer,
            first = True
        )

        self.padt = nn.ReflectionPad2d(3)
        self.convt4 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0)
        self.act_last = nn.Sigmoid()

    def forward(self, x, mask, instance_mask):

        x = self.downsampler(x)

        x = self.middle(x)

        target_h, target_w = x.shape[2:]
        if instance_mask.shape[2] != target_h or mask.shape[3] != target_w:
            mask_resized_pre = F.interpolate(instance_mask, size=(target_h, target_w), mode='nearest')
        else:
            mask_resized_pre = instance_mask
        if instance_mask.shape[2] != target_h*2 or mask.shape[3] != target_w*2:
            mask_resized_after = F.interpolate(instance_mask, size=(target_h*2, target_w*2), mode='nearest')
        else:
            mask_resized_after = instance_mask
        x = self.up_block1(x, mask_resized_pre,mask_resized_after)


        if instance_mask.shape[2] != target_h * 4 or mask.shape[3] != target_w * 4:
            mask_resized_after2 = F.interpolate(instance_mask, size=(target_h * 4, target_w * 4), mode='nearest')
        else:
            mask_resized_after2 = instance_mask
        x = self.up_block2(x, mask_resized_after,mask_resized_after2)


        if instance_mask.shape[2] != target_h * 8 or mask.shape[3] != target_w * 8:
            mask_resized_after3= F.interpolate(instance_mask, size=(target_h * 8, target_w * 8), mode='nearest')
        else:
            mask_resized_after3 = instance_mask
        x = self.up_block3(x, mask_resized_after2, mask_resized_after3)

        x = self.act(x)
        x = self.padt(x)
        x = self.convt4(x)
        x = self.act_last(x)

        return x


class ADPNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.Generator = ADPGenerator(config)

    def forward(self, batch):
        img = batch['image']
        mask = batch['mask']
        instance_mask = batch['instance_mask']

        kernel_size = self.config['kernel_size']
        padding = (kernel_size - 1) // 2
        dilated_mask = F.max_pool2d(
            instance_mask,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        masked_img = img * (1 - mask)
        masked_img = torch.cat([masked_img, mask], dim=1)

        gen_img = self.Generator(masked_img, mask,dilated_mask)

        return gen_img
