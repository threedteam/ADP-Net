import torch
import torch.nn as nn
import torch.nn.functional as F
from ADP.networks.van import VANBlock
from ADP.networks.spade_arch import SPADEResnetBlock


class MySimpleOptions:
    def __init__(self, norm_G, semantic_nc):
        self.norm_G = norm_G
        self.semantic_nc = semantic_nc


class ADU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, transpose=False,
                 output_padding=0,
                 groups=1, bias=True, padding_mode='zeros',
                 activation=F.elu,NormLayer = nn.BatchNorm2d,first = False):
        super(ADU, self).__init__()

        self.first = first
        if first:
            norm_G_setting = "spectralspadebatch3x3"
            semantic_nc_setting = 2
            custom_opt = MySimpleOptions(norm_G=norm_G_setting, semantic_nc=semantic_nc_setting)
            self.SPADEResnetBlock = SPADEResnetBlock(in_channels, in_channels, custom_opt)

        self.out_channels = out_channels
        self.activation = activation
        self.fgbnt1 = NormLayer(out_channels)
        self.bgbnt1 = NormLayer(out_channels)
        self.bgbnt2 = NormLayer(out_channels)

        conv_layer = nn.ConvTranspose2d if transpose else nn.Conv2d
        conv_kwargs = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'groups': groups,
            'bias': bias
        }
        if transpose:
            conv_kwargs['output_padding'] = output_padding
        else:
            conv_kwargs['padding_mode'] = padding_mode

        self.conv_fg = conv_layer(in_channels, out_channels, **conv_kwargs)
        self.fg11 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)

        self.conv_bg = conv_layer(in_channels, out_channels, **conv_kwargs)
        self.bg11 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bg12 =  VANBlock(dim=out_channels, k_size=23, mlp_ratio=4.0,NormLayer = NormLayer)
        self.bgout = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, stride=1)

        self.gated = nn.Conv2d(out_channels, 2 * out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, mask,mask_resized_after):
        if self.first:
            one_hot_map = F.one_hot(mask.squeeze(1).long(), num_classes=2).permute(0, 3, 1, 2).float()
            x = self.SPADEResnetBlock(x.to(torch.float32),one_hot_map)
            x = self.activation(x)

        output_fg = self.conv_fg(x)
        output_fg = self.fgbnt1(output_fg)
        output_fg = self.activation(output_fg)
        output_fg = self.fg11(output_fg)
        output_fg= self.activation(output_fg)


        output_bg = self.conv_bg(x)
        output_bg1 = self.bgbnt1(output_bg)
        output_bg1 = self.activation(output_bg1)

        output_bg2 = self.bgbnt2(output_bg)
        output_bg2 = self.activation(output_bg2)

        bg11 = self.bg11(output_bg1)
        bg12 = self.bg12(output_bg2)

        output_bg = bg11 + bg12
        output_bg = self.activation(output_bg)
        output_bg = self.bgout(output_bg)
        output_bg = self.activation(output_bg)

        combined_output = mask_resized_after * output_fg + (1-mask_resized_after) * output_bg
        combined_output = self.gated(combined_output)
        features, gate_values = torch.split(combined_output, self.out_channels, dim=1)
        final_output = features * torch.sigmoid(gate_values)

        return final_output

