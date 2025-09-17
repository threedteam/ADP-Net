from ADP.networks.ffc import *

class LamaDownsampler(nn.Module):
    def __init__(self, input_nc=4, ngf=64, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            FFC_BN_ACT(
                in_channels=input_nc,
                out_channels=ngf,
                kernel_size=7,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                ratio_gin=0,
                ratio_gout=0,
                enable_lfu=False
            )
        )


        self.downsampler_1 = FFC_BN_ACT(
            in_channels=ngf,  # 64
            out_channels=ngf * 2,  # 128
            kernel_size=3,
            stride=2,
            padding=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            ratio_gin=0,
            ratio_gout=0,
            enable_lfu=False
        )


        self.downsampler_2 = FFC_BN_ACT(
            in_channels=ngf * 2,  # 128
            out_channels=ngf * 4,  # 256
            kernel_size=3,
            stride=2,
            padding=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            ratio_gin=0,
            ratio_gout=0,
            enable_lfu=False
        )


        self.downsampler_3_splitter = FFC_BN_ACT(
            in_channels=ngf * 4,  # 256
            out_channels=ngf * 8,  # 512
            kernel_size=3,
            stride=2,
            padding=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            ratio_gin=0,
            ratio_gout=0.75,
            enable_lfu=False
        )

    def forward(self, x):

        x = self.initial_conv(x)

        x = self.downsampler_1(x)

        x = self.downsampler_2(x)

        x = self.downsampler_3_splitter(x)

        return x