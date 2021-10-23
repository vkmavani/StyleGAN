import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_modules import WSLinear, WSConv2d, AdaIN, NoiseInjection, PixelNormLayer, FusedUpSample, BlurLayer


clclass G_Mapping(nn.Module):
    """Generator Mapping Network. Used to map noise 'z' --> 'w'.

    Parameters
    ----------
    z_dim : int
        The dimension of the noise vector
    w_dim : int
        The dimension of the mapped vector
    """
    def __init__(self, z_dim, w_dim):
        super(G_Mapping, self).__init__()
        layers = [PixelNormLayer()]
        for _ in range(7):
            layers.append(WSLinear(z_dim, w_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
            z_dim = w_dim
        layers.append(WSLinear(z_dim, w_dim))
        self.mapping = nn.Sequential(*layers)

    def forward(self, noise):
        return self.mapping(noise)


class GenBlock(nn.Module):
    """Final StyleGAN generator component, which will apply the Progressive Growing of Generator.

    Parameters
    ----------
    in_chan : int
        Number of input channels
    out_chan : int
        Number of output channels
    w_dim : int
        Dimension of the intermediate noise vector
    fused : bool
        If fused is true then torch.nn.functional.conv_transpose2d else torch.nn.Upsample
    """
    def __init__(
        self, in_chan, out_chan, kernel_size=3, w_dim=512, fused=False
    ):
        super(GenBlock, self).__init__()
        if fused:
            self.conv1 = nn.Sequential(
                FusedUpSample(in_chan, out_chan, kernel_size),
                BlurLayer(),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                WSConv2d(in_chan, out_chan),
                BlurLayer(),
            )
        self.inject_noise1 = NoiseInjection()
        self.adain1 = AdaIN(out_chan, w_dim)
        self.conv2 = WSConv2d(out_chan, out_chan)
        self.inject_noise2 = NoiseInjection()
        self.adain2 = AdaIN(out_chan, w_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, w):
        x = self.activation(self.inject_noise1(self.conv1(x)))
        x = self.adain1(x, w[:, 0])
        x = self.activation(self.inject_noise2(self.conv2(x)))
        x = self.adain2(x, w[:, 1])
        return x


class Generator(nn.Module):
    def __init__(
        self,
        z_dim,
        w_dim,
        in_channels,
        img_channels=3,
        resolution=1024,
        fmap_base=8192,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_max=512,    # Maximum number of feature maps in any layer.
    ):
        super(Generator, self).__init__()

        # parameters ---
        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        resolution_log2 = int(math.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4

        self.initial_constant = nn.Parameter(torch.ones((1, in_channels, 4, 4)))
        self.mapping = G_Mapping(z_dim, w_dim)

        # Initial Block 4x4
        self.initial_noise1 = NoiseInjection()
        self.initial_adain1 = AdaIN(in_channels, w_dim)
        self.initial_conv = WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.initial_noise2 = NoiseInjection()
        self.initial_adain2 = AdaIN(in_channels, w_dim)
        self.activation = nn.LeakyReLU(0.2)
        self.initial_rgb = WSConv2d(in_channels, img_channels, kernel_size=1, stride=1, padding=0)

        self.progress_blocks, self.to_rgb = nn.ModuleList([]), nn.ModuleList([self.initial_rgb])

        for res in range(2, resolution_log2):
            conv_in_chan  = nf(res - 1)
            conv_out_chan = nf(res)
            fused = True if conv_in_chan <= 256 else False
            self.progress_blocks.append(GenBlock(conv_in_chan, conv_out_chan, kernel_size=3, w_dim=w_dim, fused=fused))
            self.to_rgb.append(WSConv2d(conv_out_chan, img_channels, kernel_size=1, stride=1, padding=0))

    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1-alpha) * upscaled)

    def forward(self, noise, alpha, steps):
        w = self.mapping(noise)

        x = self.initial_adain1(self.initial_noise1(self.initial_constant), w[:, 0])
        x = self.initial_conv(x)
        out = self.initial_adain2(self.initial_noise2(x), w[:, 1])

        if steps == 0:
            return self.initial_rgb(x)

        for step in range(steps - 1):
            out = self.progress_blocks[step](out, w)

        final_upScaled = self.to_rgb[steps - 1](F.interpolate(out, scale_factor=2, mode="bilinear"))
        final_out = self.to_rgb[steps](self.progress_blocks[steps - 1](out, w))
        return self.fade_in(alpha, final_upScaled, final_out)


if __name__ == "__main__":
    z_dim = 512
    w_dim = 512
    in_channels = 512
    gen = Generator(z_dim, w_dim, in_channels, img_channels=3)

    tot = 0
    for param in gen.parameters():
        tot += param.numel()
    print("Total Parameters: {}".format(tot))

    for img_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        num_steps = int(math.log2(img_size / 4))
        x = torch.randn((2, 2, z_dim))
        z = gen(x, alpha=0.5, steps=num_steps)
        assert z.shape == (2, 3, img_size, img_size)
        print("\nSuccess! At Image size: ", img_size)
