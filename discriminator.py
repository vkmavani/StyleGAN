import math
import torch
import torch.nn as nn

from custom_modules import WSLinear, WSConv2d, FusedDownSample, BlurLayer, View


class DiscBlock(nn.Module):
    """Final StyleGAN Discriminator component, which will apply the Progressive Growing of Discriminator.

    Parameters
    ----------
    in_chan : int
        Number of input channels
    out_chan : int
        Number of output channels
    fused : bool
        If fused is true then torch.nn.functional.conv_transpose2d else torch.nn.Upsample
    """
    def __init__(self, in_chan, out_chan, kernel_size=3, fused=False):
        super(DiscBlock, self).__init__()
        self.conv1 = nn.Sequential(
            WSConv2d(in_chan, in_chan),
            nn.LeakyReLU(0.2),
            BlurLayer(),
        )
        if fused:
            self.conv2 = nn.Sequential(
                FusedDownSample(in_chan, out_chan, kernel_size),
                nn.LeakyReLU(0.2),
            )
        else:
            self.conv2 = nn.Sequential(
                WSConv2d(in_chan, out_chan),
                nn.AvgPool2d(2),
                nn.LeakyReLU(0.2),
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels,
        img_channels=3,
        resolution=1024,
        fmap_base=8192,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_max=512,
    ):
        super(Discriminator, self).__init__()

        # parameters ---
        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        resolution_log2 = int(math.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4

        self.progress_blocks, self.from_rgb = nn.ModuleList([]), nn.ModuleList([])

        for res in range(resolution_log2, 1, -1):
            conv_in_chan = nf(res - 1)
            conv_out_chan = nf(res - 2)
            fused = True if conv_in_chan <= 128 else False
            self.progress_blocks.append(DiscBlock(conv_in_chan, conv_out_chan, kernel_size=3, fused=fused))
            self.from_rgb.append(WSConv2d(img_channels, conv_in_chan, kernel_size=1, stride=1, padding=0))

        self.from_rgb.append(WSConv2d(img_channels, in_channels, kernel_size=1, stride=1, padding=0))

        # this is the block for 4x4 input size
        self.final_block = nn.Sequential(
            # +1 to in_channels from MiniBatch std will be concatenated
            WSConv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            View(-1),      # Flatten Layer
            WSLinear(in_channels * 4 * 4, in_channels),
            nn.LeakyReLU(0.2),
            WSLinear(in_channels, 1),
        )

        self.avg_pool = nn.AvgPool2d(2)

    def fade_in(self, alpha, downscaled, out):
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        batch_statistics = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, alpha, steps):
        curr_step = len(self.progress_blocks) - steps - 1

        out = self.from_rgb[curr_step](x)

        if steps == 0:  # for 4x4 image
            out = self.minibatch_std(out)
            return self.final_block(out)

        downscaled = self.from_rgb[curr_step + 1](self.avg_pool(x))
        out = self.progress_blocks[curr_step](out)

        # the fade_in is done first between the downscaled and the input, opposite from the generator
        out = self.fade_in(alpha, downscaled, out)

        for step in range(curr_step + 1, len(self.progress_blocks) - 1):
            out = self.progress_blocks[step](out)

        out = self.minibatch_std(out)
        return self.final_block(out)



if __name__ == "__main__":
    in_channels = 512
    disc = Discriminator(in_channels=in_channels, img_channels=3)
    tot = 0
    for param in disc.parameters():
        tot += param.numel()
    print("Total Parameters in Discriminator: {}\n\n".format(tot))

    for img_size in [1024, 512, 256, 128, 64, 32, 16, 8, 4]:
        num_steps = int(math.log2(img_size / 4))
        x = torch.randn((2, 3, img_size, img_size))
        z = disc(x, alpha=0.5, steps=num_steps)
        print("\nSuccess! At Image size: ", img_size)
