import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_modules import WSLinear, WSConv2d, AdaIN, NoiseInjection, PixelNormLayer, FusedUpSample, BlurLayer, Truncation


class GMapping(nn.Module):
    """Generator Mapping Network. Used to map noise 'z' --> 'w'.

    Parameters
    ----------
    z_dim : int
        The dimension of the noise vector
    w_dim : int
        The dimension of the mapped vector
    """
    def __init__(self, z_dim, w_dim, dlatent_broadcast=None):
        super(GMapping, self).__init__()
        self.dlatent_broadcast = dlatent_broadcast
        layers = [PixelNormLayer()]
        for _ in range(7):
            layers.append(WSLinear(z_dim, w_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
            z_dim = w_dim
        layers.append(WSLinear(z_dim, w_dim))
        self.mapping = nn.Sequential(*layers)

    def forward(self, noise):
        x = self.mapping(noise)
        if self.dlatent_broadcast is not None:
            # [bs, dlatent_dim]  =>  Broadcast to [batch_size, dlatent_broadcast, dlatent_dim]
            # Make the copy of the same vector dlatent_broadcast times
            x = x.unsqueeze(1).expand(-1, self.dlatent_broadcast, -1)
        return x


class InputBlock(nn.Module):
    """Initial Block for 4x4 image size.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    dlatents_dim : int
        Disentangled latent (W) dimensionality
    """
    def __init__(self, in_channels, dlatents_dim):
        super(InputBlock, self).__init__()

        self.initial_constant = nn.Parameter(torch.ones((1, in_channels, 4, 4)))

        self.initial_noise1 = NoiseInjection()
        self.initial_adain1 = AdaIN(in_channels, dlatents_dim)
        self.initial_conv = WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.initial_noise2 = NoiseInjection()
        self.initial_adain2 = AdaIN(in_channels, dlatents_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, dlatents):
        x = self.initial_adain1(self.initial_noise1(self.initial_constant), dlatents[:, 0])
        x = self.initial_conv(x)
        x = self.initial_adain2(self.initial_noise2(x), dlatents[:, 1])
        return x


class GSynthesisBlock(nn.Module):
    """Final StyleGAN generator component, which will apply the Progressive Growing of Generator.

    Parameters
    ----------
    in_chan : int
        Number of input channels
    out_chan : int
        Number of output channels
    dlatents_dim : int
        Dimension of the intermediate noise vector
    fused : bool
        If fused is true then torch.nn.functional.conv_transpose2d else torch.nn.Upsample
    """
    def __init__(
        self, in_chan, out_chan, kernel_size=3, dlatents_dim=512, fused=False
    ):
        super(GSynthesisBlock, self).__init__()
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
        self.adain1 = AdaIN(out_chan, dlatents_dim)
        self.conv2 = WSConv2d(out_chan, out_chan)
        self.inject_noise2 = NoiseInjection()
        self.adain2 = AdaIN(out_chan, dlatents_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, dlatents):
        x = self.activation(self.inject_noise1(self.conv1(x)))
        x = self.adain1(x, dlatents[:, 0])
        x = self.activation(self.inject_noise2(self.conv2(x)))
        x = self.adain2(x, dlatents[:, 1])
        return x


class GSynthesis(nn.Module):
    """Synthesis network used in the StyleGAN.

    Parameters
    ----------
    dlatents_dim : int
        Disentangled latent (W) dimensionality
    in_channels : int
        number of channels for least image size (4x4)
    img_channels : int
        Number of image output color channels
    resolution : int
        Generated Image resolution
    fmap_base : int (fixed -> 8192)
        Overall multiplier for the number of feature maps
    fmap_decay : int (fixed -> 1.0)
        log2 feature map reduction when doubling the resolution
    fmap_max : int (fixed -> 512)
        Maximum number of feature maps in any layer
    """
    def __init__(
        self,
        dlatents_dim,
        in_channels=512,
        img_channels=3,
        resolution=1024,
        fmap_base=8192,
        fmap_decay=1.0,
        fmap_max=512,
    ):
        super(GSynthesis, self).__init__()

        # parameters ---
        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        resolution_log2 = int(math.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4

        self.initial_block = InputBlock(in_channels, dlatents_dim)
        self.initial_rgb = WSConv2d(in_channels, img_channels, kernel_size=1, stride=1, padding=0)

        self.progress_blocks, self.to_rgb = nn.ModuleList([]), nn.ModuleList([self.initial_rgb])

        for res in range(2, resolution_log2):
            conv_in_chan  = nf(res - 1)
            conv_out_chan = nf(res)
            fused = True if conv_in_chan <= 256 else False
            self.progress_blocks.append(
                GSynthesisBlock(conv_in_chan, conv_out_chan, kernel_size=3, dlatents_dim=dlatents_dim, fused=fused))
            self.to_rgb.append(WSConv2d(conv_out_chan, img_channels, kernel_size=1, stride=1, padding=0))

    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1-alpha) * upscaled)

    def forward(self, dlatents, alpha, steps):
        """
        Parameters
        ----------
        dlatents : torch.Tensor
            Disentangled latents (W) [mini_batch, num_layers, dlatents_dim]
        alpha : float (0, 1]
            Value of alpha for fade-in effect
        steps : int
            Current depth from where output is required
        """
        x = self.initial_block(dlatents[:, 0:2])
        if steps == 0:
            return self.initial_rgb(x)

        for idx, step in enumerate(range(steps - 1)):
            x = self.progress_blocks[step](x, dlatents[:, 2*(idx + 1):2*(idx + 2)])

        final_upScaled = self.to_rgb[steps - 1](F.interpolate(x, scale_factor=2, mode="bilinear"))
        final_out = self.to_rgb[steps](self.progress_blocks[steps - 1](x, dlatents[:, 2*steps:2*(steps + 1)]))
        return self.fade_in(alpha, final_upScaled, final_out)


class Generator(nn.Module):
    """Style-based Generator, composed of two sub networks => GMapping + GSynthesis

    Parameters
    ----------
    latent_dim : int
        Input Noise vector dimension
    dlatents_dim : int
        Disentangled latent (W) dimensionality
    in_channels : int
        number of channels for least image size (4x4)
    img_channels : int
        Number of image output color channels
    resolution : int
        Generated Image resolution
    truncation_psi : float
        Style strength multiplier for the truncation trick  [ None = disable ]
    truncation_cutoff : int (range -> 0-8)
        Number of layers for which to apply the truncation trick  [ None = disable ]
    dlatents_avg_beta : float => (0,1)
        Decay for tracking the moving average of W during training  [ None = disable ]
    style_mixing_prob : float => (0, 1)
        Probability of mixing styles during training  [ None = disable ]
    """
    def __init__(
        self,
        latent_dim=512,
        dlatents_dim=512,
        in_channels=512,
        img_channels=3,
        resolution=1024,
        truncation_psi=0.7,
        truncation_cutoff=8,
        dlatents_avg_beta=0.995,
        style_mixing_prob=0.9
    ):
        super(Generator, self).__init__()

        self.style_mixing_prob = style_mixing_prob
        self.num_layers = (int(np.log2(resolution)) - 1) * 2
        self.g_mapping = GMapping(latent_dim, dlatents_dim, dlatent_broadcast=self.num_layers)
        self.g_synthesis = GSynthesis(
            dlatents_dim=dlatents_dim, in_channels=in_channels, img_channels=img_channels, resolution=resolution)

        if truncation_psi > 0:
            self.truncation = Truncation(
                avg_latent=torch.zeros(dlatents_dim),
                max_layer=truncation_cutoff,
                threshold=truncation_psi,
                beta=dlatents_avg_beta)
        else:
            self.truncation = None

    def forward(self, noise, alpha, steps):
        dlatents = self.g_mapping(noise)

        if self.training:
            # Update moving average of W(dlatents)
            if self.truncation is not None:
                self.truncation.update(dlatents[0, 0].detach())

            # Perform style mixing regularization
            if self.style_mixing_prob is not None and self.style_mixing_prob > 0:
                latents2 = torch.randn(noise.shape).to(noise.device)
                dlatents2 = self.g_mapping(latents2)
                layer_idx = torch.from_numpy(np.arange(self.num_layers)[np.newaxis, :, np.newaxis]).to(noise.device)   # [1, self.num_layers, 1]
                cur_layers = 2 * (steps + 1)
                mixing_cutoff = random.randint(1, cur_layers) if random.random() < self.style_mixing_prob else cur_layers
                dlatents = torch.where(layer_idx < mixing_cutoff, dlatents, dlatents2)

            # Apply truncation trick.
            if self.truncation is not None:
                dlatents = self.truncation(dlatents)

        fake_images = self.g_synthesis(dlatents, alpha, steps)
        return fake_images


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
        x = torch.randn((2, z_dim))
        z = gen(x, alpha=0.5, steps=num_steps)
        assert z.shape == (2, 3, img_size, img_size)
        print("\nSuccess! At Image size: ", img_size)
