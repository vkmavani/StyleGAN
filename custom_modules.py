import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WSConv2d(nn.Module):
    """Weighted-Scaled Convolution.
    Parameters
    ----------
    gain : int
        Initialization constant (he-initialization)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * kernel_size ** 2)) ** 0.5

        # bias should not be scaled
        self.bias = self.conv.bias
        self.conv.bias = None

        # Initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class WSLinear(nn.Module):
    """Weighted-Scaled Linear layer.
    Parameters
    ----------
    gain : int
        Initialization constant (he-initialization)
    """
    def __init__(self, in_channels, out_channels, gain=2):
        super(WSLinear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.scale = (gain / in_channels) ** 0.5

        # bias should not be scaled
        self.bias = self.linear.bias
        self.linear.bias = None

        # Initialize conv layer
        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.linear(x * self.scale) + self.bias


class NoiseInjection(nn.Module):
    """Inject noise before each AdaIn block.
    The noise tensor is not entirely random, it is initialized as one random channel that is then
    multiplied(broadcast) by learned weights for each channel in the image.
    """
    def __init__(self):
        super(NoiseInjection, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            noise_shape = (image.shape[0], 1, image.shape[2], image.shape[3])
            noise = image.new_empty(*noise_shape).normal_()
        return image + self.weight * noise


class AdaIN(nn.Module):
    """Adaptive Instance Normalization.
    This will inject 'w'(intermediate noise) to Generator multiple times to increase the control over the image.

    Parameters
    ----------
    n_channels : int
        Number of channels in image
    w_dim : int
        The dimension of intermediate noise vector
    """
    def __init__(self, n_channels, w_dim):
        super(AdaIN, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(n_channels)
        self.style_scale_transform = WSLinear(w_dim, n_channels)
        self.style_shift_transform = WSLinear(w_dim, n_channels)

    def forward(self, image, w):
        normalized_image = self.instance_norm(image)
        style_scale = self.style_scale_transform(w)[:, :, None, None]
        style_shift = self.style_shift_transform(w)[:, :, None, None]   # style bias
        transformed_image = style_scale * normalized_image + style_shift
        return transformed_image


class PixelNormLayer(nn.Module):
    """Used before noise mapping to normalize the noise or latents."""
    def __init__(self, epsilon=1e-8):
        super(PixelNormLayer, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class FusedUpSample(nn.Module):
    """FusedUpSample uses torch.nn.functional.conv_transpose2d (learnable parameters)."""
    def __init__(self, in_chan, out_chan, kernel_size):
        super(FusedUpSample, self).__init__()
        weight = torch.randn(in_chan, out_chan, kernel_size, kernel_size)
        bias = torch.zeros(out_chan)
        fan_in = in_chan * kernel_size * kernel_size
        self.multiplier = math.sqrt(2 / fan_in)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4
        out = F.conv_transpose2d(input, weight, self.bias, stride=2, padding=(weight.size(-1) - 1) // 2)
        return out


class FusedDownSample(nn.Module):
    """FusedDownSample uses torch.nn.functional.conv2d (learnable parameters)."""
    def __init__(self, in_chan, out_chan, kernel_size):
        super(FusedDownSample, self).__init__()
        weight = torch.randn(out_chan, in_chan, kernel_size, kernel_size)
        bias = torch.zeros(out_chan)
        fan_in = in_chan * kernel_size * kernel_size
        self.multiplier = math.sqrt(2 / fan_in)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4
        out = F.conv2d(input, weight, self.bias, stride=2, padding=(weight.size(-1) - 1) // 2)
        return out


class View(nn.Module):
    """To change the shape of the layer. (To make Flatten layer.)"""
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class BlurLayer(nn.Module):
    """To mitigate the effect during upScaling as 'nn.UpSample' and 'FusedUpSample' will not be equivalent."""
    def __init__(self, normalize=True, flip=False, stride=1):
        super(BlurLayer, self).__init__()
        kernel = [1, 2, 1]
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None]
        if normalize:
            kernel = kernel / kernel.sum()
        if flip:
            kernel = kernel[:, :, ::-1, ::-1]
        self.register_buffer('kernel', kernel)
        self.stride = stride

    def forward(self, x):
        # expand kernel channels
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        x = F.conv2d(
            x,
            kernel,
            stride=self.stride,
            padding=int((self.kernel.size(2) - 1) / 2),
            groups=x.size(1)
        )
        return x


class Truncation(nn.Module):
    def __init__(self, avg_latent, max_layer=8, threshold=0.7, beta=0.995):
        super().__init__()
        self.max_layer = max_layer
        self.threshold = threshold
        self.beta = beta
        self.register_buffer('avg_latent', avg_latent)

    def update(self, last_avg):
        self.avg_latent.copy_(self.beta * self.avg_latent + (1. - self.beta) * last_avg)

    def forward(self, x):
        assert x.dim() == 3
        interp = torch.lerp(self.avg_latent, x, self.threshold)
        do_trunc = (torch.arange(x.size(1)) < self.max_layer).view(1, -1, 1).to(x.device)
        return torch.where(do_trunc, interp, x)
