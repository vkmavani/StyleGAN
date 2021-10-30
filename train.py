import copy
import numpy as np
from tqdm.auto import tqdm

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets

from config import cfg
import loss_fn
from utils import update_average
from temp_generator import Generator
from discriminator import Discriminator
from data_utils import get_data_loader, get_transform


class StyleGAN:
    """Wrapper around Generator and Discriminator.

    Parameters
    ----------
    resolution : int
        Generated Image resolution
    latent_dim : int
        Input Noise vector dimension
    dlatent_dim : int
        Disentangled latent (W) dimensionality
    in_channels : int
        number of channels for least image size (4x4)
    img_channels : int
        Number of image output color channels
    gen_args : args
        Options for Generator network
    disc_args : args
        Options for Discriminator network
    learning_rate : float
        Learning rate for both optimizers
    clip_grad_norm : int
        For Gradient Clipping
    use_ema : bool
        Whether to use Exponential Moving Averages
    ema_decay : float -> (0,1)
        EMA decay value
    device : str or torch.device()
        Device -> CPU/GPU for training
    """
    def __init__(
        self,
        resolution,
        latent_dim,
        dlatent_dim,
        in_channels,
        img_channels,
        learning_rate,
        clip_grad_norm=10,
        use_ema=False,
        ema_decay=0.999,
        device=torch.device("cpu")
    ):
        self.depth = int(np.log2(resolution)) - 1
        self.latent_dim = latent_dim
        self.disc_repeats = 1   # number of times the discriminator will be trained per G(gen.) iteration
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.device = device
        self.loss = None

        # Create the Generator and the Discriminator
        self.gen = Generator(
            latent_dim=latent_dim,
            dlatent_dim=dlatent_dim,
            in_channels=in_channels,
            img_channels=img_channels,
            resolution=resolution,
        ).to(self.device)
        self.disc = Discriminator(
            in_channels=in_channels,
            img_channels=img_channels,
            resolution=resolution,
        ).to(self.device)

        # define the optimizers for the discriminator and generator
        self.gen_optim = optim.Adam(
            params = [{"params": [param for name, param in self.gen.named_parameters() if "g_mapping" not in name]},
                      {"params": self.gen.g_mapping.parameters(), "lr": 1e-5}],
            lr=learning_rate,
            betas=(0.0, 0.99)
        )
        self.disc_optim = optim.Adam(
            params=self.disc.parameters(),
            lr=learning_rate,
            betas=(0.0, 0.99)
        )
        self.clip_grad_norm = clip_grad_norm

        # Use of EMA
        if self.use_ema:
            self.gen_shadow = copy.deepcopy(self.gen)    # create a shadow copy of the generator
            self.ema_updater = update_average   # EMA updating function
            self.ema_updater(self.gen_shadow, self.gen, beta=0)     # initialize the gen_shadow weights equal to the weights of gen

    def _setup_loss(self, loss):
        if isinstance(loss, str):
            loss = loss.lower()
            if loss == "standard-gan":
                loss = loss_fn.StandardGAN(self.disc)
            elif loss == "wgan-gp":
                loss = loss_fn.WGAN_GP(self.disc)
            elif loss == "hinge":
                loss = loss_fn.HingeGAN(self.disc)
            elif loss == "relativistic-hinge":
                loss = loss_fn.RelativisticAverageHingeGAN(self.disc)
            elif loss == "logistic":
                loss = loss_fn.LogisticGAN(self.disc)
            else:
                raise ValueError("Unknown loss function requested")
            return loss

        elif isinstance(loss, loss_fn.GANLoss):
            return loss

        else:
            raise ValueError("loss is neither an instance of GANLoss nor a string")

    def _progressive_down_sampling(self, real_batch, alpha, depth):
        """Helper for down_sampling the original images in order to facilitate the progressive growing of the layers."""

        down_sample_factor = int(np.power(2, self.depth - depth - 1))
        prior_down_sample_factor = max(int(np.power(2, self.depth - depth)), 0)
        ds_real_samples = nn.AvgPool2d(down_sample_factor)(real_batch)

        if depth > 0:
            prior_ds_real_samples = F.interpolate(nn.AvgPool2d(prior_down_sample_factor)(real_batch), scale_factor=2)
        else:
            prior_ds_real_samples = ds_real_samples

        # apply fade-in between ds_real_samples and prior_ds_real_samples
        return (alpha * ds_real_samples) + ((1 - alpha) * prior_ds_real_samples)

    def optimize_discriminator(self, fake_samples, real_batch, alpha, depth):
        """Perform one training step on Discriminator."""
        real_samples = self._progressive_down_sampling(real_batch, alpha, depth)

        final_loss = 0
        for _ in range(self.disc_repeats):
            loss = self.loss.disc_loss(real_samples, fake_samples.detach(), alpha, depth)

            self.disc_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.disc.parameters(), self.clip_grad_norm)
            self.disc_optim.step()
            final_loss += loss.item()

        return final_loss / self.disc_repeats

    def optimize_generator(self, fake_samples, real_batch, alpha, depth):
        """Perform one training step on Generator."""
        real_samples = self._progressive_down_sampling(real_batch, alpha, depth)

        self.gen_optim.zero_grad()
        loss = self.loss.gen_loss(real_samples, fake_samples, alpha, depth)
        loss.backward()
        self.gen_optim.step()

        if self.use_ema:
            self.ema_updater(self.gen_shadow, self.gen, self.ema_decay)

        return loss.item()

    def train(
        self,
        dataset,
        batch_sizes,
        epochs,
        initial_resolution=4,
        disc_repeats=1,
        loss="wgan-gp",
        num_workers=1
    ):
        """To train the StyleGAN.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Object of the Dataset for training
        batch_size : list
            List of training batch sizes for every resolution
        epochs : list
            List of number of epochs to train the network for every resolution
        initial_resolution : int
            Starting resolution or least resolution
        disc_repeats : int
            Number of times the discriminator will be trained per G(gen.) iteration
        loss : str or loss_fn.GANLoss or loss_fn.ConditionalGANLoss
            Loss function to be used,
            Can be from => ["wgan-gp", "standard-gan", "hinge", "relativistic-hinge", "logistic"]
        num_workers : int
            number of workers to read data (default -> 1)
        """
        assert self.depth <= len(epochs), "epochs not compatible with depth"
        assert self.depth <= len(batch_sizes), "batch_sizes not compatible with depth"

        self.gen.train()
        self.disc.train()
        if self.use_ema: self.gen_shadow.train()

        self.disc_repeats = disc_repeats
        self.loss = self._setup_loss(loss)

        start_depth = int(np.log2(initial_resolution / 4))
        for current_depth in range(start_depth, self.depth):
            print(f"Current image size: {4 * 2 ** current_depth}")

            dataloader = get_data_loader(dataset, batch_sizes[current_depth], num_workers)
            alpha = 1e-5   # start with very low alpha

            for epoch in range(epochs[current_depth]):
                tk = tqdm(dataloader)
                for batch_idx,(real_images,_) in enumerate(tk):
                    real_images = real_images.to(self.device)
                    gan_input = torch.randn(real_images.shape[0], self.latent_dim).to(self.device)
                    fake_samples = self.gen(gan_input, alpha, current_depth)

                    disc_loss = self.optimize_discriminator(fake_samples, real_images, alpha, current_depth)

                    gen_loss = self.optimize_generator(fake_samples, real_images, alpha, current_depth)

                    # Update alpha and ensure less than 1
                    alpha += real_images.shape[0] / ((epochs[current_depth] * 0.5) * len(dataset))
                    alpha = min(alpha, 1)

                    tk.set_postfix(gen_loss=gen_loss, disc_loss=disc_loss)


if __name__ == "__main__":
    dataset = datasets.ImageFolder(root=cfg.dataset.data_dir, transform=get_transform(image_size=cfg.resolution))
    stylegan = StyleGAN(
        resolution=cfg.resolution,
        latent_dim=cfg.model.gen.latent_dim,
        dlatent_dim=cfg.model.gen.dlatent_dim,
        in_channels=cfg.model.in_channels,
        img_channels=cfg.model.img_channels,
        learning_rate=cfg.learning_rate,
        clip_grad_norm=cfg.clip_grad_norm,
        use_ema=cfg.use_ema,
        ema_decay=cfg.ema_decay,
        device = "cuda" if torch.cuda.is_available() else "cpu"
    )

    stylegan.train(
        dataset=dataset,
        batch_sizes=cfg.train.batch_sizes,
        epochs=cfg.train.epochs,
        initial_resolution=cfg.initial_resolution,
        disc_repeats=cfg.disc_repeats,
        loss=cfg.loss,
        num_workers=cfg.num_workers,
    )