"""
-------------------------------------------------
   File Name:    Losses.py
   Author:       Zhonghao Huang
   Date:         2019/10/21
   Description:  Module implementing various loss functions
                 Copy from: https://github.com/akanimax/pro_gan_pytorch
-------------------------------------------------
"""

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import numpy as np
from utils import gradient_penalty

# =============================================================
# Interface for the losses
# =============================================================

class GANLoss:
    """ Base class for all losses
        @args:
        disc: Discriminator used for calculating the loss
             Note this must be a part of the GAN framework
    """

    def __init__(self, disc):
        self.disc = disc

    def disc_loss(self, real_samps, fake_samps, alpha, height):
        """
        calculate the discriminator loss using the following data
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("disc_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps, alpha, height):
        """
        calculate the generator loss
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("gen_loss method has not been implemented")


class ConditionalGANLoss:
    """ Base class for all conditional losses """

    def __init__(self, disc):
        self.disc = disc

    def disc_loss(self, real_samps, fake_samps, labels, alpha, height):
        raise NotImplementedError("disc_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps, labels, alpha, height):
        raise NotImplementedError("gen_loss method has not been implemented")


# =============================================================
# Normal versions of the Losses:
# =============================================================

class StandardGAN(GANLoss):

    def __init__(self, disc):
        super().__init__(disc)
        # define the criterion and activation used for object
        self.criterion = BCEWithLogitsLoss()

    def disc_loss(self, real_samps, fake_samps, alpha, height):
        # small assertion:
        assert real_samps.device == fake_samps.device, \
            "Real and Fake samples are not on the same device"

        # device for computations:
        device = fake_samps.device

        # predictions for real images and fake images separately :
        r_preds = self.disc(real_samps, alpha, height)
        f_preds = self.disc(fake_samps, alpha, height)

        # calculate the real loss:
        real_loss = self.criterion(
            torch.squeeze(r_preds),
            torch.ones(real_samps.shape[0]).to(device))

        # calculate the fake loss:
        fake_loss = self.criterion(
            torch.squeeze(f_preds),
            torch.zeros(fake_samps.shape[0]).to(device))

        # return final losses
        return (real_loss + fake_loss) / 2

    def gen_loss(self, _, fake_samps, alpha, height):
        preds, _, _ = self.disc(fake_samps, alpha, height)
        return self.criterion(
            torch.squeeze(preds),
            torch.ones(fake_samps.shape[0]).to(fake_samps.device))


class WGAN_GP(GANLoss):

    def __init__(self, disc, LAMBDA_GP=10):
        super().__init__(disc)
        self.LAMBDA_GP = LAMBDA_GP

    def disc_loss(self, real_samps, fake_samps, alpha, height):
        critic_real = self.disc(real_samps, alpha, height)
        critic_fake = self.disc(fake_samps, alpha, height)
        gp = gradient_penalty(self.disc, real_samps, fake_samps, alpha, height, device=real_samps.device)
        loss = (
            -(torch.mean(critic_real) - torch.mean(critic_fake))
            + self.LAMBDA_GP * gp
            + (0.001 * torch.mean(critic_real ** 2))
        )
        return loss

    def gen_loss(self, _, fake_samps, alpha, height):
        return -torch.mean(self.disc(fake_samps, alpha, height))


class HingeGAN(GANLoss):

    def __init__(self, disc):
        super().__init__(disc)

    def disc_loss(self, real_samps, fake_samps, alpha, height):
        r_preds = self.disc(real_samps, alpha, height)
        f_preds = self.disc(fake_samps, alpha, height)

        loss = (torch.mean(nn.ReLU()(1 - r_preds)) +
                torch.mean(nn.ReLU()(1 + f_preds)))

        return loss

    def gen_loss(self, _, fake_samps, alpha, height):
        return -torch.mean(self.disc(fake_samps, alpha, height))


class RelativisticAverageHingeGAN(GANLoss):

    def __init__(self, disc):
        super().__init__(disc)

    def disc_loss(self, real_samps, fake_samps, alpha, height):
        # Obtain predictions
        r_preds = self.disc(real_samps, alpha, height)
        f_preds = self.disc(fake_samps, alpha, height)

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        loss = (torch.mean(nn.ReLU()(1 - r_f_diff))
                + torch.mean(nn.ReLU()(1 + f_r_diff)))

        return loss

    def gen_loss(self, real_samps, fake_samps, alpha, height):
        # Obtain predictions
        r_preds = self.disc(real_samps, alpha, height)
        f_preds = self.disc(fake_samps, alpha, height)

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        return (torch.mean(nn.ReLU()(1 + r_f_diff))
                + torch.mean(nn.ReLU()(1 - f_r_diff)))


class LogisticGAN(GANLoss):
    def __init__(self, disc):
        super().__init__(disc)

    # gradient penalty
    def R1Penalty(self, real_img, alpha, height):

        # TODO: use_loss_scaling, for fp16
        apply_loss_scaling = lambda x: x * torch.exp(x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))
        undo_loss_scaling = lambda x: x * torch.exp(-x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))

        real_img = torch.autograd.Variable(real_img, requires_grad=True)
        real_logit = self.disc(real_img, alpha, height)
        # real_logit = apply_loss_scaling(torch.sum(real_logit))
        real_grads = torch.autograd.grad(outputs=real_logit, inputs=real_img,
                                         grad_outputs=torch.ones(real_logit.size()).to(real_img.device),
                                         create_graph=True, retain_graph=True)[0].view(real_img.size(0), -1)
        # real_grads = undo_loss_scaling(real_grads)
        r1_penalty = torch.sum(torch.mul(real_grads, real_grads))
        return r1_penalty

    def disc_loss(self, real_samps, fake_samps, alpha, height, r1_gamma=10.0):
        # Obtain predictions
        r_preds = self.disc(real_samps, alpha, height)
        f_preds = self.disc(fake_samps, alpha, height)

        loss = torch.mean(nn.Softplus()(f_preds)) + torch.mean(nn.Softplus()(-r_preds))

        if r1_gamma != 0.0:
            r1_penalty = self.R1Penalty(real_samps.detach(), alpha, height) * (r1_gamma * 0.5)
            loss += r1_penalty

        return loss

    def gen_loss(self, _, fake_samps, alpha, height):
        f_preds = self.disc(fake_samps, alpha, height)

        return torch.mean(nn.Softplus()(-f_preds))