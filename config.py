from yacs.config import CfgNode as CN

cfg = CN()

cfg.resolution = 512
cfg.initial_resolution = 4
cfg.output_dir = ''

cfg.loss = "wgan-gp"
cfg.learning_rate = 0.003
cfg.disc_repeats = 1
cfg.use_ema = True
cfg.ema_decay = 0.999
cfg.clip_grad_norm = 10

cfg.num_workers = 2

# ---------------------------------------------------------------------------- #
# Options for training utils
# ---------------------------------------------------------------------------- #
cfg.train = CN()

# example for {depth:9, resolution:1024}
# res --> [4, 8, 16, 32, 64, 128, 256, 512, 1024]
cfg.train.epochs = [4, 4, 8, 8, 16, 16, 32, 64, 64]
# batches for oen 1080Ti with 11G memory
cfg.train.batch_sizes = [128, 128, 128, 64, 32, 16, 8, 4, 2]

# TODO
# cfg.train.G_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
# cfg.train.D_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}

# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.data_dir = "ProgressiveGAN/data/celeba/img_align_celeba"


cfg.model = CN()
cfg.model.in_channels = 512
cfg.model.img_channels = 3
# ---------------------------------------------------------------------------- #
# Options for Generator
# ---------------------------------------------------------------------------- #
cfg.model.gen = CN()
cfg.model.gen.latent_dim = 512
cfg.model.gen.dlatent_dim = 512
cfg.model.gen.truncation_psi = 0.7
cfg.model.gen.truncation_cutoff = 8
cfg.model.gen.dlatents_avg_beta = 0.995
cfg.model.gen.style_mixing_prob = 0.7

# ---------------------------------------------------------------------------- #
# Options for Discriminator
# ---------------------------------------------------------------------------- #
cfg.model.dis = CN()