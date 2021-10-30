import torch


def gradient_penalty(critic, real, fake, alpha, depth, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, depth)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def update_average(target_model, source_model, beta):
    """Calculate Exponential Moving Average for the Generator weights.
    This function updates the exponential average weights based on the current training
    reference: https://github.com/akanimax/pro_gan_pytorch
    """
    def toggle_grad(model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)

    # turn off gradient calculation
    toggle_grad(target_model, False)
    toggle_grad(source_model, False)

    src_param_dict = dict(source_model.named_parameters())
    source_param_names = list(src_param_dict.values())

    for name, param in target_model.named_parameters():
        if name not in source_param_names:
            new_weight = param.clone().detach()
        else:
            new_weight = beta * param + (1 - beta) * src_param_dict[name]
        param.copy_(new_weight)

    # turn back on the gradient calculation
    toggle_grad(target_model, True)
    toggle_grad(source_model, True)