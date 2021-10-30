import torchvision.transforms as T
from torch.utils.data import DataLoader


def get_data_loader(dataset, batch_size, num_workers=1):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )


def get_transform(image_size=512):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.RandomHorizontalFlip(p=0.5),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
