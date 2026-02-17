import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import MultiLabelDataset


def get_dataloader(image_dir, label_file, batch_size=16):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = MultiLabelDataset(
        image_dir=image_dir,
        label_file=label_file,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0   # Windows safe
    )

    return dataloader
