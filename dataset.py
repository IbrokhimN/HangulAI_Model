# dataset.py

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def get_dataset():
    dataset_path = "/home/ibrokhim/Документы/dataset"  # путь к твоему датасету
    dataset = ImageFolder(root=dataset_path)
    return dataset

def get_dataloader(batch_size=32):
    dataset_path = "/home/ibrokhim/Документы/dataset"
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(root=dataset_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), len(dataset.classes)

