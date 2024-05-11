from torch.utils.data import Dataset


# CIFAR10 Dataset Class
class CIFAR10(Dataset):
    def __init__(self, dataset, transform=None):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image, label = sample["img"], sample["label"]

        if self.transform:
            image = self.transform(image)
        return image, label
