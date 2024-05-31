import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class CUB_ImageFolder(Dataset):
    def __init__(self, path: str, transform: transforms.Compose, train: bool=True) -> None:
        """
        Initialize an ImageFolder like the one provided in `torchvision.datasets.ImageFolder`.
        
        Args:
        - path: The path of the root directory of the dataset.
        - transform: The transform applied to the dataset.
        - train: Boolean, return the train dataset if True else test dataset, default is True. 
        """
        super(CUB_ImageFolder, self).__init__()
        self.root = path
        self.transform = transform
        self.train = train
        self.images = []
        self.labels = []
        self.train_idx = []
        self.test_idx = []
        
        self._load_dataset()
        self._get_train_test()
        
        self.idx = self.train_idx if self.train else self.test_idx
        
    def _load_dataset(self):
        """
        Load the image path and corresponding labels from the 'images.txt'
        and 'image_class_labels.txt'. 
        """
        # load image paths
        with open(os.path.join(self.root, 'images.txt')) as f:
            for line in f:
                self.images.append(line.strip().split()[1])
        # load image labels
        with open(os.path.join(self.root, 'image_class_labels.txt')) as f:
            for line in f:
                self.labels.append(line.strip().split()[1])
        
    def _get_train_test(self):
        """
        Get the indices of the training and testing dataset from the 'train_test_split.txt'.
        """
        with open(os.path.join(self.root, 'train_test_split.txt')) as f:
            for line in f: 
                idx, is_train = map(int, line.strip().split())
                self.train_idx.append(idx) if is_train == 1 else self.test_idx.append(idx)
                
    def __len__(self):
        return len(self.idx) 

    def __getitem__(self, index):
        image_id = self.idx[index] - 1
        image_path, image_label = self.images[image_id], self.labels[image_id]
        # get raw images and apply transformation
        image_matrix = Image.open(os.path.join(self.root, 'images', image_path)).convert('RGB')
        if self.transform:
            image_matrix = self.transform(image_matrix)
        # convert the returned label into a tensor, here we need "minius one" to align with the 
        # custom that Python's index starts from 0
        image_label = torch.tensor(int(image_label) - 1)
        return image_matrix, image_label

def preprocess_data(data_dir: str, batch_size: int=64) -> tuple[DataLoader, DataLoader]:
    """
    Preprocess the CUB-200-2011 dataset and return the train and test 'DataLoader'.
    
    Args:
    - data_dir: The directory of the dataset.
    - batch_size: The number of samples in one batch, default is 64.
    
    Return:
    - train_dataloder: The dataloader of the training dataset.
    - test_loader: The dataloader of the testing dataset.
    """
    # resize and normalize the images. Apply data augumentation to the training dataset.
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=(-15, 15)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # load the dataset and extract the train/test Dataloader
    train_dataset = CUB_ImageFolder(data_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = CUB_ImageFolder(data_dir, transform=test_transform, train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
