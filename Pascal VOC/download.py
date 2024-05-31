from torchvision import transforms
from torchvision.datasets import VOCDetection

def download_VOC():

    # convert PIL image to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # download = not os.path.exists('./data')
    download = True

    # download dataset
    VOCDetection(root='./data', year='2007', image_set='train', transform=transform, download=download)
    VOCDetection(root='./data', year='2007', image_set='test', transform=transform, download=download)
    VOCDetection(root='./data', year='2007', image_set='val', transform=transform, download=download)
    VOCDetection(root='./data', year='2012', image_set='train', transform=transform, download=download)
    VOCDetection(root='./data', year='2012', image_set='val', transform=transform, download=download)

if __name__ == '__main__':
    download_VOC()