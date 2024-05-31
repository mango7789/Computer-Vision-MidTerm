import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models

class CUB_ResNet_18(nn.Module):
    def __init__(self, num_classes: int=200, pretrain: bool=True):
        """
        Create a neural network with the same architecture as ResNet-18. The output layer is 
        resized to (`in_features`, `num_classes`) to fit into the specific dataset.
        
        Args:
        - num_classes: Number of classes(labels), default is 200.
        - pretrain: Boolean, whether the paramters of ResNet-18 is pretrained or not. Default
        is True.
        """
        super(CUB_ResNet_18, self).__init__()
        # initialize the parameters
        if pretrain:
            self.resnet18 = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
        else:
            self.resnet18 = models.resnet18(weights=None)
            
        # change the output layer
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)
        
        # apply Kaiming initialization to the fully connected layer
        init.kaiming_normal_(self.resnet18.fc.weight, mode='fan_out', nonlinearity='relu')
        if self.resnet18.fc.bias is not None:
            nn.init.constant_(self.resnet18.fc.bias, 0)
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass of the network.
        """
        return self.resnet18(x)