import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

class Loader():
    def __init__(self, **kwargs):
        super().__init__()
        batches = kwargs["batch_size"]
        print('Wait for both datasets to be downloaded and verified.')

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) 
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batches, shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=1)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.trainloader = trainloader
        self.testloader = testloader

