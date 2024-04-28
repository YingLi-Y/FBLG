import os
import torchvision

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])
root = os.path.join('benchmark','RAW_DATA', 'SVHN')
train_data = torchvision.datasets.SVHN(root=root,transform=transform, download=True, split='train')
test_data = torchvision.datasets.SVHN(root=root, transform=transform, download=True, split='test')