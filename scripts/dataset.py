import os
import torchvision
import torchvision.transforms as TT

def build_dataset():
    #check if data directory is present
    if not os.path.exists('data'):
        os.mkdir('data')
    #construct transformations
    transforms = TT.Compose([
        TT.ToTensor(),
        TT.Normalize((0.5,), (0.5,)),
    ])
    #return dataset object
    return torchvision.datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=transforms,
    )
