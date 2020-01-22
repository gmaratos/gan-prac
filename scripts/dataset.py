import torchvision.datasets as td
import torchvision.transforms as tt

def build_mnist(path: str, train: bool):
    """build torch dataset object, training loop will handle batches"""

    #currently I am just normalizing the data
    transforms = tt.Compose([
        tt.ToTensor(),
    ])
    #next build the dataset with all the arguments
    dataset = td.MNIST(
        path, train=train,
        transform=transforms
    )
    return dataset
