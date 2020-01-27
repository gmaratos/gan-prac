"""a script to demonstrate the generative part of the GAN"""

import argparse
import torch
import scripts.models as M
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def extract_cmd_arguments():
    #build parser object
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--samples', type=int, default=1,
        help='number of samples to draw',
    )
    #extract and return as dictionary
    return vars(parser.parse_args())

def main():
    #parse cmd args and load model weights
    args = extract_cmd_arguments()
    model = M.GAN()
    model.load()
    #get device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train(False)
    #begin showing examples
    for _ in range(args['samples']):
        inp = torch.randn((1, 100)).to(device)
        out = model.generator(inp).reshape(28, 28)
        img = TF.to_pil_image(out)
        plt.imshow(img, cmap='gray')
        plt.show()

main()
