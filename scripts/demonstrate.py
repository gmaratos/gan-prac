"""a script to demonstrate the generative part of the GAN"""

import argparse
import config
import scripts.model as M
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def extract_cmd_arguments():
    #build parser object
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'weight_path', type=str,
        help='path dir containing weights of network',
    )
    parser.add_argument(
        '--samples', type=int, default=1,
        help='number of samples to draw',
    )
    #extract and return as dictionary
    return vars(parser.parse_args())

def main():
    #parse cmd args and load model weights
    args = extract_cmd_arguments()
    model = M.GAN(**config.cfg)
    model.load(args['weight_path'])
    #get device
    device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    model.to(device)
    model.train(False)
    #begin showing examples
    for _ in range(args['samples']):
        inp = torch.normal(0, 1, (1, model.g_inp_size)).to(device)
        out = model.generator(inp).reshape(28, 28)
        img = TF.to_pil_image(out)
        plt.imshow(img)
        plt.show()

main()
