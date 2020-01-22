import argparse
import config
import scripts.dataset as D
import scripts.model as M

def extract_cmd_arguments():
    #build parser object
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_path', type=str,
        help='path to the dataset (pytorch mnist format)'
    )
    parser.add_argument(
        '--batch-size', type=int,
        default=15,
        help='batch size for training'
    )
    parser.add_argument(
        '--epochs', type=int,
        default=100,
        help='number of epochs for training'
    )
    #extract and return as dictionary
    return vars(parser.parse_args())

def main():
    #get the arguments from cmd line
    args = extract_cmd_arguments()
    #build datasets, only train for now since I am building a GAN
    train_dataset = D.build_mnist(
        args['data_path'],
        train=True,
    )
    #build model, using config from scripts/config.py
    model = M.GAN(**config.cfg)
    #fit model
    model.fit(
        train_dataset,
        args['batch_size'],
        args['epochs']
    )
    model.save()

main()
