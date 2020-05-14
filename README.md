# MNIST Gan
This is a very simple example of a Generative Adversarial Network implementation. This network learns to generate images that look like handwritten digits. It is written in pytorch. I was able to learn how to implement this with the help of George Seif's medium post, https://towardsdatascience.com/an-easy-introduction-to-generative-adversarial-networks-6f8498dc4bcd.

## Running the code
You can see all the command line arguments for training the network by running the following command.
```
python -m scripts.main -h
```
It is not a very smart script, it does look for a graphics card but if it is available it just uses *cuda:0* reguardless. When training is done the weights are saved in the directory *weights/* (script is broken and actually this part is missing, will fix later).

## Demonstrate
Once the weights are saved you can view some generated samples. You can see the options for running the demonstration with the following command.

```
python -m scripts.demonstrate -h
```

## requirements
pytorch, torchvision, tqdm, matplotlib
