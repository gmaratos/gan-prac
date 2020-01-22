import os
import torch
import tqdm
import torch.nn as nn
import torch.optim as TO
import torch.utils.data as TU
import torch.nn.functional as F

class SimpleDiscriminator(nn.Module):
    """the simple fully connected feedforward network discriminator"""
    def __init__(self, input_size: int):
        super(SimpleDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, int(1.5*input_size)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(int(1.5*input_size), input_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(input_size, int(0.75*input_size)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(int(0.75*input_size), 1), #discriminator always outputs 1
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1) #flatten input to a vector
        x = self.net(x)
        return x

class SimpleGenerator(nn.Module):
    """a simple generator fully connected one hidden layer"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SimpleGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 784),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """the input is expected to be a flat vector"""
        x = self.net(x)
        return x

class GAN(nn.Module):
    """Generative Adversarial Network, has a generator and a discriminator"""
    def __init__(self, **kwargs):
        super(GAN, self).__init__()
        #extract configurations for networks
        g_inp = kwargs['generator_input_size']
        g_hidden = kwargs['generator_hidden_size']
        g_out = kwargs['input_size']
        d_inp = kwargs['input_size']
        d_hidden = kwargs['discriminator_hidden_size']
        #build networks
        self.g_inp_size = kwargs['generator_input_size']
        self.generator = SimpleGenerator(g_inp, g_hidden, g_out)
        self.discriminator = SimpleDiscriminator(d_inp)

    def forward(self, x):
        pass

    def fit(self, dataset, batch_size: int, epochs: int):
        """training procedure for gan"""
        #build dataloader
        data = TU.DataLoader(dataset, batch_size=batch_size)
        #set up optimizers
        generative_optim = TO.SGD(self.generator.parameters(), lr=1e-3)
        discriminator_optim = TO.SGD(self.discriminator.parameters(), lr=1e-3)
        #get device
        device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device_str)
        self.to(device)
        #begin training
        for i in range(1, epochs+1):
            print('Epoch {}'.format(i))
            t = tqdm.tqdm(data)
            g_totals, d_totals = [], []
            for inp, _ in t:
                #first train the generator
                g_inp = torch.normal(0, 1, (batch_size, self.g_inp_size)).to(device)
                g_out = self.generator(g_inp)
                d_out = self.discriminator(g_out)
                target = torch.zeros((batch_size, 1)).to(device)
                #calculate loss and do backpropogation
                g_loss = F.binary_cross_entropy_with_logits(d_out, target)
                generative_optim.zero_grad()
                g_loss.backward()
                generative_optim.step()
                #next train discriminator
                g_inp = torch.normal(0, 1, (batch_size, self.g_inp_size)).to(device)
                g_out = self.generator(g_inp).detach()
                d_inp = torch.cat([g_out, torch.flatten(inp.to(device), start_dim=1)])
                d_out = self.discriminator(d_inp)
                target = torch.cat([torch.zeros((batch_size, 1)).to(device), target])
                #calculate loss and do backpropogation
                d_loss = F.binary_cross_entropy_with_logits(d_out, target)
                discriminator_optim.zero_grad()
                d_loss.backward()
                discriminator_optim.step()
                #store some info and update tqdm
                g_totals.append(g_loss.item())
                d_totals.append(d_loss.item())
                t.set_postfix(
                    g_loss='{:.3e}'.format(g_loss.item()),
                    d_loss='{:.3e}'.format(d_loss.item()),
                )
            print('\tg_loss: {:.3e}'.format(sum(g_totals)/len(g_totals)))
            print('\td_loss: {:.3e}'.format(sum(d_totals)/len(d_totals)))

    def save(self):
        self.to(torch.device('cpu'))
        torch.save(self.generator.state_dict(), 'weights/generator.pt')
        torch.save(self.discriminator.state_dict(), 'weights/discriminator.pt')

    def load(self, weight_path):
        self.generator.load_state_dict(torch.load(os.path.join(weight_path, 'generator.pt')))
        self.discriminator.load_state_dict(torch.load(os.path.join(weight_path, 'discriminator.pt')))
