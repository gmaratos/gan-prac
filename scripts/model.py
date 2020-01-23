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
        self.name = 'simpleD'
        self.net = nn.Sequential(
            nn.Linear(input_size, 2*input_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2*input_size, 1), #discriminator always outputs 1
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1) #flatten input to a vector
        x = self.net(x)
        return x

class SimpleGenerator(nn.Module):
    """a simple generator fully connected one hidden layer"""
    def __init__(self, input_size: int, output_size: int):
        super(SimpleGenerator, self).__init__()
        self.name = 'simpleG'
        self.net = nn.Sequential(
            nn.Linear(input_size, 2*output_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2*output_size, output_size),
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
        self.generator = SimpleGenerator(g_inp, g_out)
        self.discriminator = SimpleDiscriminator(d_inp)
        #build path for saving/loading
        self.prefix = os.path.join(
            'weights',
            '{}-{}'.format(self.generator.name, self.discriminator.name),
        )

    def forward(self, x):
        pass

    def build_generator_input(self, batch_size, device): #no need to check batch size, already done
        """helper function for training and inference"""
        return torch.normal(
            0, 1, (batch_size, self.g_inp_size),
            device=device
        )

    def fit(self, dataset, batch_size: int, epochs: int):
        """training procedure for gan"""
        #build dataloader
        data = TU.DataLoader(dataset, batch_size=batch_size)
        #set up optimizers
        g_optim = TO.SGD(self.generator.parameters(), lr=1e-3)
        d_optim = TO.SGD(self.discriminator.parameters(), lr=1e-3)
        #get device
        device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device_str)
        self.to(device)
        #begin training
        for i in range(1, epochs+1):
            #things for beging of epoch (bookeeping)
            print('Epoch {}'.format(i))
            t = tqdm.tqdm(data)
            g_totals, d_totals = [], []
            for inp, _ in t:
                #build inp/target building blocks
                ones = torch.ones(
                    (batch_size, 1),
                    device=device
                )
                zeros = torch.zeros(
                    (batch_size, 1),
                    device=device
                )
                inp = inp.to(device) #this needs to be done for a concat later
                #to train generator, target should be ones (fool discriminator)
                noise = self.build_generator_input(batch_size, device)
                g_out = self.generator(noise)
                d_out = self.discriminator(g_out)
                #calculate generator loss and do backpropogation
                g_optim.zero_grad()
                g_loss = F.binary_cross_entropy_with_logits(d_out, ones)
                g_loss.backward()
                g_optim.step()
                #next train discriminator
                noise = self.build_generator_input(batch_size, device)
                g_out = self.generator(noise)
                d_inp = torch.cat([g_out, torch.flatten(inp, start_dim=1)])
                target = torch.cat([zeros, ones])
                d_out = self.discriminator(d_inp)
                #calculate loss and do backpropogation
                d_optim.zero_grad()
                d_loss = F.binary_cross_entropy_with_logits(d_out, target)
                d_loss.backward()
                d_optim.step()
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
        if not os.path.exists(self.prefix):
            os.mkdir(self.prefix)
        self.to(torch.device('cpu'))
        torch.save(
            self.generator.state_dict(),
            os.path.join(self.prefix, 'generator.pt')
        )
        torch.save(
            self.discriminator.state_dict(),
            os.path.join(self.prefix, 'discriminator.pt')
        )

    def load(self):
        self.generator.load_state_dict(torch.load(os.path.join(self.prefix, 'generator.pt')))
        self.discriminator.load_state_dict(torch.load(os.path.join(self.prefix, 'discriminator.pt')))
