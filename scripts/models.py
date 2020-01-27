import torch
import torch.nn as nn
from tqdm import tqdm

class GAN(nn.Module):

    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def fit(self, dataset):
        #set hyper parameters
        lr = 1e-4
        epochs = 40
        batch_size = 32
        #get device for training, set model
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        #build data loader
        train_loader = torch.utils.data.DataLoader(dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        #loss function and optimizers
        criterion = nn.BCEWithLogitsLoss()
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=lr)
        #begin training
        for epoch in range(1, epochs+1):
            #construct loop and datastructures for recording info
            g_losses, d_real_losses, d_fake_losses = [], [], []
            t = tqdm(train_loader, leave=False)
            t.set_description(desc='E:{}'.format(epoch))
            #run a pass through the data
            for (images, _) in t:
                #generate targets
                fake_target = torch.zeros((batch_size, 1), device=device)
                real_target = torch.ones((batch_size, 1), device=device)

                #prepare discriminator input
                real_images = images.to(device)
                noise = torch.randn(batch_size, 100, device=device)
                fake_images = self.generator(noise)
                #get loss on the fake input
                d_fake = self.discriminator(fake_images)
                d_fake_loss = criterion(d_fake, fake_target)
                #get loss on the read input
                d_real = self.discriminator(real_images)
                d_real_loss = criterion(d_real, real_target)
                #train the discriminator
                d_loss = d_fake_loss + d_real_loss
                d_opt.zero_grad()
                d_loss.backward()
                d_opt.step()

                #prepare the generator input
                noise = torch.randn(batch_size, 100, device=device)
                fake_images = self.generator(noise)
                #get loss on fake input
                d_predictions = self.discriminator(fake_images)
                g_loss = criterion(d_predictions, real_target)
                #train the generator
                g_opt.zero_grad()
                g_loss.backward()
                g_opt.step()

                #record losses
                g_losses.append(g_loss.item())
                d_real_losses.append(d_real_loss.item())
                d_fake_losses.append(d_fake_loss.item())

            #compute final losses for the epoch
            avg_g_loss = sum(g_losses)/len(g_losses)
            avg_d_real_loss = sum(d_real_losses)/len(d_real_losses)
            avg_d_fake_loss = sum(d_fake_losses)/len(d_fake_losses)
            #create msg for log and stdout
            output_msg = 'E {} ~ '.format(epoch)
            output_msg += 'g {:.3e} ~ '.format(avg_g_loss)
            output_msg += 'd_r {:.3e} ~ '.format(avg_d_real_loss)
            output_msg += 'd_f {:.3e} ~ '.format(avg_d_fake_loss)
            print(output_msg)

# Our Discriminator class
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        #first make sure input is correct size
        x = x.view(x.shape[0], 784)
        #forward pass
        x = self.model(x)
        return x

# Our Generator class
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, x):
        #first make sure input is correct size
        x = x.view(x.shape[0], 100)
        #forward pass
        x = self.model(x)
        return x
