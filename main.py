import torch
import torchvision.datasets as dsets
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

#check if gpu avalable:
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def discrete_loss_function(recon_x, x, qy):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum') / x.shape[0]

    log_ratio = torch.log(qy * qy.size(-1) + 1e-20)
    KLD = torch.sum(qy * log_ratio, dim=-1).mean()

    return BCE + KLD
              
                
def reparametrization_trick(mu, log_var):
    # Using reparameterization trick to sample from a gaussian
    eps = torch.randn_like(log_var)
    return mu + torch.exp(log_var / 2) * eps
           
            


def sample_gumbel(shape, eps=1e-20):
    #Sample from Gumbel(0, 1)
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))

def gumbel_softmax_sample(logits, tau=1, eps=1e-20):    
    dims = len(logits.size())
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    logits = logits
    logits = torch.tensor(logits).to(device)
    gumbel_noise = torch.tensor(gumbel_noise).to(device)
    y = logits + gumbel_noise
    return F.softmax(y / tau, dim=-1)

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    
    bs, N,K = logits.size()    
    y_soft = gumbel_softmax_sample(logits.view(bs*N,K), tau=tau, eps=eps)

    if hard:
        k = torch.argmax(y_soft, dim=-1)
        y_hard = F.one_hot(k, num_classes=K)

        # 1. makes the output value exactly one-hot
        # 2.makes the gradient equal to y_soft gradient
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
        
    return y.reshape(bs, N*K)


def show(image):
    ''' plot image'''
    fig = plt.figure
    plt.imshow(image, cmap='gray')
    plt.show()


def plot_reconstructed(autoencoder,  r0=(-5, 10), r1=(-10, 5), n=12):
    w = 28
    
    img = []
    for i, z2 in enumerate(np.linspace(r1[1],r1[0], n)):
        for j, z1 in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[z1, z2]]).to(device)
            #print(z)
            x_hat = autoencoder.combine_decoder(z)
            img.append(x_hat)
            
    img = torch.cat(img)
    img = img.to('cpu')
    img = torchvision.utils.make_grid(img,nrow=12).permute(1, 2, 0).detach().numpy()
    plt.imshow(img,extent=[*r0, *r1])






            

               
            
class Combine_VAE(nn.Module):
    ''' gets the numbeer of categories and the number of wanted latent dimentions '''
    def __init__(self,latent_dims,categorical_dim):
        super(Combine_VAE, self).__init__()
        
        
        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 128)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4)
        self.pooling1 = nn.MaxPool2d(kernel_size=4)
        self.to_mean_logvar = nn.Linear(128, 2*latent_dims)
        self.to_gumbel_softmax = nn.Linear(128, latent_dims * categorical_dim)
        
        self.to_decoder = nn.Linear(latent_dims + latent_dims * categorical_dim, 256)
        #self.upscale = nn.ConvTranspose2d(1,1,latent_dims + latent_dims * categorical_dim,kernel_size=2)#in chennels,out chennels 
        #self.upscale = nn.ConvTranspose2d(in_channels=latent_dims + latent_dims * categorical_dim,out_channels=256,kernel_size=2)
        self.linear3 = nn.Linear(256, 512)
        self.linear4 = nn.Linear(512, 784)      
    
        self.N = latent_dims
        self.K = categorical_dim
        self.temp = 1
        self.hard = False
    def image_process(self,x):
        x = F.relu(self.conv1(x))
        x = self.pooling1(x)
        x = F.relu(self.conv2(x))
        
        x = F.relu(self.conv3(x))
        x = self.pooling1(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x
    def continues_encoder(self,x):
        x = self.image_process(x)
        mu, log_var = torch.split(self.to_mean_logvar(x),3, dim=-1)
        self.kl_continues = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        '''  I need to save kl and plot it'''
        return reparametrization_trick(mu, log_var)

    
    def discreate_encoder(self,x,temp,hard):
        x = self.image_process(x)
        x = F.relu(self.to_gumbel_softmax(x))
        q_y = x.view(x.size(0), self.N, self.K)
        q = F.softmax(q_y, dim=-1).reshape(x.size(0)*self.N, self.K)
        log_ratio = torch.log(q * q.size(-1) + 1e-20)
        self.kl_discrete = torch.sum(q * log_ratio, dim=-1).mean()

        return F.gumbel_softmax(q_y, temp, hard).reshape(x.size(0), self.N * self.K)
    
    def combine_decoder(self,z):
        #z = torch.cat((z_continuous, z_discrete), 1) #probebly not the best way
        #print(z.shape)
        z = F.relu(self.to_decoder (z))
        z = F.relu(self.linear3(z))
        z = torch.sigmoid(self.linear4(z))
        return z.reshape((-1, 1, 28, 28))

    def forward(self ,images):
        continues_output = self.continues_encoder(images)
        print('continues_output:',continues_output)
        
        discrete_output = self.discreate_encoder(images, self.temp, self.hard)
        print('discrete_output:',discrete_output)
        # # the outpt of the 2 encoders is combined to the decoder
        # # print(z)
        # # print(z.shape)
        # z_c =  torch.cat((z, c), 1) 
        # #print(z_c.shape)
        # return self.combine_decoder(z_c)         

def Train (model,data,num_epochs, temp=1.0, hard=False):
    temp_min = 0.5
    ANNEAL_RATE = 0.00003
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loss = 0
    kl_continues_list = []
    kl_discrete_list = []
    for epoch in range(num_epochs):
        for batch_idx, (x, _) in enumerate(data):
            optimizer.zero_grad()
            x = x.to(device) #GPU
            x_hat = model(x, temp, hard) # create image with Combine_VAE
            loss = F.binary_cross_entropy(x_hat, x, reduction='sum') + model.kl_continues + model.kl_discrete
            kl_continues_list.append(model.kl_continues)
            kl_discrete_list.append(model.kl_discrete)
            loss.backward()
            train_loss += loss.item() * len(x)
            optimizer.step()
            if batch_idx % 100 == 1:
                temp = np.maximum(temp * np.exp(-ANNEAL_RATE * batch_idx), temp_min)

    x_hat = x_hat.to('cpu').detach().numpy() # transform the tensor to numpy
    show(x_hat[0].reshape(28,28))
    plt.plot(kl_continues_list,label = 'continues')
    #plt.plot(kl_discrete_list[100,:],label = 'discrete ')
    plt.legend()
    plt.grid()
    plt.xlabel("batch number")
        
    return model      

if __name__ == '__main__':
    ''' load data and create dataloader'''
    batch_size = 1
    image_size = 64

    train_dataset = dsets.CelebA(root='/datashare',split = 'train',
                                 transform=transforms.Compose([
                                 transforms.Resize((64,64)),
                                 transforms.CenterCrop(image_size),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),download=False)# make sure you set it to False
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)
    
    '''parameters for VAE model'''
    latent_dims = 3
    categorical_dim = 40
    temp = 1
    hard = False
    ANNEAL_RATE = 0.00003
    model = Combine_VAE(latent_dims, categorical_dim)
    #model = model.to(device)
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        x_hat = model(images)
        # print(labels)
        # print(labels.shape)
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # optimizer.zero_grad()
        #x = x.to(device) #GPU
        #print(x.shape)
        break

