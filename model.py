import torch
import torch.nn.functional as F
from torch import nn
from torch import distributions

class Encoder(nn.Module):
    def __init__(self, input_size, H, latent_size):
        super(Encoder, self).__init__()
        self.linear_1 = nn.Linear(input_size, H)
        self.linear_2 = nn.Linear(H, H) # to gain additional information
        self.enc_mean = nn.Linear(H, latent_size)
        self.enc_log_sigma = nn.Linear(H, latent_size) # log_sigma ensure positive values

    def forward(self, x):
        x = F.tanh(self.linear_1(x))
        x = F.tanh(self.linear_2(x))
        mu = self.enc_mean(x)
        log_sigma = self.enc_log_sigma(x)
        sigma = torch.exp(log_sigma)
        return distributions.Normal(loc=mu, scale=sigma)

class Decoder(nn.Module):
    def __init__(self, input_size, H, output_size):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(input_size, H)
        self.dec = nn.Linear(H, output_size)

    def forward(self, x):
        x = F.tanh(self.linear(x))
        mu = F.tanh(self.dec(x)) # narrow data range
        return distributions.Normal(loc=mu, scale=torch.ones_like(mu)) # simplify training

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        q_z = self.encoder(x)
        z = q_z.rsample()
        p_x = self.decoder(z)
        return p_x, q_z # return new_x, and approximated z