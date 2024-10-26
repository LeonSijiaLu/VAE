import torch
from config import get_config
from model import Encoder, Decoder, VAE
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import distributions

if torch.cuda.is_available():
    print("running on cuda")
    device = torch.device("cuda")
else:
    print("running on cpu")
    device = torch.device("cpu")

mnist = datasets.MNIST('./', download=True, transform=transforms.Compose([
    # turn images into tensors, also change pixels range from [0, 255] to [0, 1]
    transforms.ToTensor(),

    # x_normalized = (x - mean) / std
    # change pixels range from [0, 1] to [-0.5, 0.5]
    transforms.Normalize(0.5, 1),
]))

config = get_config()
input_size = config['input_size']
hidden_size = config['hidden_size']
latent_size = config['latent_size']
batch_size = config['batch_size']
epochs = config['epochs']
learning_rate = config['learning_rate']

dataloader = DataLoader(dataset=mnist,
                        batch_size=batch_size,
                        shuffle=True,
                        pin_memory=torch.cuda.is_available())

encoder = Encoder(input_size=input_size, H=hidden_size, latent_size=latent_size)
decoder = Decoder(input_size=latent_size, H=hidden_size, output_size=input_size)
vae = VAE(encoder=encoder, decoder=decoder).to(device)

optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for data in dataloader:
        inputs, _ = data
        inputs = inputs.view(-1, input_size).to(device)

        optimizer.zero_grad() # clean grad in every epoch
        p_x, q_z = vae(inputs)

        log_px = p_x.log_prob(inputs).sum(-1).mean()
        kl = distributions.kl_divergence(p=q_z, q=distributions.Normal(0, 1)).sum(-1).mean()

        # maximize ELBO
        # Adam can only finds minimum, however we would like to find the maximum of ELBO
        # ELBO = (log_px - kl)
        # loss = -ELBO = - (log_px - kl)
        loss = -(log_px - kl)
        loss.backward()
        optimizer.step()
        l = loss.item()

    print(epoch, l, log_px.item(), kl.item())
