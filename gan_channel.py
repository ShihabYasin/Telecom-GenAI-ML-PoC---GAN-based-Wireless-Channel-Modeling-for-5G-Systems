import torch
import torch.nn as nn
import numpy as np

# Simulate real channel data (path loss + fading)
def generate_channel_data(num_samples, num_subcarriers):
    # Simulate 28 GHz mmWave channel characteristics
    path_loss = 10**(-(128.1 + 37.6*np.log10(100))/10)  # 100m distance
    fading = np.random.rayleigh(scale=0.5, size=(num_samples, num_subcarriers))
    return torch.tensor(path_loss * fading, dtype=torch.float32)

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# GAN Training
def train_gan(generator, discriminator, real_data, epochs=500, batch_size=32, lr=0.0002):
    # Loss and optimizers
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # Train Discriminator
        d_loss_total = 0
        for i in range(0, len(real_data), batch_size):
            # Real data batch
            real_batch = real_data[i:i+batch_size]
            real_labels = torch.ones(real_batch.size(0), 1)
            
            # Fake data batch
            noise = torch.randn(real_batch.size(0), generator.model[0].in_features)
            fake_batch = generator(noise)
            fake_labels = torch.zeros(fake_batch.size(0), 1)
            
            # Discriminator loss
            d_optimizer.zero_grad()
            real_output = discriminator(real_batch)
            fake_output = discriminator(fake_batch.detach())
            d_loss_real = criterion(real_output, real_labels)
            d_loss_fake = criterion(fake_output, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            d_loss_total += d_loss.item()
        
        # Train Generator
        g_loss_total = 0
        for i in range(0, len(real_data), batch_size):
            noise = torch.randn(batch_size, generator.model[0].in_features)
            gen_labels = torch.ones(batch_size, 1)
            
            g_optimizer.zero_grad()
            fake_batch = generator(noise)
            output = discriminator(fake_batch)
            g_loss = criterion(output, gen_labels)
            g_loss.backward()
            g_optimizer.step()
            g_loss_total += g_loss.item()
        
        if (epoch+1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}] | D Loss: {d_loss_total:.4f} | G Loss: {g_loss_total:.4f}')

# Main execution
if __name__ == "__main__":
    # Parameters
    num_samples = 1000
    num_subcarriers = 64  # For 20MHz channel
    latent_dim = 100
    
    # Generate real channel data
    real_data = generate_channel_data(num_samples, num_subcarriers)
    
    # Initialize models
    generator = Generator(latent_dim, num_subcarriers)
    discriminator = Discriminator(num_subcarriers)
    
    # Train GAN
    train_gan(generator, discriminator, real_data)
    
    print("GAN training completed!")
