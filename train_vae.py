import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import selfies as sf
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset
df = pd.read_csv("molecular_dataset.csv")
selfies_list = df["SELFIES"].tolist()

# Tokenization: Convert SELFIES to integer sequences
unique_chars = sorted(set("".join(selfies_list)))
char_to_index = {char: i for i, char in enumerate(unique_chars)}
index_to_char = {i: char for char, i in char_to_index.items()}
max_length = max(len(s) for s in selfies_list)
vocab_size = len(unique_chars)

def selfies_to_seq(selfies):
    seq = [char_to_index[char] for char in selfies]
    return seq + [0] * (max_length - len(seq))  # Padding

# Prepare dataset
sequences = [selfies_to_seq(s) for s in selfies_list]
train_data, _ = train_test_split(sequences, test_size=0.2, random_state=42)
train_tensor = torch.tensor(train_data, dtype=torch.float32)

# Define VAE Model
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Softmax(dim=-1)
        )
        self.latent_dim = latent_dim

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu, log_var = encoded[:, :self.latent_dim], encoded[:, self.latent_dim:]
        z = self.reparametrize(mu, log_var)
        return self.decoder(z), mu, log_var

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = max_length
latent_dim = 128
vae = VAE(input_dim, latent_dim).to(device)
optimizer = optim.Adam(vae.parameters(), lr=0.0005)
criterion = nn.MSELoss()

if __name__ == "__main__":
    # Train VAE
    epochs = 10000
    for epoch in range(epochs):
        optimizer.zero_grad()
        inputs = train_tensor.to(device)
        outputs, mu, log_var = vae(inputs)
        loss = criterion(outputs, inputs) + 0.5 * torch.sum(torch.exp(log_var) - log_var - 1 + mu.pow(2))
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Save model
    torch.save(vae.state_dict(), "vae_model.pth")
    print("✅ VAE Model saved as vae_model.pth")

