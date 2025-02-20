import torch
import numpy as np
import selfies as sf
from train_vae import VAE, max_length, char_to_index, index_to_char, latent_dim, device
from rdkit import Chem

# Load trained VAE model
vae = VAE(max_length, latent_dim).to(device)
vae.load_state_dict(torch.load("vae_model.pth", map_location=device))
vae.eval()

MAX_TRIES = 10

def generate_molecule(attempt=0):
    if attempt >= MAX_TRIES:
        print("❌ Failed to generate a valid molecule.")
        return None  # Stop infinite recursion

    with torch.no_grad():
        # Sample a new latent vector
        z = torch.randn(1, latent_dim).to(device)
        generated_seq = vae.decoder(z).cpu().numpy().reshape(max_length, -1)

        # Convert probabilities to character indices
        generated_indices = np.argmax(generated_seq, axis=1)
        generated_selfies = "".join(index_to_char.get(i, "?") for i in generated_indices)

        # Convert SELFIES to SMILES
        try:
            generated_smiles = sf.decoder(generated_selfies)
        except:
            print(f"⚠️ Invalid SELFIES: {generated_selfies}, retrying... ({attempt+1}/{MAX_TRIES})")
            return generate_molecule(attempt + 1)

        # Validate SMILES
        mol = Chem.MolFromSmiles(generated_smiles)
        if mol and len(generated_smiles) > 3:  # Ensure it's a valid structure
            return generated_smiles
        else:
            print(f"⚠️ Invalid SMILES generated: {generated_smiles}, retrying... ({attempt+1}/{MAX_TRIES})")
            return generate_molecule(attempt + 1)

# Test molecule generation
if __name__ == "__main__":
    generated_smiles = generate_molecule()
    if generated_smiles:
        print("✅ Generated SMILES:", generated_smiles)
