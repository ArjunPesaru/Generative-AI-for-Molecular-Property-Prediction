import torch
import numpy as np
import selfies as sf
from train_vae import VAE, max_length, char_to_index, index_to_char, latent_dim, device
from rdkit import Chem
from rdkit.Chem import Descriptors
from skopt import gp_minimize
from skopt.space import Real

# Load trained VAE model
vae = VAE(max_length, latent_dim).to(device)
vae.load_state_dict(torch.load("vae_model.pth", map_location=device))
vae.eval()

# Target properties for optimization
TARGET_LOGP = 2.0  # Desired LogP value
TARGET_SOLUBILITY = 3.0  # Desired solubility

def get_molecular_properties(smiles):
    """ Compute LogP and Solubility for a given molecule """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        logp = Descriptors.MolLogP(mol)
        solubility = Descriptors.TPSA(mol)
        return logp, solubility
    return None, None

def generate_optimized_molecule(params=None):
    """ Generate molecule and optimize properties using Bayesian Optimization """

    if params is None:
        # Random sampling if no optimization parameters provided
        z = torch.randn(1, latent_dim).to(device)
    else:
        # Use optimized latent space
        z = torch.tensor(params, dtype=torch.float32).to(device).unsqueeze(0)

    with torch.no_grad():
        generated_seq = vae.decoder(z).cpu().numpy().reshape(max_length, -1)

    # Convert probabilities to character indices
    generated_indices = np.argmax(generated_seq, axis=1)
    generated_selfies = "".join(index_to_char.get(i, "?") for i in generated_indices)

    # Convert SELFIES to SMILES
    try:
        generated_smiles = sf.decoder(generated_selfies)
    except:
        return None  # Return None if decoding fails

    mol = Chem.MolFromSmiles(generated_smiles)
    if not mol or len(generated_smiles) < 3:
        return None  # Ensure valid SMILES

    return generated_smiles

# Define search space for optimization
search_space = [Real(-3, 3, name=f"latent_{i}") for i in range(latent_dim)]

def optimize_molecule():
    """ Optimize molecule properties using Bayesian Optimization """
    def objective_function(params):
        generated_smiles = generate_optimized_molecule(params)
        if not generated_smiles:
            return 100  # High penalty for invalid molecules

        logp, solubility = get_molecular_properties(generated_smiles)
        if logp is None or solubility is None:
            return 100  # High penalty for invalid molecules

        # Objective: minimize the difference from target properties
        return abs(logp - TARGET_LOGP) + abs(solubility - TARGET_SOLUBILITY)

    # Run Bayesian Optimization
    res = gp_minimize(objective_function, search_space, n_calls=30, random_state=42)

    # Get best parameters and generate molecule
    best_params = res.x
    best_molecule = generate_optimized_molecule(best_params)

    # Ensure valid molecule is returned
    if not best_molecule:
        print("❌ Failed to find a valid optimized molecule.")
        return None

    return best_molecule

# Test optimization
if __name__ == "__main__":
    optimized_smiles = optimize_molecule()
    if optimized_smiles:
        print("✅ Optimized Molecule:", optimized_smiles)
    else:
        print("❌ No valid molecule found.")
