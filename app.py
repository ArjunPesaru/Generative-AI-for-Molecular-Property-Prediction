import streamlit as st
import torch
from generate_molecule import generate_molecule
from optimize_molecule import optimize_molecule
from train_chemberta import MolecularPropertyPredictor, tokenizer
from train_vae import VAE, max_length, latent_dim, device
from rdkit import Chem
from rdkit.Chem import Draw

# Load trained ChemBERTa model for property prediction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.write("🔄 Loading ChemBERTa model...")
try:
    chemberta_model = MolecularPropertyPredictor().to(device)
    chemberta_model.load_state_dict(torch.load("chemberta_model.pth", map_location=device))
    chemberta_model.eval()
    st.write("✅ ChemBERTa model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading ChemBERTa model: {e}")

# Load trained VAE model for molecule generation
st.write("🔄 Loading VAE model...")
try:
    vae_model = VAE(max_length, latent_dim).to(device)
    vae_model.load_state_dict(torch.load("vae_model.pth", map_location=device))
    vae_model.eval()
    st.write("✅ VAE model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading VAE model: {e}")

# Function to predict molecular properties using ChemBERTa
def predict_properties(smiles):
    try:
        encoding = tokenizer(smiles, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        input_ids, attention_mask = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

        with torch.no_grad():
            preds = chemberta_model(input_ids, attention_mask).cpu().numpy().flatten()

        return {"LogP": preds[0], "Molecular Weight": preds[1], "Solubility": preds[2]}
    except Exception as e:
        st.error(f"❌ Error in prediction: {e}")
        return None

# Streamlit UI
st.title("🔬 AI for Molecular Property Prediction & Novel Molecule Generation")

# 🔍 **Predict Molecular Properties**
st.header("🔬 Predict Molecular Properties")
smiles_input = st.text_input("Enter a SMILES string:")
if st.button("Predict"):
    if smiles_input:
        properties = predict_properties(smiles_input)
        if properties:
            st.write(properties)
            mol = Chem.MolFromSmiles(smiles_input)
            if mol:
                st.image(Draw.MolToImage(mol), caption="Molecular Structure")

# 🧪 **Generate a Novel Molecule using VAE**
st.header("🧪 Generate a Novel Molecule")
if st.button("Generate New Molecule"):
    try:
        generated_smiles = generate_molecule()
        if generated_smiles:
            st.write(f"**Generated SMILES:** {generated_smiles}")
            mol = Chem.MolFromSmiles(generated_smiles)
            if mol:
                st.image(Draw.MolToImage(mol), caption="Generated Molecular Structure")
        else:
            st.warning("⚠️ No valid molecule generated. Try again.")
    except Exception as e:
        st.error(f"❌ Error generating molecule: {e}")

# 🎯 **Optimize a Molecule**
st.header("🎯 Optimize a Molecule for Drug-like Properties")
if st.button("Optimize Molecule"):
    try:
        optimized_smiles = optimize_molecule()
        if optimized_smiles:
            st.write(f"**Optimized SMILES:** {optimized_smiles}")
            mol = Chem.MolFromSmiles(optimized_smiles)
            if mol:
                st.image(Draw.MolToImage(mol), caption="Optimized Molecular Structure")
        else:
            st.warning("⚠️ Optimization failed. Try again.")
    except Exception as e:
        st.error(f"❌ Error optimizing molecule: {e}")
