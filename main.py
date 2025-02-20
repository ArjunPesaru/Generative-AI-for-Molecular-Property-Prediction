import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors
import selfies as sf

# Fetch ChEMBL data with error handling
def fetch_chembl_data(limit=1000):
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?limit={limit}"
    response = requests.get(url)

    # Validate response
    if response.status_code != 200:
        print(f"❌ Error fetching ChEMBL data: {response.status_code}")
        return []

    data = response.json()

    # Ensure response contains 'molecules' key
    if "molecules" not in data or not isinstance(data["molecules"], list):
        print("❌ Error: 'molecules' key missing or not a list in API response.")
        return []

    return data["molecules"]

# Extract molecular properties
def process_molecules(molecule_list):
    data = []
    
    for mol in molecule_list:
        # Skip if mol is None
        if mol is None:
            continue

        # Ensure 'molecule_structures' exists and contains 'canonical_smiles'
        if "molecule_structures" not in mol or mol["molecule_structures"] is None:
            continue  # Skip if no structure info

        smiles = mol["molecule_structures"].get("canonical_smiles", None)
        if not smiles:
            continue  # Skip if SMILES is missing

        # Convert SMILES to RDKit molecule
        mol_obj = Chem.MolFromSmiles(smiles)
        if not mol_obj:
            continue  # Skip if molecule conversion fails

        # Compute molecular properties
        mol_weight = Descriptors.MolWt(mol_obj)
        logp = Descriptors.MolLogP(mol_obj)
        solubility = Descriptors.TPSA(mol_obj)  # Approximate solubility measure

        # Convert SMILES to SELFIES (More stable than SMILES)
        try:
            selfies = sf.encoder(smiles)
        except:
            selfies = None  # Skip molecule if SELFIES conversion fails

        if selfies:
            data.append([smiles, selfies, mol_weight, logp, solubility])

    return data

# Save to CSV
def save_to_csv(data, filename="molecular_dataset.csv"):
    df = pd.DataFrame(data, columns=["SMILES", "SELFIES", "Molecular_Weight", "LogP", "Solubility"])
    df.to_csv(filename, index=False)
    print(f"✅ Dataset saved as {filename} ({len(df)} molecules)")

# Main execution
print("🔍 Fetching molecular data from ChEMBL...")
molecules = fetch_chembl_data(limit=2000)

if molecules:
    print(f"📦 Fetched {len(molecules)} molecules. Processing...")
    processed_data = process_molecules(molecules)

    if processed_data:
        save_to_csv(processed_data)
        print("✅ Dataset successfully created!")
    else:
        print("⚠️ No valid molecules were processed.")
else:
    print("⚠️ Failed to fetch molecules from ChEMBL.")
