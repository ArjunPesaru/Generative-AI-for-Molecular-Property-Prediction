import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("molecular_dataset.csv")
tokenizer = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

# Convert SMILES to tokenized sequences
def tokenize_smiles(smiles_list):
    return tokenizer(smiles_list, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

# Define ChemBERTa Model
class MolecularPropertyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.chemberta = RobertaModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        self.fc = nn.Linear(768, 3)  # Predicts LogP, Molecular Weight, Solubility

    def forward(self, input_ids, attention_mask):
        outputs = self.chemberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.fc(pooled_output)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MolecularPropertyPredictor().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Prepare data
smiles_list = df["SMILES"].tolist()
properties = df[["LogP", "Molecular_Weight", "Solubility"]].values

tokens = tokenize_smiles(smiles_list)
input_ids, attention_mask = tokens["input_ids"].to(device), tokens["attention_mask"].to(device)
labels = torch.tensor(properties, dtype=torch.float32).to(device)

if __name__ == "__main__":
    # Train Model
    epochs = 10
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Save Model
    torch.save(model.state_dict(), "chemberta_model.pth")
    print("✅ Model saved as chemberta_model.pth")

