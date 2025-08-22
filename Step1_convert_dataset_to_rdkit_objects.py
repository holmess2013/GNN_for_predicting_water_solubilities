import pandas as pd
import os
from rdkit import Chem
import joblib

# Our first task is to concatenate all of these cvs files into one giant pandas dataframe.

# Going to start using list comprehension more. Less code to write. 

# Let's make a list of dataframes, where each element is one csv as a datframe.
csvs = [pd.read_csv(file) for file in os.listdir() if file.endswith("csv")]

# Now let's concatenate them:
df = pd.concat(csvs, ignore_index=True)

# Let's drop the author's predictions who curated the dataset. They trained a ML model
# using non-specific descriptors instead of graphs. Let's beat those predictions!
df = df.drop(columns = ["Prediction"])

# RDKit carries out its functions on molecule objects, so lets convert each SMILE to a molecule,
# and if any have issues, don't include them. 

print(df.head(10))

mols = []
mol_indices = []

for id, row in df.iterrows():
    smiles = row['SMILES']
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        mols.append(mol)
        mol_indices.append(id)

mol_with_sol = [(m, df.loc[j, 'Solubility']) for m, j in zip(mols, mol_indices)]

print("First 10 SMILES in mol_with_sol:")
for i, (mol, sol) in enumerate(mol_with_sol[:10]):
    print(f"{i}: {Chem.MolToSmiles(mol)} | Solubility: {sol}")

joblib.dump(mol_with_sol, "mol_with_sol.pkl")









