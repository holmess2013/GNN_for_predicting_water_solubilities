
# Create Bemis–Murcko scaffold-based train/val/test indices from mol_with_sol.pkl

import joblib, pickle, numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# ===== CONFIG =====
INPUT_PKL  = "mol_with_sol.pkl"   # [(mol, y), ...] or [{'mol': Mol, 'y': float}, ...]
VAL_FRAC   = 0.10                 # 10% validation
TEST_FRAC  = 0.10                 # 10% test
INCLUDE_CHIRALITY = False         # scaffold SMILES with/without chirality
# ===================

def load_mols(path):
    """Return list of RDKit Mol in the canonical order used across the project."""
    pairs = joblib.load(path)
    mols = []
    for item in pairs:
        if isinstance(item, tuple) and len(item) == 2:
            mol = item[0]
        elif isinstance(item, dict):
            mol = item.get("mol")
        else:
            raise ValueError("Unsupported pickle format: use (mol, y) tuples or dicts with keys 'mol' and 'y'.")
        if mol is None:
            raise ValueError("Found None Mol; check your input pickle.")
        mols.append(mol)
    return mols

def scaffold_smiles(mol, include_chirality=False):
    core = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(core, isomericSmiles=include_chirality)

def scaffold_split_indices(mols, val_frac=0.10, test_frac=0.10, include_chirality=False):
    """Allocate *whole* scaffold buckets to test -> val -> train. Returns integer arrays."""
    buckets = defaultdict(list)  # scaffold_smiles -> [indices]
    for i, m in enumerate(mols):
        try:
            s = scaffold_smiles(m, include_chirality)
            if not s:  # empty scaffold (acyclic molecule) -> fall back to full SMILES core
                s = f"EMPTY_SCAF_{i}"
        except Exception:
            s = f"NOSCAF_{i}"
        buckets[s].append(i)

    # Largest scaffold groups first for stability
    groups = sorted(buckets.values(), key=len, reverse=True)

    n = len(mols)
    n_test = int(round(test_frac * n))
    n_val  = int(round(val_frac  * n))

    test, val, train = [], [], []
    for g in groups:
        if len(test) < n_test:
            test.extend(g)
        elif len(val) < n_val:
            val.extend(g)
        else:
            train.extend(g)

    # Sanity checks
    assert set(train).isdisjoint(test)
    assert set(train).isdisjoint(val)
    assert set(test).isdisjoint(val)

    return np.array(train, dtype=int), np.array(val, dtype=int), np.array(test, dtype=int), buckets

def main():
    mols = load_mols(INPUT_PKL)
    train_idx, val_idx, test_idx, buckets = scaffold_split_indices(
        mols, VAL_FRAC, TEST_FRAC, INCLUDE_CHIRALITY
    )

    # Save for reuse everywhere (GNN train/eval + baselines)
    with open("train_idx.pkl", "wb") as f: pickle.dump(train_idx, f)
    with open("val_idx.pkl",   "wb") as f: pickle.dump(val_idx,   f)
    with open("test_idx.pkl",  "wb") as f: pickle.dump(test_idx,  f)

    # Report
    n_scaffolds = len(buckets)
    print(f"Total molecules: {len(mols)} | Unique scaffolds: {n_scaffolds}")
    print(f"Split sizes -> train: {len(train_idx)} | val: {len(val_idx)} | test: {len(test_idx)}")
    # Optional: show top-5 largest scaffold buckets
    sizes = sorted((len(v) for v in buckets.values()), reverse=True)[:5]
    print(f"Largest scaffold bucket sizes (top-5): {sizes}")

if __name__ == "__main__":
    main()
