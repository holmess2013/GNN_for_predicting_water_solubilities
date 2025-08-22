import joblib
import numpy as np
from rdkit import Chem
from spektral.data import Graph
from spektral.data import Dataset

# Load in your list of rdkit molecule objects and solubilities

mol_with_sol = joblib.load("mol_with_sol.pkl")

def hybridization_to_int(hybridization):
    mapping = {
        Chem.rdchem.HybridizationType.SP: 1,
        Chem.rdchem.HybridizationType.SP2: 2,
        Chem.rdchem.HybridizationType.SP3: 3,
        Chem.rdchem.HybridizationType.SP3D: 4,
        Chem.rdchem.HybridizationType.SP3D2: 5,
        Chem.rdchem.HybridizationType.UNSPECIFIED: 0,
    }
    return mapping.get(hybridization, -1)


def atom_features(atom):
    return np.array([
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        hybridization_to_int(atom.GetHybridization()),
        atom.GetIsAromatic()
    ], dtype=np.float32)


def mol_to_graph(mol, label=None):
    # Calculate number of atoms in your molecule
    N = mol.GetNumAtoms()

    # Node feature matrix
    x = np.array([atom_features(atom) for atom in mol.GetAtoms()])

    # Adjacency matrix
    A = np.zeros((N,N), dtype=np.float32)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        A[i,j] = 1
        A[j,i] = 1

    # Label (solubility)
    y = np.array([label], dtype=np.float32) if label is not None else None

    # Return a spektral graph object of you molecule
    return Graph(x=x, a=A, y=y)

graph_list = [mol_to_graph(mol, sol) for mol,sol in mol_with_sol]

joblib.dump(graph_list, "molecule_graphs.pkl")