# Solubility Prediction with GNNs (Bemis–Murcko Scaffold Split)

This project predicts aqueous solubility of small molecules using deep learning (TensorFlow/Spektral) and compares against a strong classical baseline (Morgan fingerprints + Random Forest). The dataset is **AqSolDB** (public, curated). Splits use **Bemis–Murcko scaffolds** so the test set contains **unseen chemotypes**—no leakage from random splits.

## TL;DR Results (MAE, original units)
| Model | MAE |
|---|---:|
| GNN (2-layer GCN) | **1.48** |
| Random Forest (ECFP4, 2048 bits) | 1.51 |

## Pipeline (6 scripts)
1. **Step 1** – Convert AqSolDB SMILES to RDKit Mol objects and save `mol_with_sol.pkl`.  
2. **Step 2** – Compute Bemis–Murcko scaffolds, bucket molecules, and save **train/val/test** indices.  
3. **Step 3** – Convert RDKit molecules into Spektral `Graph` objects (`molecule_graphs.pkl`).  
4. **Step 4** – Train the GNN (labels standardized on **train** only).  
5. **Step 5** – Evaluate the GNN on the **scaffold test** set; inverse-transform to original units.  
6. **Step 6** – Baseline: Morgan fingerprints + Random Forest on the same scaffold split.

## How to Run
Place the AqSolDB CSV files in the repo root, then:
```bash
python Step1_convert_dataset_to_rdkit_objects.py
python Step2_save_BM_scaffold_indices.py
python Step3_convert_rdkit_objects_to_graphs.py
python Step4_GNN_training.py
python Step5_GNN_evaluation.py
python Step6_baseline_random_forest.py
