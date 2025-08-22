# Step6_baseline_random_forest.py  (faster, shallower trees)

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np, joblib, pickle, time
from pathlib import Path

# --- Load data in the SAME order used to create BM indices ---  # :contentReference[oaicite:2]{index=2}
pairs = joblib.load("mol_with_sol.pkl")
mols, y = [], []
for item in pairs:
    if isinstance(item, tuple) and len(item) == 2:
        mol, label = item
    elif isinstance(item, dict):
        mol, label = item.get("mol"), item.get("y")
    else:
        raise ValueError("Unexpected item format")
    if mol is None:
        continue
    mols.append(mol)
    y.append(float(label))
y = np.asarray(y, dtype=float)

# --- Load BM scaffold indices ---  # :contentReference[oaicite:3]{index=3}
with open("train_idx.pkl", "rb") as f: train_idx = pickle.load(f)
with open("val_idx.pkl",   "rb") as f: val_idx   = pickle.load(f)
with open("test_idx.pkl",  "rb") as f: test_idx  = pickle.load(f)

N = len(mols)
assert max(train_idx) < N and max(val_idx) < N and max(test_idx) < N

# --- Morgan fingerprints (ECFP) ---
radius, nBits = 2, 2048
def morgan_bits(m):
    bv = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

X = np.vstack([morgan_bits(m) for m in mols]).astype(np.float32)

# --- Slice by BM indices ---
X_train, y_train = X[train_idx], y[train_idx]
X_val,   y_val   = X[val_idx],   y[val_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

# --- Faster Random Forest config ---
rf = RandomForestRegressor(
    n_estimators=200,        # ↓ from 500
    max_depth=20,            # cap depth (speeds up a LOT)
    max_features="sqrt",     # subsample features at each split
    min_samples_leaf=2,      # small regularization + fewer leaves
    bootstrap=True,
    oob_score=False,         # set True if you want OOB (slower)
    n_jobs=-1,
    random_state=0,
)

t0 = time.time()
rf.fit(X_train, y_train)
t1 = time.time()

# --- Validate ---
y_val_pred = rf.predict(X_val)
val_mae = mean_absolute_error(y_val, y_val_pred)
val_r2  = r2_score(y_val, y_val_pred)
print(f"Validation -> MAE: {val_mae:.4f} | R2: {val_r2:.4f} | Train time: {t1 - t0:.1f}s")

# --- Retrain on train+val for final test ---
X_trv = np.vstack([X_train, X_val])
y_trv = np.concatenate([y_train, y_val])

rf_final = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    max_features="sqrt",
    min_samples_leaf=2,
    bootstrap=True,
    n_jobs=-1,
    random_state=0,
)
t2 = time.time()
rf_final.fit(X_trv, y_trv)
t3 = time.time()

y_test_pred = rf_final.predict(X_test)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2  = r2_score(y_test, y_test_pred)
rmse = float(np.sqrt(np.mean((y_test_pred - y_test)**2)))
print(f"✅ Test -> RMSE: {rmse:.4f} | MAE: {test_mae:.4f} | R2: {test_r2:.4f} | Train+Val time: {t3 - t2:.1f}s")

# --- Save predictions ---
import pandas as pd
out = pd.DataFrame({"y_true": y_test, "y_pred": y_test_pred})
out_path = Path("rf_test_predictions_shallow.csv")
out.to_csv(out_path, index=False)
print(f"Saved RF predictions to: {out_path.resolve()}")

