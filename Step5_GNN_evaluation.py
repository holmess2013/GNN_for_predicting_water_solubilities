# === eval_gcn.py ===
import numpy as np
import joblib, pickle
from pathlib import Path

from spektral.data import Dataset
from spektral.data.loaders import DisjointLoader
from spektral.layers import GCNConv, GlobalSumPool

import tensorflow as tf
from tensorflow import keras
from keras import layers

# --- Model must match training script exactly ---
class TwoLayerGCN(keras.Model):
    def __init__(self, hidden=64, dropout=0.0):
        super().__init__()
        self.conv1 = GCNConv(hidden, activation="relu")
        self.conv2 = GCNConv(hidden, activation="relu")
        self.dropout = layers.Dropout(dropout)
        self.pool = GlobalSumPool()
        self.d1   = layers.Dense(hidden, activation="relu")
        self.out  = layers.Dense(1)

    def call(self, inputs, training=False):
        x, a, i = inputs
        x = self.conv1([x, a])
        x = self.dropout(x, training=training)
        x = self.conv2([x, a])
        g = self.pool([x, i])
        g = self.d1(g)
        return self.out(g)

# --- Dataset helper ---
class ListDataset(Dataset):
    def __init__(self, graphs, **kwargs):
        self._graphs = graphs
        super().__init__(**kwargs)
    def read(self):
        return self._graphs

def main():
    print(">>> Starting evaluation...")

    # === Paths ===
    weights_path = Path("gcn_solubility.weights.h5")
    scaler_path  = Path("label_scaler.pkl")
    graphs_path  = Path("molecule_graphs.pkl")

    # Sanity checks
    for p in (weights_path, scaler_path, graphs_path):
        status = "OK" if p.exists() else "MISSING"
        print(f"Checking {p} -> {status}")
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    # --- Load artifacts ---
    print("Loading scaler and graphs...")
    scaler = joblib.load(scaler_path)                 # fitted on TRAIN only during training
    graphs = joblib.load(graphs_path)                 # y values in ORIGINAL units

    # --- Load BM-scaffold TEST indices ---
    with open("test_idx.pkl", "rb") as f: test_idx = pickle.load(f)

    # Bounds/sanity
    N = len(graphs)
    assert len(test_idx) > 0, "Empty test index set."
    assert max(test_idx) < N, "Test indices out of range."

    # --- Build test split from saved indices ---
    test_data = [graphs[i] for i in test_idx]
    print(f"Test size: {len(test_data)}")

    # --- Build test loader ---
    print("Building test loader...")
    test_ds = ListDataset(test_data)
    batch_size = 32
    test_loader = DisjointLoader(test_ds, batch_size=batch_size, shuffle=False)
    test_tf = test_loader.load()
    print(f"Loader steps_per_epoch: {test_loader.steps_per_epoch}")

    # --- Rebuild model and load weights ---
    print("Rebuilding model...")
    model = TwoLayerGCN()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])

    # Warm-up to create variables, then reset iterator so evaluation starts from first batch
    print("Warming up model (creating variables)...")
    sample_inputs, _ = next(iter(test_tf))
    _ = model(sample_inputs, training=False)
    test_tf = test_loader.load()

    print("Loading weights...")
    model.load_weights(str(weights_path))
    print("Weights loaded.")

    # --- Predict (scaled space) ---
    print("Predicting on test set...")
    y_pred_scaled = model.predict(
        test_tf,
        steps=test_loader.steps_per_epoch,
        verbose=1,
        workers=0,
        use_multiprocessing=False,
    )

    # --- Ground truth (original units) ---
    y_true = np.vstack([g.y for g in test_data]).astype(np.float32)
    print(f"Shapes -> y_pred_scaled: {y_pred_scaled.shape}, y_true: {y_true.shape}")

    # --- Inverse-transform predictions back to original units ---
    y_pred = scaler.inverse_transform(y_pred_scaled)

    # --- Metrics in original units ---
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae  = float(np.mean(np.abs(y_pred - y_true)))
    print(f"\n✅ Test RMSE: {rmse:.6f} | Test MAE: {mae:.6f} (original units)")

    # --- (Optional) save a CSV of predictions vs truth ---
    try:
        import pandas as pd
        out = pd.DataFrame({
            "y_true": y_true.ravel(),
            "y_pred": y_pred.ravel(),
            "y_pred_scaled": y_pred_scaled.ravel(),
        })
        out_path = Path("test_predictions.csv")
        out.to_csv(out_path, index=False)
        print(f"Saved predictions to: {out_path.resolve()}")
    except Exception as e:
        print(f"(Skipping CSV save) {e}")

if __name__ == "__main__":
    main()
    tf.keras.backend.clear_session()
