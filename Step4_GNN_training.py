# === Imports ===
import numpy as np
import joblib, pickle
from pathlib import Path

from sklearn.preprocessing import StandardScaler

from spektral.data import Dataset, Graph
from spektral.data.loaders import DisjointLoader
from spektral.layers import GCNConv, GlobalSumPool

import tensorflow as tf
keras = tf.keras
layers = keras.layers

# === Load graphs (same order as mol_with_sol.pkl) ===
graphs = joblib.load("molecule_graphs.pkl")  # list of spektral.Graph objects

# === Load BM-scaffold split indices ===
with open("train_idx.pkl", "rb") as f: train_idx = pickle.load(f)
with open("val_idx.pkl",   "rb") as f: val_idx   = pickle.load(f)
with open("test_idx.pkl",  "rb") as f: test_idx  = pickle.load(f)

# === Quick sanity checks ===
N = len(graphs)
assert max(train_idx) < N and max(val_idx) < N and max(test_idx) < N
assert len(set(train_idx) & set(val_idx)) == 0
assert len(set(train_idx) & set(test_idx)) == 0
assert len(set(val_idx) & set(test_idx)) == 0

# === Dataset wrappers ===
class ListDataset(Dataset):
    def __init__(self, graphs, **kwargs):
        self._graphs = graphs
        super().__init__(**kwargs)
    def read(self):
        return self._graphs

# === Model ===
class TwoLayerGCN(keras.Model):
    def __init__(self, hidden=64, dropout=0.0):
        super().__init__()
        self.conv1 = GCNConv(hidden, activation="relu")
        self.conv2 = GCNConv(hidden, activation="relu")
        self.dropout = layers.Dropout(dropout)
        self.pool = GlobalSumPool()
        self.d1   = layers.Dense(hidden, activation="relu")
        self.out  = layers.Dense(1)  # regression output (scaled units)

    def call(self, inputs, training=False):
        x, a, i = inputs                   # node feats, adjacency (sparse), graph ids
        x = self.conv1([x, a])
        x = self.dropout(x, training=training)
        x = self.conv2([x, a])
        g = self.pool([x, i])              # one vector per graph in the batch
        g = self.d1(g)
        return self.out(g)

def main():
    # === Slice graphs by BM-scaffold indices ===
    train_data = [graphs[i] for i in train_idx]
    val_data   = [graphs[i] for i in val_idx]
    test_data  = [graphs[i] for i in test_idx]

    # === Scale labels (fit on train only) ===
    # Expect each g.y to be shape (1,) floats
    y_train = np.vstack([g.y for g in train_data]).astype(np.float32)
    y_val   = np.vstack([g.y for g in val_data]).astype(np.float32)
    y_test  = np.vstack([g.y for g in test_data]).astype(np.float32)

    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train)
    y_val_scaled   = scaler.transform(y_val)
    y_test_scaled  = scaler.transform(y_test)

    for g, y in zip(train_data, y_train_scaled): g.y = y
    for g, y in zip(val_data,   y_val_scaled):   g.y = y
    for g, y in zip(test_data,  y_test_scaled):  g.y = y

    # === Wrap splits into Datasets (Spektral loaders expect a Dataset) ===
    train_ds = ListDataset(train_data)
    val_ds   = ListDataset(val_data)
    test_ds  = ListDataset(test_data)

    # === Build DisjointLoaders ===
    batch_size = 32
    train_loader = DisjointLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DisjointLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DisjointLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # Keep generator references alive (Windows tf.data finalization quirk)
    train_tf = train_loader.load()
    val_tf   = val_loader.load()
    test_tf  = test_loader.load()

    # === Model, compile ===
    model = TwoLayerGCN()  # defaults: hidden=64, dropout=0.0
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )

    # === Callbacks ===
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]

    # === Train ===
    history = model.fit(
        train_tf,
        steps_per_epoch = train_loader.steps_per_epoch,
        validation_data = val_tf,
        validation_steps = val_loader.steps_per_epoch,
        epochs=50,
        callbacks=callbacks,
        verbose=1,
        workers=0,
        use_multiprocessing=False,
    )

    # === Save weights + scaler for later evaluation ===
    out_dir = Path(".")
    weights_path = out_dir / "gcn_solubility.weights.h5"
    scaler_path  = out_dir / "label_scaler.pkl"

    model.save_weights(str(weights_path))
    joblib.dump(scaler, str(scaler_path))

    print(f"\nSaved model weights to: {weights_path.resolve()}")
    print(f"Saved label scaler to:   {scaler_path.resolve()}")

if __name__ == "__main__":
    main()
    tf.keras.backend.clear_session()
