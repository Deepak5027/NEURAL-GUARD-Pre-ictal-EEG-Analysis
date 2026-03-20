"""
Stage 5 — Patient Adapter + ONNX Export

The novel piece: EEG is highly patient-specific.
Instead of training one global model, we:
  1. Pre-train on all available patients (Stage 4)
  2. Fine-tune only the classifier head on a few labeled hours
     from the NEW patient (few-shot adaptation, ~15 min of data)
  3. Export to ONNX for real-time inference on wearable/mobile

This is the core novelty: subject-adaptive temporal GNN.
"""

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
import numpy as np


# ── 1. Patient-adaptive fine-tuning ──────────────────────────────────────────
def freeze_backbone(model: nn.Module):
    """Freeze all layers except the classification head."""
    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (head only): {trainable:,}")


def unfreeze_backbone(model: nn.Module):
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def patient_adapt(model: nn.Module,
                  patient_graphs: list,
                  epochs: int   = 20,
                  lr:    float  = 5e-5,
                  device: str   = "cpu") -> nn.Module:
    """
    Few-shot fine-tune on patient-specific data.
    Phase 1: freeze backbone, train head  (10 epochs)
    Phase 2: unfreeze all, low-LR full    (10 epochs)
    """
    model = model.to(device)
    loader = DataLoader(patient_graphs, batch_size=8, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    # Phase 1 — head only
    freeze_backbone(model)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr * 10
    )
    for ep in range(epochs // 2):
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = criterion(model(batch), batch.y.squeeze())
            loss.backward()
            opt.step()
        print(f"[Phase 1] Epoch {ep+1}/{epochs//2}  loss={loss.item():.4f}")

    # Phase 2 — full fine-tune at lower LR
    unfreeze_backbone(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(epochs // 2):
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = criterion(model(batch), batch.y.squeeze())
            loss.backward()
            opt.step()
        print(f"[Phase 2] Epoch {ep+1}/{epochs//2}  loss={loss.item():.4f}")

    return model


# ── 2. ONNX export for edge/mobile deployment ─────────────────────────────────
def export_onnx(model: nn.Module,
                n_nodes: int,
                n_time:  int,
                n_edges: int,
                path:    str = "seizure_predictor.onnx"):
    """
    Export model to ONNX.
    Inputs: node_features, edge_index, edge_attr, batch
    Output: logits (B, 2)

    Note: PyG models need careful ONNX handling.
    We wrap the model to accept dense tensors.
    """
    model.eval()

    class ONNXWrapper(nn.Module):
        """Wraps PyG model to accept plain tensors (no Data object)."""
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x, edge_index, edge_attr, batch):
            from torch_geometric.data import Data
            data = Data(x=x, edge_index=edge_index,
                        edge_attr=edge_attr, batch=batch)
            return self.base(data)

    wrapper = ONNXWrapper(model)
    wrapper.eval()

    # dummy inputs
    x          = torch.randn(n_nodes, n_time)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_attr  = torch.rand(n_edges)
    batch      = torch.zeros(n_nodes, dtype=torch.long)

    torch.onnx.export(
        wrapper,
        (x, edge_index, edge_attr, batch),
        path,
        input_names  = ["x", "edge_index", "edge_attr", "batch"],
        output_names = ["logits"],
        opset_version = 17,
        dynamic_axes  = {"x":         {0: "nodes"},
                         "edge_index": {1: "edges"},
                         "edge_attr":  {0: "edges"},
                         "batch":      {0: "nodes"}},
    )
    print(f"ONNX model saved → {path}")


# ── 3. Real-time inference class (runs on device with ONNX Runtime) ───────────
INFERENCE_CODE = '''
# inference.py — runs on wearable/mobile via ONNX Runtime
import onnxruntime as ort
import numpy as np

class SeizurePredictor:
    ALERT_THRESH = 0.75      # probability threshold for alert
    HORIZON_SEC  = 5         # seconds before alerting (avoid false alarms)

    def __init__(self, onnx_path: str):
        self.sess    = ort.InferenceSession(onnx_path,
                           providers=["CPUExecutionProvider"])
        self.history = []    # sliding window of recent probabilities

    def predict(self, x, edge_index, edge_attr, batch) -> dict:
        outputs = self.sess.run(
            None,
            {"x": x, "edge_index": edge_index,
             "edge_attr": edge_attr, "batch": batch}
        )
        logits = outputs[0]
        prob   = float(np.exp(logits[0, 1]) /
                       (np.exp(logits[0, 0]) + np.exp(logits[0, 1])))
        self.history.append(prob)
        if len(self.history) > self.HORIZON_SEC:
            self.history.pop(0)

        sustained = all(p > self.ALERT_THRESH for p in self.history)
        return {
            "prob_preictal": prob,
            "alert":         sustained,
            "window_probs":  list(self.history),
        }
'''

if __name__ == "__main__":
    from model import SeizureTGNN

    N, T = 23, 7680
    model = SeizureTGNN(n_channels=N, n_samples=T)

    # Simulate patient-specific data (replace with real preprocessed graphs)
    from torch_geometric.data import Data
    patient_data = [
        Data(x=torch.randn(N, T),
             edge_index=torch.randint(0, N, (2, 40)),
             edge_attr=torch.rand(40),
             y=torch.tensor([i % 2]))
        for i in range(30)
    ]

    print("\n--- Patient adaptation ---")
    model = patient_adapt(model, patient_data, epochs=4, lr=1e-4)

    print("\n--- ONNX export ---")
    export_onnx(model, n_nodes=N, n_time=T, n_edges=40)

    # Save inference helper
    with open("inference.py", "w") as f:
        f.write(INFERENCE_CODE)
    print("\nInference helper written → inference.py")
    print("Deploy seizure_predictor.onnx + inference.py on device.")
