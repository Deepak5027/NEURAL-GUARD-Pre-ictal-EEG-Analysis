"""
Stage 4 — Training Loop
- Handles class imbalance (pre-ictal windows are rare ~5% of data)
- Focal loss to down-weight easy negatives
- Early stopping on validation AUC
- Saves best checkpoint
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import numpy as np


# ── Focal Loss (handles extreme class imbalance) ──────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha           # weight for positive (pre-ictal) class

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        ce    = F.cross_entropy(logits, targets, reduction="none")
        pt    = probs[range(len(targets)), targets]
        alpha = torch.where(targets == 1,
                            torch.tensor(self.alpha),
                            torch.tensor(1 - self.alpha)).to(logits.device)
        loss  = alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()


# ── Training utilities ────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss   = criterion(logits, batch.y.squeeze())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch)
        probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        labels = batch.y.squeeze().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels)

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    preds      = (all_probs >= 0.5).astype(int)

    auc = roc_auc_score(all_labels, all_probs)
    f1  = f1_score(all_labels, preds, zero_division=0)
    cm  = confusion_matrix(all_labels, preds)
    return {"auc": auc, "f1": f1, "cm": cm}


# ── Main training function ────────────────────────────────────────────────────
def train(model,
          train_dataset,
          val_dataset,
          epochs:    int   = 50,
          batch_size: int  = 32,
          lr:        float = 3e-4,
          patience:  int   = 10,
          save_path: str   = "best_model.pt"):

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    criterion = FocalLoss(gamma=2.0, alpha=0.75)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=4)

    best_auc      = 0.0
    patience_cnt  = 0

    print(f"Training on {device} | {len(train_dataset)} train / {len(val_dataset)} val samples")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics    = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch:3d} | loss {train_loss:.4f} | "
              f"AUC {metrics['auc']:.4f} | F1 {metrics['f1']:.4f}")

        if metrics["auc"] > best_auc:
            best_auc     = metrics["auc"]
            patience_cnt = 0
            torch.save(model.state_dict(), save_path)
            print(f"           >>> best model saved (AUC={best_auc:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nBest validation AUC: {best_auc:.4f}")
    return best_auc


if __name__ == "__main__":
    from model        import SeizureTGNN
    from graph_builder import build_graph_dataset
    import torch
    from torch_geometric.data import Data

    N, T = 23, 7680

    def fake_dataset(n, pos_ratio=0.05):
        graphs, labels = [], []
        for i in range(n):
            label = 1 if np.random.rand() < pos_ratio else 0
            x  = torch.randn(N, T)
            ei = torch.randint(0, N, (2, 40))
            ea = torch.rand(40)
            graphs.append(Data(x=x, edge_index=ei, edge_attr=ea,
                               y=torch.tensor([label])))
        return graphs

    train_data = fake_dataset(200)
    val_data   = fake_dataset(50)
    model      = SeizureTGNN(n_channels=N, n_samples=T)

    train(model, train_data, val_data,
          epochs=5, batch_size=8, save_path="best_model.pt")
