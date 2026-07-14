from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer,
        scheduler,
        loss_fn: nn.Module,
        device: torch.device,
        checkpoint_dir: Path,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_iou = 0.0

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> dict:
        history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_f1": []}
        for epoch in range(1, epochs + 1):
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics = self._run_epoch(val_loader, train=False)
            if self.scheduler is not None:
                self.scheduler.step()

            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_iou"].append(val_metrics["iou"])
            history["val_f1"].append(val_metrics["f1"])

            is_best = val_metrics["iou"] > self.best_val_iou
            if is_best:
                self.best_val_iou = val_metrics["iou"]
            self._save_checkpoint(epoch, val_metrics["iou"], is_best)

            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_iou={val_metrics['iou']:.4f} | "
                f"val_f1={val_metrics['f1']:.4f}"
                + (" *" if is_best else "")
            )
        return history

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict:
        self.model.train(train)
        total_loss = 0.0
        tp = fp = fn = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for images, masks in loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                logits = self.model(images)
                loss = self.loss_fn(logits, masks)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()

                # metrics: ignore 255
                preds = (torch.sigmoid(logits) > 0.5).long()
                valid = masks != 255
                p = preds[valid]
                t = masks[valid].long()
                tp += (p * t).sum().item()
                fp += (p * (1 - t)).sum().item()
                fn += ((1 - p) * t).sum().item()

        iou = tp / (tp + fp + fn + 1e-6)
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
        return {"loss": total_loss / len(loader), "iou": iou, "f1": f1}

    def _save_checkpoint(self, epoch: int, val_iou: float, is_best: bool):
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_iou": val_iou,
        }
        torch.save(state, self.checkpoint_dir / "latest.pth")
        if is_best:
            torch.save(state, self.checkpoint_dir / "best.pth")

    def load_checkpoint(self, path: str | Path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.best_val_iou = state.get("val_iou", 0.0)
        return state.get("epoch", 0)
