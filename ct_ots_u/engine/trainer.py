"""Training utilities for conditional transport models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Mapping, Optional

import math

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
    from torch import Tensor
    from torch.nn import Module
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from torch.nn.utils import clip_grad_norm_
except ImportError as exc:  # pragma: no cover - safety guard
    raise ImportError("ct_ots_u.engine.trainer requires PyTorch.") from exc

from ..transport.cfm import CFMConfig, CondFlowField
from ..transport.crr import CRRConfig, CondResidualRegressor
from ..transport.datasets import TransportBatch, TransportDataset

__all__ = [
    "TransportConfig",
    "TrainingSummary",
    "TransportTrainer",
]


LossKind = Literal["huber", "mse"]


@dataclass
class TransportConfig:
    """Configuration for :class:`TransportTrainer`."""

    model: Literal["crr", "cfm"] = "crr"
    hidden_dim: int = 128
    layers: int = 3
    dropout: float = 0.1
    activation: Literal["gelu", "relu", "silu"] = "gelu"
    layer_norm: bool = True
    spectral_norm: bool = True
    loss: LossKind = "huber"
    huber_delta: float = 1.0
    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 4096
    epochs: int = 30
    patience: int = 5
    grad_clip: float = 1.0
    ode_steps: int = 8
    time_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    monotonic_weight: float = 0.0
    homolog_weight: float = 0.0
    coral_weight: float = 0.0
    device: str = "cpu"
    dtype: str = "float32"


@dataclass
class TrainingSummary:
    """Container with training diagnostics."""

    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = math.inf
    state_dict: dict[str, Tensor] | None = None


class TransportTrainer:
    """High-level trainer for CRR / CFM transport models."""

    def __init__(
        self,
        config: TransportConfig,
        *,
        homolog_pairs: Optional[np.ndarray] = None,
        decoder: Optional[Module] = None,
    ) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)
        self.dtype = self._resolve_dtype(config.dtype, self.device)
        self.homolog_pairs = None
        if homolog_pairs is not None:
            hp = torch.as_tensor(homolog_pairs, dtype=torch.long)
            if hp.ndim != 2 or hp.shape[1] != 2:
                raise ValueError("homolog_pairs must have shape (n, 2)")
            self.homolog_pairs = hp
        self.decoder = decoder
        if self.decoder is not None:
            self.decoder.to(self.device, dtype=self.dtype)

        self.model: Module | None = None
        self.summary = TrainingSummary()

    @staticmethod
    def _resolve_device(name: str) -> torch.device:
        try:
            device = torch.device(name)
        except (RuntimeError, ValueError):  # pragma: no cover - defensive
            device = torch.device("cpu")
        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return device

    @staticmethod
    def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
        mapping = {
            "float32": torch.float32,
            "float": torch.float32,
            "float64": torch.float64,
            "double": torch.float64,
            "float16": torch.float16,
            "half": torch.float16,
            "bf16": torch.bfloat16,
        }
        if name not in mapping:
            raise ValueError(f"Unsupported dtype '{name}'")
        dtype = mapping[name]
        if dtype in {torch.float16, torch.bfloat16} and device.type == "cpu":
            return torch.float32
        return dtype

    def _build_model(self, dataset: TransportDataset) -> Module:
        cfg = self.config
        common_kwargs = dict(
            hidden_dim=cfg.hidden_dim,
            layers=cfg.layers,
            dropout=cfg.dropout,
            activation=cfg.activation,
            layer_norm=cfg.layer_norm,
            spectral_norm=cfg.spectral_norm,
        )
        if cfg.model == "crr":
            module = CondResidualRegressor(
                dataset.feature_dim,
                dataset.cond_dim,
                config=CRRConfig(**common_kwargs),
            )
        elif cfg.model == "cfm":
            module = CondFlowField(
                dataset.feature_dim,
                dataset.cond_dim,
                config=CFMConfig(time_grid=cfg.time_grid, **common_kwargs),
            )
        else:  # pragma: no cover - guard
            raise ValueError(f"Unsupported model '{cfg.model}'")
        return module.to(self.device, dtype=self.dtype)

    def fit(
        self,
        train: TransportDataset,
        *,
        valid: Optional[TransportDataset] = None,
    ) -> TrainingSummary:
        if self.model is None:
            self.model = self._build_model(train)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        train_loader = DataLoader(
            train,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=TransportDataset.collate,
        )
        valid_loader = (
            DataLoader(
                valid,
                batch_size=self.config.batch_size,
                collate_fn=TransportDataset.collate,
            )
            if valid is not None
            else None
        )

        best_state: dict[str, Tensor] | None = None
        best_val = math.inf
        best_epoch = -1
        patience = 0
        history: list[dict[str, float]] = []

        for epoch in range(self.config.epochs):
            train_metrics = self._run_epoch(train_loader, optimizer, training=True)
            if valid_loader is not None:
                val_metrics = self._run_epoch(valid_loader, optimizer=None, training=False)
                val_loss = float(val_metrics["base_loss"])
            else:
                val_metrics = {"base_loss": float(train_metrics["base_loss"])}
                val_loss = float(train_metrics["base_loss"])

            history.append(
                {
                    **{f"train_{k}": float(v) for k, v in train_metrics.items()},
                    **{f"valid_{k}": float(v) for k, v in val_metrics.items()},
                }
            )

            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                best_epoch = epoch
                patience = 0
                state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                best_state = state
            else:
                patience += 1

            if patience >= self.config.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.to(self.device)

        self.summary = TrainingSummary(
            history=history,
            best_epoch=best_epoch,
            best_val_loss=best_val,
            state_dict=best_state,
        )
        return self.summary

    def _run_epoch(
        self,
        loader: DataLoader[TransportBatch],
        optimizer: Optional[torch.optim.Optimizer],
        *,
        training: bool,
    ) -> Mapping[str, float]:
        assert self.model is not None
        model = self.model
        cfg = self.config
        model.train(mode=training)

        base_total = 0.0
        mono_total = 0.0
        homo_total = 0.0
        coral_total = 0.0
        total_loss = 0.0
        count = 0

        if training and optimizer is None:
            raise RuntimeError("Optimizer must be provided when training=True")

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                batch = batch.to(self.device, self.dtype)
                losses = self._batch_loss(batch)
                loss = losses["total"]

                if training and optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if cfg.grad_clip > 0:
                        clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.step()

                batch_size = batch.h0.shape[0]
                base_total += float(losses["base"].detach()) * batch_size
                mono_tensor = losses.get("monotonic")
                homo_tensor = losses.get("homolog")
                coral_tensor = losses.get("coral")
                mono_total += (float(mono_tensor.detach()) if mono_tensor is not None else 0.0) * batch_size
                homo_total += (float(homo_tensor.detach()) if homo_tensor is not None else 0.0) * batch_size
                coral_total += (float(coral_tensor.detach()) if coral_tensor is not None else 0.0) * batch_size
                total_loss += float(loss.detach()) * batch_size
                count += batch_size

        if count == 0:
            return {"base_loss": 0.0, "total_loss": 0.0}
        return {
            "base_loss": base_total / count,
            "monotonic_loss": mono_total / count,
            "homolog_loss": homo_total / count,
            "coral_loss": coral_total / count,
            "total_loss": total_loss / count,
        }

    def _batch_loss(self, batch: TransportBatch) -> Dict[str, Tensor]:
        assert self.model is not None
        cfg = self.config
        h0 = batch.h0
        h1 = batch.h1
        cond = batch.cond

        if cfg.model == "crr":
            residual = self.model(h0, cond)
            h_hat = h0 + residual
            base = self.model.loss(h_hat, h1, kind=cfg.loss, delta=cfg.huber_delta)
            delta = residual
        else:
            flow_model: CondFlowField = self.model  # type: ignore[assignment]
            t = flow_model.sample_time(h0.shape[0], h0.device)
            ht = (1.0 - t) * h0 + t * h1
            pred = flow_model(ht, t, cond)
            target = h1 - h0
            base = flow_model.loss(pred, target, kind=cfg.loss, delta=cfg.huber_delta)
            delta = pred
            h_hat = h0 + delta

        penalties: Dict[str, Tensor] = {
            "base": base,
            "total": base,
        }

        if cfg.monotonic_weight > 0 and batch.group is not None and batch.dose is not None:
            mono = self._monotonic_penalty(delta, batch.dose, batch.group)
            penalties["monotonic"] = mono
            penalties["total"] = penalties["total"] + cfg.monotonic_weight * mono

        if cfg.homolog_weight > 0 and self.homolog_pairs is not None:
            homo = self._homolog_penalty(delta, self.homolog_pairs.to(delta.device))
            penalties["homolog"] = homo
            penalties["total"] = penalties["total"] + cfg.homolog_weight * homo

        if cfg.coral_weight > 0 and batch.batch is not None:
            coral = self._coral_penalty(h_hat, batch.batch)
            penalties["coral"] = coral
            penalties["total"] = penalties["total"] + cfg.coral_weight * coral

        return penalties

    @staticmethod
    def _monotonic_penalty(delta: Tensor, dose: Tensor, group: Tensor) -> Tensor:
        delta_norm = delta.norm(dim=1)
        dose = dose.view(-1)
        group = group.view(-1)
        unique_groups = torch.unique(group)
        penalties = []
        for g in unique_groups:
            mask = group == g
            if mask.sum() < 2:
                continue
            d = dose[mask]
            n = delta_norm[mask]
            diff_dose = d.unsqueeze(1) - d.unsqueeze(0)
            diff_effect = n.unsqueeze(1) - n.unsqueeze(0)
            mask_hi = diff_dose > 0
            if mask_hi.any():
                penalties.append(torch.relu(-diff_effect[mask_hi]).mean())
        if not penalties:
            return delta.new_tensor(0.0)
        return torch.stack(penalties).mean()

    def _homolog_penalty(self, delta: Tensor, homolog_pairs: Tensor) -> Tensor:
        decoded = self.decoder(delta) if self.decoder is not None else delta
        idx1 = homolog_pairs[:, 0]
        idx2 = homolog_pairs[:, 1]
        max_index = int(homolog_pairs.max().item())
        if decoded.shape[1] <= max_index:
            raise ValueError("Homolog pair index out of bounds for decoded features")
        diff = decoded[:, idx1] - decoded[:, idx2]
        return (diff ** 2).mean()

    @staticmethod
    def _coral_penalty(embeddings: Tensor, batch: Tensor) -> Tensor:
        batch = batch.view(-1)
        unique_batches = torch.unique(batch)
        means = []
        covs = []
        for b in unique_batches:
            mask = batch == b
            if mask.sum() < 2:
                continue
            subset = embeddings[mask]
            mean = subset.mean(dim=0, keepdim=True)
            centered = subset - mean
            denom = max(int(mask.sum()) - 1, 1)
            cov = (centered.T @ centered) / float(denom)
            means.append(mean.squeeze(0))
            covs.append(cov)
        if len(covs) < 2:
            return embeddings.new_tensor(0.0)
        mean_penalty = 0.0
        cov_penalty = 0.0
        pairs = 0
        for i in range(len(covs)):
            for j in range(i + 1, len(covs)):
                mean_penalty += F.mse_loss(means[i], means[j])
                cov_penalty += F.mse_loss(covs[i], covs[j])
                pairs += 1
        if pairs == 0:
            return embeddings.new_tensor(0.0)
        return (mean_penalty + cov_penalty) / pairs

    def predict(self, dataset: TransportDataset) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been trained")
        loader = DataLoader(dataset, batch_size=self.config.batch_size)
        outputs: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device, self.dtype)
                if self.config.model == "crr":
                    residual = self.model(batch.h0, batch.cond)
                    h_hat = batch.h0 + residual
                else:
                    flow_model: CondFlowField = self.model  # type: ignore[assignment]
                    h_hat = flow_model.predict(batch.h0, batch.cond, steps=self.config.ode_steps)
                outputs.append(h_hat.cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def state_dict(self) -> Mapping[str, Tensor]:
        if self.model is None:
            raise RuntimeError("Model not initialised")
        return self.model.state_dict()

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        if self.model is None:
            raise RuntimeError("Model not initialised")
        self.model.load_state_dict(state)

