"""Datasets for conditional transport models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
    from torch import Tensor
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover - safety guard
    raise ImportError(
        "ct_ots_u.transport.datasets requires PyTorch to be installed."
    ) from exc

__all__ = ["TransportBatch", "TransportDataset"]


@dataclass
class TransportBatch:
    """Mini-batch container returned by :class:`TransportDataset`.

    The batch exposes the raw tensors required by the transport trainer. All
    fields are optional to keep the dataset flexible – penalties such as
    monotonicity or CORAL simply remain inactive when their corresponding
    tensors are absent.
    """

    h0: Tensor
    h1: Tensor
    cond: Tensor
    dose: Optional[Tensor] = None
    group: Optional[Tensor] = None
    batch: Optional[Tensor] = None

    def to(self, device: torch.device, dtype: torch.dtype) -> "TransportBatch":
        """Move tensors to ``device`` / ``dtype`` without modifying in place."""

        kwargs: MutableMapping[str, Optional[Tensor]] = {}
        for name, value in self.__dict__.items():
            if value is None:
                kwargs[name] = None
            else:
                if name in {"group", "batch"}:
                    kwargs[name] = value.to(device=device, dtype=torch.long)
                else:
                    kwargs[name] = value.to(device=device, dtype=dtype)
        return TransportBatch(**kwargs)  # type: ignore[arg-type]


class TransportDataset(Dataset[TransportBatch]):
    """Tensor dataset wrapping paired embeddings and conditioning vectors.

    Parameters
    ----------
    h0, h1:
        Arrays describing the unperturbed and perturbed embeddings.
    cond:
        Conditioning vectors (concatenated perturbation / metadata encodings).
    dose:
        Optional per-sample scalar describing perturbation magnitude.
    group:
        Optional integer identifier grouping samples that share the same
        biological context (perturbation, cell type, mLOY state, ...). The
        trainer uses this together with ``dose`` to enforce monotonicity.
    batch:
        Optional batch / donor identifiers used by the CORAL penalty.
    """

    def __init__(
        self,
        h0: np.ndarray | Tensor,
        h1: np.ndarray | Tensor,
        cond: np.ndarray | Tensor,
        *,
        dose: Optional[np.ndarray | Tensor] = None,
        group: Optional[np.ndarray | Tensor] = None,
        batch: Optional[np.ndarray | Tensor] = None,
    ) -> None:
        if isinstance(h0, np.ndarray):
            h0 = torch.as_tensor(h0, dtype=torch.float32)
        if isinstance(h1, np.ndarray):
            h1 = torch.as_tensor(h1, dtype=torch.float32)
        if isinstance(cond, np.ndarray):
            cond = torch.as_tensor(cond, dtype=torch.float32)
        if h0.shape != h1.shape:
            raise ValueError(f"h0 {tuple(h0.shape)} and h1 {tuple(h1.shape)} must match")
        if h0.shape[0] != cond.shape[0]:
            raise ValueError("Number of samples in h0/h1 and cond must align")
        self.h0 = h0.contiguous()
        self.h1 = h1.contiguous()
        self.cond = cond.contiguous()
        self.feature_dim = self.h0.shape[1]
        self.cond_dim = self.cond.shape[1]

        self.dose = self._optional_vector(dose, torch.float32)
        self.group = self._optional_vector(group, torch.long)
        self.batch = self._optional_vector(batch, torch.long)

    @staticmethod
    def _optional_vector(
        array: Optional[np.ndarray | Tensor], dtype: torch.dtype
    ) -> Optional[Tensor]:
        if array is None:
            return None
        if isinstance(array, np.ndarray):
            return torch.as_tensor(array, dtype=dtype).contiguous()
        return array.to(dtype=dtype).contiguous()

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return self.h0.shape[0]

    def __getitem__(self, index: int) -> TransportBatch:
        batch = TransportBatch(
            h0=self.h0[index],
            h1=self.h1[index],
            cond=self.cond[index],
            dose=None if self.dose is None else self.dose[index],
            group=None if self.group is None else self.group[index],
            batch=None if self.batch is None else self.batch[index],
        )
        return batch

    def numpy(self) -> Mapping[str, np.ndarray]:
        """Return dataset arrays as ``numpy.ndarray`` for logging/debugging."""

        result: dict[str, np.ndarray] = {
            "h0": self.h0.cpu().numpy(),
            "h1": self.h1.cpu().numpy(),
            "cond": self.cond.cpu().numpy(),
        }
        if self.dose is not None:
            result["dose"] = self.dose.cpu().numpy()
        if self.group is not None:
            result["group"] = self.group.cpu().numpy()
        if self.batch is not None:
            result["batch"] = self.batch.cpu().numpy()
        return result

    @classmethod
    def from_iterable(
        cls,
        iterator: Iterable[Mapping[str, np.ndarray | Tensor]],
    ) -> "TransportDataset":
        """Build dataset from an iterator of mapping objects.

        The iterator should yield dictionaries containing the keys ``h0``,
        ``h1`` and ``cond``; optional metadata keys (``dose``, ``group``,
        ``batch``) are detected automatically.
        """

        arrays: dict[str, list[Tensor]] = {"h0": [], "h1": [], "cond": []}
        optional_keys = {"dose", "group", "batch"}
        optional: dict[str, list[Tensor]] = {key: [] for key in optional_keys}
        feature_shapes: dict[str, torch.Size] = {}
        n_samples = 0

        def _ensure_feature_tensor(value: np.ndarray | Tensor, key: str) -> Tensor:
            tensor = torch.as_tensor(value)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim == 2 and tensor.shape[0] == 1:
                tensor = tensor.contiguous()
            else:
                raise ValueError(
                    f"Iterator key '{key}' must provide a single sample with shape (feature_dim,)"
                )
            nonlocal feature_shapes
            shape = tensor.shape[1:]
            if key not in feature_shapes:
                feature_shapes[key] = shape
            elif feature_shapes[key] != shape:
                raise ValueError(
                    f"Inconsistent feature dimensions for '{key}': expected {feature_shapes[key]},"
                    f" got {shape}"
                )
            return tensor.contiguous()

        def _ensure_optional_tensor(value: np.ndarray | Tensor, key: str) -> Tensor:
            tensor = torch.as_tensor(value)
            if tensor.ndim == 0:
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim == 1:
                tensor = tensor.contiguous()
            elif tensor.ndim == 2 and tensor.shape[0] == 1:
                tensor = tensor.reshape(-1)
            else:
                raise ValueError(
                    f"Iterator key '{key}' must provide a single scalar per sample"
                )
            if tensor.numel() != 1:
                raise ValueError(
                    f"Iterator key '{key}' must provide exactly one value per sample"
                )
            return tensor.contiguous()

        for sample in iterator:
            for key in ("h0", "h1", "cond"):
                if key not in sample:
                    raise KeyError(f"Iterator sample missing required key '{key}'")
                arrays[key].append(_ensure_feature_tensor(sample[key], key))
            for key in optional_keys:
                value = sample.get(key)
                if value is not None:
                    optional[key].append(_ensure_optional_tensor(value, key))
            n_samples += 1

        if n_samples == 0:
            raise ValueError("Iterator must yield at least one sample")

        stacked = {k: torch.cat(vals, dim=0) for k, vals in arrays.items()}
        kwargs: dict[str, np.ndarray | Tensor] = {}
        for key, values in optional.items():
            if values:
                if len(values) != n_samples:
                    raise ValueError(
                        f"Optional key '{key}' present for {len(values)} samples; expected {n_samples}"
                    )
                kwargs[key] = torch.cat(values, dim=0)
        return cls(stacked["h0"], stacked["h1"], stacked["cond"], **kwargs)
