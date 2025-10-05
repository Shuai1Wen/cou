# -*- coding: utf-8 -*-
"""Automatic GPU detection and device selection utilities."""

from __future__ import annotations

import warnings
from typing import Tuple, Literal

DeviceType = Literal["cuda", "cpu", "auto"]


def detect_best_device(verbose: bool = True) -> Tuple[str, str, bool]:
    """
    Automatically detect the best available device for computation.

    Returns:
        Tuple of (device, backend, use_torch):
            - device: 'cuda' or 'cpu'
            - backend: 'geomloss' or 'pot'
            - use_torch: True if PyTorch optimization available

    Priority:
        1. CUDA GPU + GeomLoss (fastest)
        2. CPU + GeomLoss (moderate)
        3. CPU + POT (slowest, fallback)
    """
    # Check PyTorch availability
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False
        if verbose:
            print("⚠️  PyTorch not available, using NumPy backend")
        return "cpu", "pot", False

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()

    # Check GeomLoss availability
    try:
        from geomloss import SamplesLoss
        has_geomloss = True
    except ImportError:
        has_geomloss = False

    # Determine best configuration
    if cuda_available and has_geomloss:
        device = "cuda"
        backend = "geomloss"
        use_torch = True
        if verbose:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[OK] GPU detected: {gpu_name}")
            print(f"     Using CUDA + GeomLoss (fastest mode)")
    elif has_geomloss:
        device = "cpu"
        backend = "geomloss"
        use_torch = True
        if verbose:
            print("[OK] GeomLoss available")
            print(f"     Using CPU + GeomLoss (moderate speed)")
    else:
        device = "cpu"
        backend = "pot"
        use_torch = False
        if verbose:
            print("[WARN] GeomLoss not available, falling back to POT")
            print(f"       Using CPU + POT (slower, but stable)")
            print(f"       Tip: Install for better performance:")
            print(f"            pip install torch geomloss pykeops")

    return device, backend, use_torch


def get_device_info() -> dict:
    """Get detailed device information for diagnostics."""
    info = {
        "torch_available": False,
        "cuda_available": False,
        "geomloss_available": False,
        "device_count": 0,
        "device_names": [],
        "recommended_device": "cpu",
        "recommended_backend": "pot",
    }

    try:
        import torch
        info["torch_available"] = True
        info["cuda_available"] = torch.cuda.is_available()

        if info["cuda_available"]:
            info["device_count"] = torch.cuda.device_count()
            info["device_names"] = [
                torch.cuda.get_device_name(i)
                for i in range(info["device_count"])
            ]
    except ImportError:
        pass

    try:
        from geomloss import SamplesLoss
        info["geomloss_available"] = True
    except ImportError:
        pass

    # Determine recommendation
    device, backend, _ = detect_best_device(verbose=False)
    info["recommended_device"] = device
    info["recommended_backend"] = backend

    return info


def print_device_info():
    """Print a detailed device capability report."""
    info = get_device_info()

    print("\n" + "="*60)
    print("Device Detection Report")
    print("="*60)

    print(f"\nDependencies:")
    print(f"   PyTorch:  {'Installed' if info['torch_available'] else 'Not installed'}")
    print(f"   GeomLoss: {'Installed' if info['geomloss_available'] else 'Not installed'}")

    print(f"\nGPU Status:")
    if info['cuda_available']:
        print(f"   CUDA:     Available")
        print(f"   Devices:  {info['device_count']}")
        for i, name in enumerate(info['device_names']):
            print(f"             [{i}] {name}")
    else:
        print(f"   CUDA:     Not available")

    print(f"\nRecommended Configuration:")
    print(f"   Device:   {info['recommended_device'].upper()}")
    print(f"   Backend:  {info['recommended_backend']}")

    if not info['geomloss_available'] or not info['cuda_available']:
        print(f"\nPerformance Tips:")
        if not info['torch_available']:
            print(f"   - Install PyTorch for better performance")
            print(f"     pip install torch")
        if not info['geomloss_available']:
            print(f"   - Install GeomLoss for GPU acceleration")
            print(f"     pip install geomloss pykeops")
        if info['torch_available'] and not info['cuda_available']:
            print(f"   - Install CUDA-enabled PyTorch for GPU support")
            print(f"     See: https://pytorch.org/get-started/locally/")

    print("="*60 + "\n")


if __name__ == "__main__":
    print_device_info()
