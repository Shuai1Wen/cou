#!/usr/bin/env python3
"""Example usage of the standardized CT-OTS-U API."""

import os
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ct_ots_u.config import CTOTSUConfig
from ct_ots_u.gating import fit_gmm_BICICL, merge_by_w2
from ct_ots_u.eval.consistency import semigroup_consistency, pushforward
from ct_ots_u.selection.hparam_plateau import tau_eps_grid


def example_minimal_training():
    """Minimal example showing the new standardized API."""

    # 1. Configuration
    config = CTOTSUConfig.from_env()
    print("Default config:", config.to_dict())

    # 2. Example data (in practice, load from h5ad)
    import numpy as np
    np.random.seed(config.seed)

    X0 = np.random.randn(100, 20)  # Source
    Xt = np.random.randn(120, 20)  # Target

    # 3. Gate discovery using new API
    best_gmm, table = fit_gmm_BICICL(
        Xt,
        k_grid=config.gate.k_grid,
        random_state=config.seed
    )
    print(f"Best K: {best_gmm[0]}, ICL: {best_gmm[1]:.3f}")

    # Get initial labels
    labels_init = best_gmm[2].predict(Xt)

    # 4. Wasserstein merging using new API
    labels_merged, K_final = merge_by_w2(
        Xt,
        labels_init,
        w_thresh=config.gate.wmerge_thresh
    )
    print(f"After merging: {K_final} clusters")

    # 5. Hyperparameter scanning using new API
    hyperparam_results = tau_eps_grid(
        X0, Xt,
        taus=config.hyperparam.tau_grid,
        epsilons=config.hyperparam.reg_grid,
        repeats=config.hyperparam.repeats
    )
    print(f"Best hyperparams: tau={hyperparam_results['best'][0]:.2f}, eps={hyperparam_results['best'][1]:.3f}")
    print(f"Plateau size: {len(hyperparam_results['plateau'])}")

    # 6. Consistency evaluation (simplified example)
    d = X0.shape[1]
    L1 = np.random.randn(d, d) * 0.01  # Example generator
    L2 = np.random.randn(d, d) * 0.01  # Example generator

    consistency_metrics = semigroup_consistency(
        X0, L1, L2, Xt,
        tau1=config.uot.tau,
        tau2=config.uot.tau,
        eps=config.uot.eps,
        reg_m=config.uot.reg_m
    )
    print(f"Consistency improvement: {consistency_metrics['improve']:.3f}")

    return {
        'config': config,
        'K_final': K_final,
        'hyperparams': hyperparam_results,
        'consistency': consistency_metrics
    }


def example_with_real_data():
    """Example using the training script with real data."""

    print("\nExample with real data (requires h5ad file):")

    # Check if we have the microglia data
    data_path = ROOT / "data" / "GSE160936" / "GSE160936_microglia_only.h5ad"

    if data_path.exists():
        print(f"Found data file: {data_path}")

        # Example training command
        train_cmd = f"""
python -m ct_ots_u.scripts.run_train \\
    --h5ad {data_path} \\
    --stage_key braak_stage \\
    --src_stage 0 \\
    --dst_stage 6 \\
    --celltype_key cell_type \\
    --celltype "microglial cell" \\
    --out results/example_train_summary.json \\
    --verbose
"""
        print("Training command:")
        print(train_cmd)

        # Example validation command
        validate_cmd = """
python -m ct_ots_u.scripts.run_validate \\
    --summary results/example_train_summary.json \\
    --out results/example_validation_report.json \\
    --verbose
"""
        print("\nValidation command:")
        print(validate_cmd)

    else:
        print(f"Data file not found: {data_path}")
        print("Please ensure the microglia data is downloaded and processed.")


if __name__ == "__main__":
    print("=== CT-OTS-U Standardized API Example ===")

    # Run minimal example
    results = example_minimal_training()

    # Show real data usage
    example_with_real_data()

    print("\n=== Example completed ===")
    print("Key improvements:")
    print("✅ Unified configuration management")
    print("✅ Standardized training/validation scripts")
    print("✅ API compliance with documentation")
    print("✅ Modular component structure")