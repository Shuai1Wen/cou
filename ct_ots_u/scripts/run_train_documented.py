#!/usr/bin/env python3
"""CT-OTS-U training using documented algorithm implementation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ct_ots_u.config import CTOTSUConfig
from ct_ots_u.data import preprocess_adata, extract_condition_arrays
from ct_ots_u.gating import fit_gmm_BICICL, merge_by_w2
from ct_ots_u.model.train import fit_branch_generator  # Use documented algorithm
from ct_ots_u.selection.hparam_plateau import tau_eps_grid
from ct_ots_u.eval.consistency import semigroup_consistency
from ct_ots_u.utils import set_global_seed

import anndata as ad


def train_with_documented_algorithm(h5ad_path: str, config: CTOTSUConfig,
                                  stage_key: str, src_stage: str, dst_stage: str,
                                  celltype_key: str = None, celltype: str = None):
    """Train CT-OTS-U model using the documented algorithm."""

    # Set seed for reproducibility
    set_global_seed(config.seed)

    if config.verbose:
        print(f"Loading data from: {h5ad_path}")

    # Load and preprocess
    adata = ad.read_h5ad(h5ad_path)
    preprocess_adata(adata, n_pcs=config.data.n_pcs, seed=config.seed)

    # Filter by cell type if specified
    if celltype_key and celltype:
        if celltype_key in adata.obs.columns:
            mask = adata.obs[celltype_key] == celltype
            adata = adata[mask].copy()
            if config.verbose:
                print(f"Filtered to {celltype}: {adata.n_obs} cells")

    # Extract stage-specific data
    if stage_key in adata.obs.columns:
        src_mask = adata.obs[stage_key] == src_stage
        dst_mask = adata.obs[stage_key] == dst_stage

        if src_mask.sum() == 0:
            raise ValueError(f"No cells found for source stage: {src_stage}")
        if dst_mask.sum() == 0:
            raise ValueError(f"No cells found for destination stage: {dst_stage}")

        Xs = adata[src_mask].obsm['X_pca']
        Xt = adata[dst_mask].obsm['X_pca']

        if config.verbose:
            print(f"Source stage {src_stage}: {Xs.shape[0]} cells")
            print(f"Destination stage {dst_stage}: {Xt.shape[0]} cells")
    else:
        raise ValueError(f"Stage key '{stage_key}' not found in data")

    # Sample if datasets are too large
    if Xs.shape[0] > config.data.sample_size:
        rng = np.random.default_rng(config.seed)
        idx_s = rng.choice(Xs.shape[0], config.data.sample_size, replace=False)
        Xs = Xs[idx_s]

    if Xt.shape[0] > config.data.sample_size:
        rng = np.random.default_rng(config.seed)
        idx_t = rng.choice(Xt.shape[0], config.data.sample_size, replace=False)
        Xt = Xt[idx_t]

    if config.verbose:
        print(f"Final data shapes: Xs={Xs.shape}, Xt={Xt.shape}")

    # 1. Gate discovery using documented API
    if config.verbose:
        print("Stage 1: Gate discovery with GMM+ICL selection")

    best_gmm, gmm_table = fit_gmm_BICICL(
        Xt,
        k_grid=config.gate.k_grid,
        random_state=config.seed
    )

    K_initial = best_gmm[0]
    icl_score = best_gmm[1]
    gmm_model = best_gmm[2]

    if config.verbose:
        print(f"   Initial K selected: {K_initial} (ICL: {icl_score:.2f})")

    # Get initial cluster labels
    labels_initial = gmm_model.predict(Xt)

    # 2. Wasserstein merging using documented API
    if config.verbose:
        print("Stage 2: Wasserstein-based cluster merging")

    labels_merged, K_final = merge_by_w2(
        Xt, labels_initial,
        w_thresh=config.gate.wmerge_thresh
    )

    if config.verbose:
        print(f"   Final K after merging: {K_final}")

    # 3. Train branch generators using documented algorithm
    if config.verbose:
        print("Stage 3: Training branch-specific generators")

    generators = []
    training_losses = []

    for k in range(K_final):
        if config.verbose:
            print(f"   Training generator for branch {k+1}/{K_final}")

        # Get branch-specific target data
        branch_mask = labels_merged == k
        Xt_branch = Xt[branch_mask]

        if len(Xt_branch) < 10:  # Skip branches with too few cells
            if config.verbose:
                print(f"   Skipping branch {k} (only {len(Xt_branch)} cells)")
            generators.append(None)
            training_losses.append(np.inf)
            continue

        # Train using documented algorithm
        L_k = fit_branch_generator(
            Xs, Xt_branch,
            tau=config.uot.tau,
            rank=min(config.rank.grid),  # Use smallest rank for speed
            steps=100,  # Fewer steps for demo
            lr=1e-2,
            alpha=config.stable.alpha,
            seed=config.seed + k  # Different seed per branch
        )

        generators.append(L_k)

        # Compute final loss for this branch
        from ct_ots_u.model.semigroup import pushforward
        from ct_ots_u.ot.uot_losses import uot_sinkhorn_cost

        Y_pred = pushforward(Xs, L_k, tau=config.uot.tau)
        loss, _, _ = uot_sinkhorn_cost(Y_pred, Xt_branch, reg=config.uot.eps, reg_m=config.uot.reg_m)
        training_losses.append(loss)

        if config.verbose:
            print(f"   Branch {k} final loss: {loss:.6f}")

    # 4. Hyperparameter optimization
    if config.verbose:
        print("Stage 4: Hyperparameter optimization")

    hyperparam_results = tau_eps_grid(
        Xs, Xt,
        taus=config.hyperparam.tau_grid,
        epsilons=config.hyperparam.reg_grid,
        repeats=config.hyperparam.repeats
    )

    best_tau, best_eps = hyperparam_results['best'][:2]
    if config.verbose:
        print(f"   Best hyperparameters: tau={best_tau:.2f}, eps={best_eps:.3f}")

    # 5. Consistency evaluation (simplified)
    if config.verbose:
        print("Stage 5: Consistency evaluation")

    consistency_results = {}
    if len(generators) >= 2 and generators[0] is not None and generators[1] is not None:
        # Use first two valid generators for consistency test
        L1, L2 = generators[0], generators[1]

        consistency_metrics = semigroup_consistency(
            Xs, L1, L2, Xt,
            tau1=config.uot.tau,
            tau2=config.uot.tau,
            eps=config.uot.eps,
            reg_m=config.uot.reg_m
        )

        consistency_results = {
            'err_composed': consistency_metrics['err_composed'],
            'err_direct': consistency_metrics['err_direct'],
            'improvement': consistency_metrics['improve']
        }

        if config.verbose:
            print(f"   Consistency improvement: {consistency_metrics['improve']:.4f}")
    else:
        if config.verbose:
            print("   Skipped (insufficient valid generators)")

    # 6. Compile results
    results = {
        'algorithm': 'documented_ct_ots_u',
        'config': config.to_dict(),
        'data_info': {
            'source_cells': int(Xs.shape[0]),
            'target_cells': int(Xt.shape[0]),
            'features': int(Xs.shape[1]),
            'stages': f"{src_stage} -> {dst_stage}"
        },
        'gating': {
            'K_initial': int(K_initial),
            'K_final': int(K_final),
            'icl_score': float(icl_score),
            'merge_threshold': config.gate.wmerge_thresh
        },
        'training': {
            'n_generators': len([g for g in generators if g is not None]),
            'training_losses': [float(loss) if loss != np.inf else None for loss in training_losses],
            'mean_loss': float(np.mean([loss for loss in training_losses if loss != np.inf]))
        },
        'hyperparameters': {
            'best_tau': float(best_tau),
            'best_eps': float(best_eps),
            'plateau_size': len(hyperparam_results['plateau'])
        },
        'consistency': consistency_results,
        'generators_shapes': [g.shape if g is not None else None for g in generators]
    }

    return results, generators


def main():
    """Main training entry point."""

    parser = argparse.ArgumentParser(description="CT-OTS-U Training (Documented Algorithm)")
    parser.add_argument('--h5ad', required=True, help='Path to h5ad file')
    parser.add_argument('--stage_key', default='braak_stage', help='Stage column name')
    parser.add_argument('--src_stage', default='0', help='Source stage value')
    parser.add_argument('--dst_stage', default='6', help='Destination stage value')
    parser.add_argument('--celltype_key', help='Cell type column name')
    parser.add_argument('--celltype', help='Cell type value to filter')
    parser.add_argument('--out', default='results/documented_train_summary.json', help='Output JSON file')
    parser.add_argument('--seed', type=int, help='Random seed override')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Load configuration
    config = CTOTSUConfig.from_env()
    if args.seed is not None:
        config.seed = args.seed
    if args.verbose:
        config.verbose = True

    print("CT-OTS-U Training with Documented Algorithm")
    print("=" * 50)

    try:
        results, generators = train_with_documented_algorithm(
            args.h5ad, config, args.stage_key,
            args.src_stage, args.dst_stage,
            args.celltype_key, args.celltype
        )

        # Save results
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)

        print("=" * 50)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print(f"Results saved to: {args.out}")
        print(f"Final K: {results['gating']['K_final']}")
        print(f"Valid generators: {results['training']['n_generators']}")
        print(f"Mean training loss: {results['training']['mean_loss']:.6f}")

        if results['consistency']:
            improvement = results['consistency']['improvement']
            print(f"Consistency improvement: {improvement:.4f}")
            decision = "MODEL_GO" if improvement >= config.eval_tol.semigroup_improve_min else "NEEDS_REVISION"
            print(f"Model decision: {decision}")

        return results

    except Exception as e:
        print(f"TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()