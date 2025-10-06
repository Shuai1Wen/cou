#!/usr/bin/env python3
"""CT-OTS-U training entry point with gating, stability controls, and optional PyTorch acceleration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Sequence

import anndata as ad
import numpy as np
import re
import scanpy as sc

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ct_ots_u.config import CTOTSUConfig
from ct_ots_u.data import (
    ConditionSplits,
    extract_condition_arrays,
    make_pseudotime_bins,
    preprocess_adata,
)
from ct_ots_u.gating import fit_gmm_BICICL, merge_by_w2
from ct_ots_u.gating_kselect import enforce_min_cluster, train_gating
from ct_ots_u.metric_msigdb import load_gmt
from ct_ots_u.model import GatedSemigroup, semigroup_consistency
from ct_ots_u.selection import rank_selection_cv
from ct_ots_u.selection.hparam_plateau import tau_eps_grid, cross_domain_plateau_check
from ct_ots_u.utils import ensure_dir

try:  # pragma: no cover - optional path
    from ct_ots_u.gating.calibration import calibrate_temperature
except Exception:  # noqa: BLE001
    calibrate_temperature = None

try:  # pragma: no cover - optional path
    from ct_ots_u.gating.ood_detection import MahalanobisOOD
except Exception:  # noqa: BLE001
    MahalanobisOOD = None

try:  # pragma: no cover - optional path
    from ct_ots_u.data.batch_decision import decide_harmony
    from ct_ots_u.data.harmony_integration import apply_harmony_integration
except Exception:  # noqa: BLE001
    decide_harmony = None
    apply_harmony_integration = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_stage_arg(value: str) -> Sequence[str]:
    parts = [token.strip() for token in value.split(',') if token.strip()]
    if not parts:
        raise ValueError('Stage arguments must contain at least one label')
    return parts


def _build_stage_map(args: argparse.Namespace) -> Dict[str, Sequence[str]]:
    return {
        'early': _parse_stage_arg(args.src_stage),
        'mid': _parse_stage_arg(args.mid_stage),
        'late': _parse_stage_arg(args.dst_stage),
    }


def _parse_target_stage_mapping(spec: str) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    if not spec:
        return mapping
    chunks = re.split(r'[;]', spec)
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk or ':' not in chunk:
            continue
        key, value = chunk.split(':', 1)
        key = key.strip().lower()
        labels = [lab.strip() for lab in re.split(r'[|,]', value) if lab.strip()]
        if not labels:
            continue
        if key in {'early', 'mid', 'late'}:
            mapping[key] = labels
        elif key in {'src', 'source', 'control'}:
            mapping.setdefault('early', []).extend(labels)
        elif key in {'midpoint', 'middle'}:
            mapping.setdefault('mid', []).extend(labels)
        elif key in {'dst', 'target', 'case'}:
            mapping.setdefault('late', []).extend(labels)
    if 'mid' not in mapping:
        if 'early' in mapping:
            mapping['mid'] = list(mapping['early'])
        elif 'late' in mapping:
            mapping['mid'] = list(mapping['late'])
    return mapping


def _preprocess_data(
    h5ad_path: str,
    config: CTOTSUConfig,
    stage_key: str,
    stage_map: Dict[str, Sequence[str]],
    celltype_key: str | None,
    celltype: str | None,
    pct_mt_limit: float | None,
    batch_key: str | None,
    harmony_backend: str,
    use_batch_decision: bool,
    batch_delta_thresh: float,
    bio_drop_thresh: float,
) -> tuple[ConditionSplits, ad.AnnData]:
    adata = ad.read_h5ad(h5ad_path)

    inferred_pct_limit = pct_mt_limit
    if inferred_pct_limit is None:
        stage_key_lower = stage_key.lower()
        celltype_lower = (celltype or '').lower()
        if 'microglia' in celltype_lower or 'braak' in stage_key_lower:
            inferred_pct_limit = config.data.pct_mt_microglia
        else:
            inferred_pct_limit = config.data.pct_mt_pbmc

    preprocess_adata(
        adata,
        n_pcs=config.data.n_pcs,
        seed=config.seed,
        hvgs=config.data.n_hvg,
        scale_max=config.data.max_value,
        min_counts=config.data.min_counts,
        min_genes=config.data.min_genes,
        min_cells=config.data.min_cells,
        pct_mt_limit=inferred_pct_limit,
    )

    if 'X_pca' not in adata.obsm:
        sc.tl.pca(adata, n_comps=config.data.n_pcs, svd_solver='arpack', random_state=config.seed)

    if config.verbose:
        print(
            f"[*] QC thresholds: min_counts={config.data.min_counts}, "
            f"min_genes={config.data.min_genes}, min_cells={config.data.min_cells}, "
            f"pct_mt_limit={inferred_pct_limit}"
        )

    if celltype_key and celltype:
        mask = adata.obs[celltype_key].astype(str) == celltype
        if mask.sum() == 0:
            raise ValueError(
                f"No cells matched celltype '{celltype}' via column '{celltype_key}'"
            )
        adata = adata[mask].copy()

    if config.train.use_harmony and batch_key:
        if apply_harmony_integration is None:
            raise RuntimeError('Harmony integration requested but optional dependency is missing')
        if batch_key not in adata.obs.columns:
            raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")
        batch_labels = adata.obs[batch_key].values
        if use_batch_decision:
            if decide_harmony is None:
                raise RuntimeError('Batch auto-decision requested but optional dependency is missing')
            X_raw = adata.obsm['X_pca']
            bio_labels = adata.obs.get(celltype_key or stage_key, adata.obs[stage_key]).values
            X_harmony = apply_harmony_integration(
                X_raw,
                batch_labels,
                theta=config.train.harmony_theta,
                backend=harmony_backend,
            )
            decision = decide_harmony(
                X_raw,
                X_harmony,
                batch_labels,
                bio_labels,
                batch_delta_thresh,
                bio_drop_thresh,
            )
            if decision['enable_harmony']:
                if config.verbose:
                    print('[*] Harmony applied (auto-decided)')
                adata.obsm['X_pca'] = X_harmony
            else:
                if config.verbose:
                    print('[!] Harmony skipped (auto-decided)')
            adata.uns['batch_decision'] = decision
        else:
            X_harmony = apply_harmony_integration(
                adata.obsm['X_pca'],
                batch_labels,
                theta=config.train.harmony_theta,
                backend=harmony_backend,
            )
            adata.obsm['X_pca'] = X_harmony
            if config.verbose:
                print(f"[*] Harmony applied: batch_key={batch_key}, theta={config.train.harmony_theta}")

    if stage_key == 'pseudotime':
        adata = make_pseudotime_bins(
            adata,
            cond_key=celltype_key or 'disease_state',
            ctrl_label='Control',
            seed=config.seed,
        )
        splits = extract_condition_arrays(
            adata,
            stage_key='stage_bin',
            stage_map={'early': ['early'], 'mid': ['mid'], 'late': ['late']},
            sample_size=config.data.sample_size,
            seed=config.seed,
        )
    else:
        splits = extract_condition_arrays(
            adata,
            stage_key=stage_key,
            stage_map=stage_map,
            sample_size=config.data.sample_size,
            seed=config.seed,
        )

    return splits, adata

def _lock_uot_hparams(
    splits: ConditionSplits,
    config: CTOTSUConfig,
    sample_size: int = 512,
) -> dict:
    rng = np.random.default_rng(config.seed)
    n_src = splits.X0.shape[0]
    n_dst = splits.Xt2.shape[0]
    if n_src == 0 or n_dst == 0:
        return {}
    sample = min(sample_size, n_src, n_dst)
    if sample < 2:
        return {}
    idx_src = rng.choice(n_src, sample, replace=False)
    idx_dst = rng.choice(n_dst, sample, replace=False)
    results = tau_eps_grid(
        splits.X0[idx_src],
        splits.Xt2[idx_dst],
        taus=config.hyperparam.tau_grid,
        epsilons=config.hyperparam.reg_grid,
        reg_ms=config.hyperparam.reg_m_grid,
        repeats=config.hyperparam.repeats,
    )
    plateau = results.get('plateau') or []
    candidates = plateau if plateau else results.get('grid', [])
    if not candidates:
        return {}
    locked = min(candidates, key=lambda item: (item[3], -item[2], -item[1], item[0]))
    tau_lock, eps_lock, reg_m_lock, mean_err, se_err = locked
    config.uot.tau = float(tau_lock)
    config.uot.eps = float(eps_lock)
    config.uot.reg_m = float(reg_m_lock)
    return {
        'grid': results.get('grid'),
        'plateau': plateau,
        'locked': {
            'tau': float(tau_lock),
            'eps': float(eps_lock),
            'reg_m': float(reg_m_lock),
            'mean_error': float(mean_err),
            'se_error': float(se_err),
            'one_se_threshold': float(results.get('one_se', mean_err + se_err)),
        },
    }


def _aggregate_responsibilities(
    resp: np.ndarray,
    labels_enforced: np.ndarray,
    labels_merged: np.ndarray,
    K_final: int,
) -> np.ndarray:
    mapping: dict[int, int] = {}
    for old_label, new_label in zip(labels_enforced, labels_merged):
        mapping.setdefault(int(old_label), int(new_label))

    resp_merged = np.zeros((resp.shape[0], K_final), dtype=np.float64)
    for old_label, new_label in mapping.items():
        if old_label < resp.shape[1]:
            resp_merged[:, new_label] += resp[:, old_label]

    row_sums = resp_merged.sum(axis=1)
    mask = row_sums > 0
    if np.any(mask):
        resp_merged[mask] /= row_sums[mask][:, None]
    if np.any(~mask):
        resp_merged[~mask] = 1.0 / K_final
    return resp_merged

def _fit_gating_model(
    Xt: np.ndarray,
    config: CTOTSUConfig,
    domain_labels: np.ndarray | None = None,
):
    Xt64 = np.asarray(Xt, dtype=np.float64, order='C')
    try:
        best, table = fit_gmm_BICICL(
            Xt64,
            k_grid=config.gate.k_grid,
            mode=config.gate.mode,
            random_state=config.seed,
            covariance_type=config.gate.covariance_type,
            reg_covar=config.gate.reg_covar,
            max_iter=config.gate.gmm_max_iter,
            n_init=config.gate.gmm_n_init,
            weight_concentration_prior=config.gate.weight_concentration_prior,
        )
    except ValueError as exc:
        raise RuntimeError(f"GMM model selection failed: {exc}") from exc

    if best is None:
        raise RuntimeError('GMM model selection failed')

    scores = {
        int(k): {
            'bic': float(bic),
            'icl': float(icl),
        }
        for k, bic, icl, _ in table
    }

    K_initial = int(best[0])
    gmm = best[2]
    resp = gmm.predict_proba(Xt64)

    try:
        bic_retrained = float(gmm.bic(Xt64))
    except AttributeError:
        bic_retrained = float(-2.0 * gmm.score(Xt64) * Xt64.shape[0])
    entropy_penalty = float(np.sum(resp * np.log(resp + 1e-12)))
    scores.setdefault(K_initial, {})
    scores[K_initial]['bic_retrained'] = bic_retrained
    scores[K_initial]['icl_retrained'] = bic_retrained - entropy_penalty

    labels_enforced, min_frac_enforced = enforce_min_cluster(
        gmm,
        Xt64,
        min_frac=config.gate.min_cluster_frac,
    )
    labels_merged, K_final = merge_by_w2(
        Xt64,
        labels_enforced,
        w_thresh=config.gate.wmerge_thresh,
    )

    counts_final = np.bincount(labels_merged, minlength=K_final)
    min_frac_actual = float(counts_final.min() / labels_merged.size) if counts_final.size else 0.0

    resp_merged = _aggregate_responsibilities(resp, labels_enforced, labels_merged, K_final)

    if config.train.use_dann and domain_labels is not None:
        from ct_ots_u.model.dann import DomainAdversarialGating  # pragma: no cover - optional path

        gating_model = DomainAdversarialGating(
            n_gates=K_final,
            lambda_domain=config.train.lambda_domain,
        )
        gate_labels = np.argmax(resp_merged, axis=1)
        gating_model.fit(Xt64, gate_labels, domain_labels)
        if config.verbose:
            print(f"[*] DANN gating trained with {len(np.unique(domain_labels))} domains")
    else:
        gating_model = train_gating(
            Xt64,
            resp_merged,
            C=config.gate.gating_C,
            max_iter=config.gate.gating_max_iter,
        )

    info = {
        'mode': config.gate.mode,
        'weight_concentration_prior': config.gate.weight_concentration_prior,
        'bic_icl_scores': scores,
        'K_initial': int(K_initial),
        'K_after_merge': int(K_final),
        'min_cluster_fraction_enforced': float(min_frac_enforced),
        'min_cluster_fraction': min_frac_actual,
    }
    return gating_model, info

def _train_branch_generators(
    splits: ConditionSplits,
    gating_model,
    config: CTOTSUConfig,
    *,
    use_torch_compile: bool,
    sinkhorn_minibatch: bool,
    sinkhorn_batch_size: int | None,
):
    branch_min = max(config.data.sample_size // 10 if config.data.sample_size > 0 else 64, 32)
    r_pick, r_best, rank_stats, rank_conf = rank_selection_cv(
        splits.X0,
        splits.Xt1,
        splits.Xt2,
        splits.Xt12,
        gating_model,
        ranks=list(config.rank.grid),
        reg=config.uot.eps,
        reg_m=config.uot.reg_m,
        tau=config.uot.tau,
        top_p=config.gate.top_p,
        repeats=config.rank.cv_repeats,
        min_branch=branch_min,
        seed=config.seed,
        num_iter_max=config.uot.max_iter,
        stop_thr=config.uot.stop_thr,
        use_one_se=config.rank.one_se,
        train_steps=config.train.steps,
        train_lr=config.train.lr,
        stable_alpha=config.stable.alpha,
        stable_margin=config.stable.max_spectral_margin,
        stable_lambda=config.stable.lambda_,
        stable_soft_weight=config.stable.soft_weight,
        stable_use_soft_penalty=config.stable.use_soft_penalty,
        lambda_soft=config.stable.lambda_soft,
        reg_nuc=0.0,
        use_torch=config.train.use_torch,
        use_torch_compile=use_torch_compile,
        device=config.train.device,
        backend=config.uot.backend,
        sinkhorn_backend=config.uot.sinkhorn_backend,
        sinkhorn_scaling=config.uot.sinkhorn_scaling,
        sinkhorn_minibatch=sinkhorn_minibatch,
        sinkhorn_batch_size=sinkhorn_batch_size,
        dynamics_cfg=config.dynamics,
        regular_cfg=config.regular,
        align_cfg=config.align,
    )

    if r_pick is None:
        r_pick = max(config.rank.grid)

    model = GatedSemigroup(
        K=gating_model.classes_.shape[0],
        rank=r_pick,
        reg=config.uot.eps,
        reg_m=config.uot.reg_m,
        stable_margin=config.stable.max_spectral_margin,
        stable_alpha=config.stable.alpha,
        stable_lambda=config.stable.lambda_,
        stable_soft_weight=config.stable.soft_weight,
        stable_use_soft_penalty=config.stable.use_soft_penalty,
        stable_lambda_soft=config.stable.lambda_soft,
        ot_backend=config.uot.backend,
        sinkhorn_backend=config.uot.sinkhorn_backend,
        sinkhorn_scaling=config.uot.sinkhorn_scaling,
        min_branch_samples=branch_min,
        tau=config.uot.tau,
        num_iter_max=config.uot.max_iter,
        stop_thr=config.uot.stop_thr,
        train_steps=config.train.steps,
        train_lr=config.train.lr,
        use_swa=config.train.use_swa,
        swa_start_ratio=config.train.swa_start_ratio,
        swa_lr=config.train.swa_lr,
        swa_schedule=config.train.swa_schedule,
        reg_nuc=0.0,
        use_torch=config.train.use_torch,
        use_torch_compile=use_torch_compile,
        device=config.train.device,
        log_diagnostics=config.stable.log_raw_stability,
        sinkhorn_minibatch=sinkhorn_minibatch,
        sinkhorn_batch_size=sinkhorn_batch_size,
        dynamics_cfg=config.dynamics,
        regular_cfg=config.regular,
        align_cfg=config.align,
    ).fit_branchwise(
        splits.X0,
        splits.Xt1,
        gating_model=gating_model,
        top_p=config.gate.top_p,
    )

    return model, {
        'r_pick': r_pick,
        'r_best': r_best,
        'rank_stats': rank_stats,
        'rank_conf': rank_conf,
    }

def _evaluate_consistency(
    model: GatedSemigroup,
    splits: ConditionSplits,
    config: CTOTSUConfig,
    disable_projection: bool,
) -> dict:
    report = semigroup_consistency(
        model,
        splits.X0,
        splits.Xt1,
        splits.Xt2,
        splits.Xt12,
        reg=config.uot.eps,
        clip_eps=config.train.clip_neg_dist_eps,
        eval_disable_projection=disable_projection,
    )
    return {
        'mid_error': report.mid_error,
        'two_step_error': report.two_step_error,
        'direct_error': report.direct_error,
        'consistency_error': report.consistency_error,
        'improvement': report.improvement,
        'stability_penalty': report.stability_penalty,
        'clip_eps': config.train.clip_neg_dist_eps,
        'eval_disable_projection': disable_projection,
    }


def _save_model_artifacts(out_path: Path, model: GatedSemigroup) -> None:
    ensure_dir(out_path.parent)
    payload = {
        'L': [L.tolist() for L in model.L_list],
        'bias': [b.tolist() for b in model.bias_list],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _stability_summary(model: GatedSemigroup, config: CTOTSUConfig) -> dict:
    diagnostics = getattr(model, 'stability_diagnostics', []) or []
    per_branch = [dict(entry) for entry in diagnostics if isinstance(entry, dict)]

    def _safe_max(key: str):
        values = [entry.get(key) for entry in per_branch if key in entry]
        return max(values) if values else None

    total_violation = float(
        sum(entry.get('violation', 0.0) for entry in per_branch)
    )

    return {
        'margin': config.stable.max_spectral_margin,
        'lambda': config.stable.lambda_,
        'log_raw_stability': config.stable.log_raw_stability,
        'eval_disable_projection': config.stable.eval_disable_projection,
        'clip_neg_eps': config.train.clip_neg_dist_eps,
        'max_alpha_raw': _safe_max('alpha_raw'),
        'max_mu2_raw': _safe_max('mu2_raw'),
        'max_alpha_proj': _safe_max('alpha_proj'),
        'max_mu2_proj': _safe_max('mu2_proj'),
        'total_violation': total_violation,
        'per_branch': per_branch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='CT-OTS-U Training Script')
    script_dir = Path(__file__).resolve().parents[2]
    default_h5ad = script_dir / 'data' / 'raw-use' / 'GSE160936' / 'processed' / 'GSE160936_microglia_processed.h5ad'
    default_out = script_dir / 'results' / 'gse160936_microglia' / 'train_model.json'
    default_summary = script_dir / 'results' / 'gse160936_microglia' / 'train_summary.json'

    parser.add_argument('--h5ad', default=str(default_h5ad), help='Input h5ad file')
    parser.add_argument('--stage_key', default='braak_stage')
    parser.add_argument('--src_stage', default='0,1')
    parser.add_argument('--mid_stage', default='2,3,4')
    parser.add_argument('--dst_stage', default='5,6')
    parser.add_argument('--celltype_key')
    parser.add_argument('--celltype')
    parser.add_argument('--config', help='JSON file of configuration overrides')
    parser.add_argument('--pct-mt-limit', type=float, help='Override mitochondrial percentage QC threshold.')
    parser.add_argument('--use-torch', action='store_true', help='Enable PyTorch backend for branch generators.')
    parser.add_argument('--torch-device', help='Torch device string, e.g. cpu, cuda, cuda:0.')
    parser.add_argument('--torch-compile', action='store_true', help='Use torch.compile() if available for training step acceleration.')
    parser.add_argument('--sinkhorn-minibatch', action='store_true', help='Use minibatch approximation of Sinkhorn loss for gradients (keeps expectation).')
    parser.add_argument('--sinkhorn-batch-size', type=int, default=0, help='Minibatch size for Sinkhorn gradient (0=auto).')
    parser.add_argument('--msigdb_gmt', nargs='*')
    parser.add_argument('--out', default=str(default_out))
    parser.add_argument('--summary_out', default=str(default_summary))
    parser.add_argument('--stable-margin', type=float, help='Override stability margin (negative).')
    parser.add_argument('--stable-lambda', type=float, help='Multiplier for stability penalty.')
    parser.add_argument('--stable-soft-weight', type=float, help='Fraction (0-1) of hard projection applied each iteration.')
    parser.add_argument('--log-raw-stability', action='store_true', help='Record raw/projection stability diagnostics during training.')
    parser.add_argument('--clip-neg-eps', type=float, help='Numerical floor applied to Sinkhorn distances.')
    parser.add_argument('--eval-no-projection', action='store_true', help='Disable stability projections when running consistency checks.')
    parser.add_argument('--ot-backend', choices=['geomloss', 'pot'], help='Optimal transport backend to use.')
    parser.add_argument('--sinkhorn-backend', help='Sinkhorn backend when using geomloss (e.g. online, multiscale).')
    parser.add_argument('--sinkhorn-scaling', type=float, help='Scaling factor for epsilon annealing in Sinkhorn solvers.')
    parser.add_argument('--uot-regm', type=float, help='Unbalanced Sinkhorn marginal relaxation parameter.')
    parser.add_argument('--gating-mode', choices=['gaussian', 'bayesian'], help='Mixture model type for gating discovery.')
    parser.add_argument('--min-cluster-frac', type=float, help='Minimum fraction threshold for cluster enforcement.')
    parser.add_argument('--merge-wass', type=float, help='Wasserstein threshold for cluster merging.')
    parser.add_argument('--weight-concentration', type=float, help='Dirichlet-process concentration prior for Bayesian GMM.')
    parser.add_argument('--gmm-max-iter', type=int, help='Maximum EM iterations for mixture fitting.')
    parser.add_argument('--gmm-n-init', type=int, help='Number of initialisations for Gaussian mixture mode.')
    parser.add_argument('--use-swa', action='store_true', help='Enable Stochastic Weight Averaging.')
    parser.add_argument('--swa-start-ratio', type=float, default=0.8, help='Start SWA at this fraction of total epochs.')
    parser.add_argument('--swa-lr', type=float, default=1e-3, help='Learning rate during SWA.')
    parser.add_argument('--swa-schedule', choices=['constant', 'cosine'], default='constant', help='SWA learning rate schedule.')
    parser.add_argument('--use-soft-penalty', action='store_true', help='Enable soft (quadratic) stability penalty.')
    parser.add_argument('--lambda-soft', type=float, default=0.5, help='Soft penalty weight.')
    parser.add_argument('--use-dann', action='store_true', help='Enable domain adversarial gating.')
    parser.add_argument('--lambda-domain', type=float, default=0.1, help='Domain adversarial weight.')
    parser.add_argument('--domain-key', type=str, help='Column name for domain labels (e.g., donor_id).')
    parser.add_argument('--use-harmony', action='store_true', help='Enable Harmony batch integration.')
    parser.add_argument('--harmony-theta', type=float, default=2.0, help='Harmony diversity penalty.')
    parser.add_argument('--harmony-lambda', type=float, default=1.0, help='Harmony ridge regularisation.')
    parser.add_argument('--batch-key', type=str, help='Column name for batch labels.')
    parser.add_argument('--harmony-backend', choices=['harmonypy', 'scanpy'], default='harmonypy', help='Harmony backend.')
    parser.add_argument('--use-batch-decision', action='store_true', help='Auto-decide whether Harmony should be applied via scIB deltas.')
    parser.add_argument('--batch-delta-thresh', type=float, default=0.03, help='scIB batch delta threshold.')
    parser.add_argument('--bio-drop-thresh', type=float, default=0.02, help='scIB biological drop threshold.')
    parser.add_argument('--use-plateau-lock', action='store_true', help='Enable UOT plateau locking with 1-SE rule.')
    parser.add_argument('--plateau-prefer-small-tau', action='store_true', default=True, help='Prefer smaller tau within plateau.')
    parser.add_argument('--use-temperature-scaling', action='store_true', help='Enable temperature scaling for gate calibration.')
    parser.add_argument('--temp-scale-target', type=str, help='Stage label used for temperature scaling calibration.')
    parser.add_argument('--use-cross-domain-uot', action='store_true', help='Enable cross-domain UOT plateau check.')
    parser.add_argument('--target-stage-key', type=str, help='Stage key for target dataset in cross-domain checks.')
    parser.add_argument('--target-stages', type=str, help='Alternate target stage mapping string (e.g. early:Control;late:PD).')
    parser.add_argument('--target-h5ad', type=str, help='Target dataset h5ad for cross-domain checks.')
    parser.add_argument('--target-stage-map', type=str, help='JSON string describing stage map for target dataset.')
    parser.add_argument('--use-ood-detection', action='store_true', help='Enable Mahalanobis OOD detection on gating embeddings.')
    parser.add_argument('--ood-percentile', type=float, default=95.0, help='Percentile threshold for OOD detection.')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    config = CTOTSUConfig.from_env()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as fh:
            overrides = json.load(fh)
        for section, values in overrides.items():
            if hasattr(config, section):
                target = getattr(config, section)
                if isinstance(values, dict):
                    for key, value in values.items():
                        if hasattr(target, key):
                            setattr(target, key, value)
                else:
                    setattr(config, section, values)

    if args.verbose:
        config.verbose = True

    if args.use_torch:
        config.train.use_torch = True
    if args.torch_device:
        config.train.device = args.torch_device

    if args.use_swa:
        config.train.use_swa = True
        config.train.swa_start_ratio = args.swa_start_ratio
        config.train.swa_lr = args.swa_lr
        config.train.swa_schedule = args.swa_schedule
    if args.use_soft_penalty:
        config.stable.use_soft_penalty = True
        config.stable.lambda_soft = args.lambda_soft
    if args.use_dann:
        config.train.use_dann = True
        config.train.lambda_domain = args.lambda_domain
    if args.use_harmony:
        config.train.use_harmony = True
        config.train.harmony_theta = args.harmony_theta
        config.train.harmony_lambda = args.harmony_lambda
    if args.use_plateau_lock:
        config.train.use_plateau_lock = True
        config.train.plateau_prefer_small_tau = args.plateau_prefer_small_tau

    if args.stable_margin is not None:
        config.stable.max_spectral_margin = args.stable_margin
    if args.stable_lambda is not None:
        config.stable.lambda_ = args.stable_lambda
    if args.stable_soft_weight is not None:
        config.stable.soft_weight = max(0.0, min(1.0, args.stable_soft_weight))
    if args.log_raw_stability:
        config.stable.log_raw_stability = True
    if args.clip_neg_eps is not None:
        config.train.clip_neg_dist_eps = args.clip_neg_eps
    if args.eval_no_projection:
        config.stable.eval_disable_projection = True
    if args.ot_backend:
        config.uot.backend = args.ot_backend
    if args.sinkhorn_backend:
        config.uot.sinkhorn_backend = args.sinkhorn_backend
    if args.sinkhorn_scaling is not None:
        config.uot.sinkhorn_scaling = args.sinkhorn_scaling
    if args.uot_regm is not None:
        config.uot.reg_m = args.uot_regm
    if args.gating_mode:
        config.gate.mode = args.gating_mode
    if args.min_cluster_frac is not None:
        config.gate.min_cluster_frac = args.min_cluster_frac
    if args.merge_wass is not None:
        config.gate.wmerge_thresh = args.merge_wass
    if args.weight_concentration is not None:
        config.gate.weight_concentration_prior = args.weight_concentration
    if args.gmm_max_iter is not None:
        config.gate.gmm_max_iter = args.gmm_max_iter
    if args.gmm_n_init is not None:
        config.gate.gmm_n_init = args.gmm_n_init

    stage_map = _build_stage_map(args)

    if args.msigdb_gmt:
        for gmt_path in args.msigdb_gmt:
            load_gmt(Path(gmt_path))

    splits, processed = _preprocess_data(
        args.h5ad,
        config,
        args.stage_key,
        stage_map,
        args.celltype_key,
        args.celltype,
        pct_mt_limit=args.pct_mt_limit,
        batch_key=args.batch_key,
        harmony_backend=args.harmony_backend,
        use_batch_decision=bool(args.use_batch_decision),
        batch_delta_thresh=args.batch_delta_thresh,
        bio_drop_thresh=args.bio_drop_thresh,
    )

    uot_plateau = _lock_uot_hparams(splits, config) if config.train.use_plateau_lock else {}
    if args.verbose and uot_plateau:
        locked = uot_plateau.get('locked', {})
        print(
            "[*] Locked UOT: "
            f"tau={locked.get('tau')}, eps={locked.get('eps')}, reg_m={locked.get('reg_m')} "
            f"(mean={locked.get('mean_error')}, se={locked.get('se_error')})"
        )

    domain_labels = None
    if config.train.use_dann and args.domain_key:
        if args.domain_key not in processed.obs.columns:
            print(f"[!] Warning: domain_key '{args.domain_key}' not found, DANN disabled")
            config.train.use_dann = False
        else:
            late_stages = set(stage_map['late'])
            mask = processed.obs[args.stage_key].astype(str).isin(late_stages)
            domain_labels = processed.obs.loc[mask, args.domain_key].values[: len(splits.Xt2)]

    gating_model, gating_info = _fit_gating_model(splits.Xt2, config, domain_labels=domain_labels)

    temp_scaling_info = None
    if args.use_temperature_scaling and calibrate_temperature is not None:
        gate_probs = gating_model.predict_proba(splits.Xt2)
        gate_logits = np.log(gate_probs + 1e-12)
        gate_labels = gating_model.predict(splits.Xt2)
        if args.temp_scale_target:
            target_mask = processed.obs[args.stage_key].astype(str) == args.temp_scale_target
            if target_mask.sum() > 0:
                target_X = processed.obsm['X_pca'][target_mask]
                target_logits = np.log(gating_model.predict_proba(target_X) + 1e-12)
                target_labels = gating_model.predict(target_X)
                calib_result = calibrate_temperature(target_logits, target_labels)
                temp_scaling_info = {
                    'method': 'target_domain',
                    'target_stage': args.temp_scale_target,
                    **calib_result,
                }
        if temp_scaling_info is None:
            calib_result = calibrate_temperature(gate_logits, gate_labels)
            temp_scaling_info = {
                'method': 'late_stage',
                **calib_result,
            }
        if args.verbose and temp_scaling_info:
            print(f"[*] Temperature scaling applied: T={temp_scaling_info.get('temperature', 1.0):.3f}")

    torch_compile_flag = bool(args.torch_compile)
    sinkhorn_minibatch_flag = bool(args.sinkhorn_minibatch)
    sinkhorn_batch_size_val = None if args.sinkhorn_batch_size in (None, 0) else int(args.sinkhorn_batch_size)

    model, rank_info = _train_branch_generators(
        splits,
        gating_model,
        config,
        use_torch_compile=torch_compile_flag,
        sinkhorn_minibatch=sinkhorn_minibatch_flag,
        sinkhorn_batch_size=sinkhorn_batch_size_val,
    )
    consistency_info = _evaluate_consistency(model, splits, config, args.eval_no_projection)
    stability_info = _stability_summary(model, config)

    ood_info = None
    if args.use_ood_detection and MahalanobisOOD is not None:
        gate_labels_train = gating_model.predict(splits.Xt2)
        ood_detector = MahalanobisOOD()
        ood_detector.fit(splits.Xt2, gate_labels_train)
        base_scores = ood_detector.score(splits.Xt2)
        if hasattr(ood_detector, 'compute_threshold'):
            base_threshold = float(
                ood_detector.compute_threshold(
                    splits.Xt2,
                    percentile=args.ood_percentile,
                )
            )
        else:
            base_threshold = float(np.percentile(base_scores, args.ood_percentile))

        ood_scores = {}
        for stage_name, X_stage in [('early', splits.X0), ('mid', splits.Xt1), ('late', splits.Xt2)]:
            scores = ood_detector.score(X_stage)
            threshold = base_threshold
            is_ood = scores > threshold
            ood_scores[stage_name] = {
                'mean_score': float(scores.mean()),
                'max_score': float(scores.max()),
                'threshold': threshold,
                'ood_fraction': float(is_ood.mean()),
                'n_ood': int(is_ood.sum()),
                'n_total': int(len(scores)),
            }
        ood_info = {
            'percentile': args.ood_percentile,
            'scores_by_stage': ood_scores,
        }

    cross_domain_info = None
    if args.use_cross_domain_uot and args.target_h5ad:
        if not os.path.exists(args.target_h5ad):
            print(f"[!] Warning: target h5ad '{args.target_h5ad}' not found; skipping cross-domain check")
        else:
            target_stage_map = stage_map
            if args.target_stage_map:
                try:
                    target_stage_map = json.loads(args.target_stage_map)
                except json.JSONDecodeError as exc:
                    raise ValueError('target-stage-map must be valid JSON') from exc
            elif args.target_stages:
                parsed = _parse_target_stage_mapping(args.target_stages)
                if parsed:
                    target_stage_map = {
                        'early': parsed.get('early', stage_map.get('early', [])),
                        'mid': parsed.get('mid', stage_map.get('mid', [])),
                        'late': parsed.get('late', stage_map.get('late', [])),
                    }
            target_stage_key = args.target_stage_key or args.stage_key
            target_splits, _ = _preprocess_data(
                args.target_h5ad,
                config,
                target_stage_key,
                target_stage_map,
                args.celltype_key,
                args.celltype,
                pct_mt_limit=args.pct_mt_limit,
                batch_key=None,
                harmony_backend='harmonypy',
                use_batch_decision=False,
                batch_delta_thresh=args.batch_delta_thresh,
                bio_drop_thresh=args.bio_drop_thresh,
            )
            source_plateau = uot_plateau.get('plateau') if uot_plateau else None
            if source_plateau:
                cross_domain_info = cross_domain_plateau_check(
                    source_plateau,
                    target_splits.X0,
                    target_splits.Xt2,
                )

    summary = {
        'stage_map': stage_map,
        'gating': gating_info,
        'rank': rank_info,
        'uot_plateau': uot_plateau,
        'consistency': consistency_info,
        'stability': stability_info,
        'msigdb': {'gmt_files': args.msigdb_gmt or []},
        'config_used': config.to_dict(),
    }
    if temp_scaling_info:
        summary['temperature_scaling'] = temp_scaling_info
    if cross_domain_info:
        summary['cross_domain_uot'] = cross_domain_info
    if ood_info:
        summary['ood_detection'] = ood_info

    summary_path = Path(args.summary_out)
    ensure_dir(summary_path.parent)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    model_path = Path(args.out)
    _save_model_artifacts(model_path, model)

    if args.verbose:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print('Training complete!')
        print(f"Model saved to: {model_path}")
        print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()

