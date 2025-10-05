#!/usr/bin/env python3
"""CT-OTS-U validation script with MSigDB pathway analysis."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ct_ots_u.config import CTOTSUConfig
from ct_ots_u.data import (
    ConditionSplits,
    extract_condition_arrays,
    make_pseudotime_bins,
    preprocess_adata,
)
from ct_ots_u.metric_msigdb import load_gmt
from ct_ots_u.ot import uot_sinkhorn_cost
from ct_ots_u.ot.metrics import sinkhorn_divergence_bal
from ct_ots_u.pathways import pathway_enrichment_analysis
from ct_ots_u.stats_eval import bootstrap_ci

import anndata as ad
import pandas as pd


Y_GENES = ['RPS4Y1', 'EIF1AY', 'DDX3Y', 'KDM5D', 'UTY']
HK_GENES = ['RPLP0', 'RPL13', 'RPS18', 'GAPDH', 'ACTB']


def _parse_stage_arg(value: str) -> Sequence[str]:
    parts = [token.strip() for token in value.split(',') if token.strip()]
    if not parts:
        raise ValueError('Stage arguments must contain at least one label')
    return parts


def _stage_map_from_args(args) -> dict:
    return {
        'early': _parse_stage_arg(args.src_stage),
        'mid': _parse_stage_arg(args.mid_stage),
        'late': _parse_stage_arg(args.dst_stage),
    }


def _extract_condition_splits(
    raw_adata: ad.AnnData,
    config: CTOTSUConfig,
    stage_key: str,
    stage_map: dict,
    celltype_key: str | None,
    celltype: str | None,
    pct_mt_limit: float | None = None,
) -> tuple[ConditionSplits, ad.AnnData, ad.AnnData]:
    adata = raw_adata.copy()
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
    if config.verbose:
        print(
            f"[*] Validation QC thresholds: min_counts={config.data.min_counts}, "
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
        raw_subset = raw_adata[mask].copy()
    else:
        raw_subset = raw_adata

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

    return splits, adata, raw_subset


def _compute_uot_metrics(splits: ConditionSplits, config: CTOTSUConfig) -> dict:
    uot_cost, _, _ = uot_sinkhorn_cost(
        splits.X0,
        splits.Xt2,
        reg=config.uot.eps,
        reg_m=config.uot.reg_m,
        backend=config.uot.backend,
        sinkhorn_backend=config.uot.sinkhorn_backend,
        scaling=config.uot.sinkhorn_scaling,
    )
    bal_cost = sinkhorn_divergence_bal(
        splits.X0,
        splits.Xt2,
        reg=config.uot.eps,
        backend=config.uot.backend,
        sinkhorn_backend=config.uot.sinkhorn_backend,
        scaling=config.uot.sinkhorn_scaling,
        num_iter_max=config.uot.max_iter,
        stop_thr=config.uot.stop_thr,
    )
    delta = uot_cost - bal_cost

    def _stat_fn(X_sample, Y_sample):
        cost, _, _ = uot_sinkhorn_cost(
            X_sample,
            Y_sample,
            reg=config.uot.eps,
            reg_m=config.uot.reg_m,
            backend=config.uot.backend,
            sinkhorn_backend=config.uot.sinkhorn_backend,
            scaling=config.uot.sinkhorn_scaling,
        )
        bal = sinkhorn_divergence_bal(
            X_sample,
            Y_sample,
            reg=config.uot.eps,
            backend=config.uot.backend,
            sinkhorn_backend=config.uot.sinkhorn_backend,
            scaling=config.uot.sinkhorn_scaling,
            num_iter_max=config.uot.max_iter,
            stop_thr=config.uot.stop_thr,
        )
        return cost - bal

    mean_delta, (ci_lo, ci_hi) = bootstrap_ci(
        _stat_fn,
        splits.X0,
        splits.Xt2,
        n_resamples=100,
        seed=config.seed,
    )

    passed = ci_hi < -config.eval_tol.delta_tolerance

    return {
        'uot_cost': float(uot_cost),
        'balanced_cost': float(bal_cost),
        'delta': float(delta),
        'delta_ci': [float(ci_lo), float(ci_hi)],
        'pass': passed,
        'status': 'PASS' if passed else 'FAIL',
        'reasoning': 'UOT cost significantly lower than balanced cost'
        if passed
        else 'UOT improvement not significant',
    }


def _load_gene_sets(gmt_paths: Sequence[str]) -> dict:
    gene_sets = {}
    for gmt in gmt_paths:
        path = Path(gmt)
        sets = load_gmt(path)
        for name, genes in sets.items():
            key = f"{path.stem}:{name}" if name not in gene_sets else f"{path.stem}:{name}"
            gene_sets[key] = genes
    return gene_sets


def _compute_differential_genes(
    adata: ad.AnnData,
    stage_key: str,
    stage_map: dict,
    top_n: int = 200,
) -> tuple[list[str], list[str]]:
    stage_series = adata.obs[stage_key].astype(str)
    early_mask = stage_series.isin(stage_map['early']).values
    late_mask = stage_series.isin(stage_map['late']).values

    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    if early_mask.sum() == 0 or late_mask.sum() == 0:
        raise ValueError('Insufficient cells to compute differential genes for pathway analysis')

    mean_early = X[early_mask].mean(axis=0)
    mean_late = X[late_mask].mean(axis=0)

    diff = mean_late - mean_early
    top_n = min(top_n, adata.n_vars)
    top_idx = np.argsort(np.abs(diff))[-top_n:]
    differential = adata.var_names[top_idx]
    return differential.tolist(), adata.var_names.tolist()


def _pathway_analysis(
    adata: ad.AnnData,
    stage_key: str,
    stage_map: dict,
    gmt_paths: Sequence[str],
    alpha: float,
) -> dict:
    gene_sets = _load_gene_sets(gmt_paths)
    diff_genes, background = _compute_differential_genes(adata, stage_key, stage_map)
    results_df = pathway_enrichment_analysis(diff_genes, background, gene_sets, alpha=alpha)
    if results_df is None or results_df.empty:
        return {
            'status': 'WARN',
            'pass': False,
            'reason': 'No pathways passed enrichment criteria',
            'top_pathways': [],
        }
    results_df = results_df.sort_values('p_value')
    signif = results_df[results_df['adjusted_p_value'] <= alpha]
    top_rows = results_df.head(10)
    top_pathways = [
        {
            'pathway': row['pathway'],
            'overlap_genes': int(row['overlap_genes']),
            'p_value': float(row['p_value']),
            'adjusted_p_value': float(row['adjusted_p_value']),
            'fold_enrichment': float(row['fold_enrichment']),
        }
        for _, row in top_rows.iterrows()
    ]
    return {
        'status': 'PASS' if not signif.empty else 'WARN',
        'pass': not signif.empty,
        'tested_pathways': int(len(results_df)),
        'significant_pathways': int(len(signif)),
        'top_pathways': top_pathways,
    }


def _compute_mloy_statistics(raw_adata: ad.AnnData) -> dict:
    genes_upper = raw_adata.var_names.str.upper()
    y_idx = [i for i, g in enumerate(genes_upper) if g in set(Y_GENES)]
    hk_idx = [i for i, g in enumerate(genes_upper) if g in set(HK_GENES)]

    if not y_idx or not hk_idx:
        return {
            'status': 'FAIL',
            'pass': False,
            'reason': 'Required Y or housekeeping genes missing',
        }

    X = raw_adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    y_score = X[:, y_idx].mean(axis=1) - X[:, hk_idx].mean(axis=1)

    sexes = raw_adata.obs.get('sex', pd.Series(['unknown'] * raw_adata.n_obs))
    sexes = sexes.astype(str).str.lower()

    male_scores = y_score[sexes == 'male']
    female_scores = y_score[sexes == 'female']

    if female_scores.size == 0 or male_scores.size == 0:
        return {
            'status': 'FAIL',
            'pass': False,
            'reason': 'Insufficient male or female cells for mLOY diagnostics',
        }

    threshold = float(np.quantile(female_scores, 0.99))
    fraction = float(np.mean(male_scores < threshold))

    return {
        'status': 'PASS',
        'pass': True,
        'threshold': threshold,
        'male_fraction_below_threshold': fraction,
        'n_male_cells': int(male_scores.size),
        'n_female_cells': int(female_scores.size),
    }


def make_final_decision(results: dict) -> str:
    core_pass = results['H1_uot_advantage']['pass'] and results['H2_H3_semigroup_consistency']['pass']
    gating_pass = results['H4_cluster_lockdown']['pass']
    return 'MODEL_GO' if core_pass and gating_pass else 'REVISION'


def main() -> dict:
    parser = argparse.ArgumentParser(description='CT-OTS-U Model Validation')
    parser.add_argument('--summary', required=True, help='Training summary JSON')
    parser.add_argument('--h5ad', required=True, help='Input h5ad used during training')
    parser.add_argument('--stage_key', default='braak_stage')
    parser.add_argument('--src_stage', default='0,1')
    parser.add_argument('--mid_stage', default='2,3,4')
    parser.add_argument('--dst_stage', default='5,6')
    parser.add_argument('--celltype_key')
    parser.add_argument('--celltype')
    parser.add_argument('--pct-mt-limit', type=float, default=None, help='Override mitochondrial percentage QC threshold.')
    parser.add_argument('--config')
    parser.add_argument('--msigdb_gmt', nargs='*', help='MSigDB GMT files for pathway enrichment')
    parser.add_argument('--out')
    parser.add_argument('--ot-backend', choices=['geomloss', 'pot'], help='Optimal transport backend to use during validation.')
    parser.add_argument('--sinkhorn-backend', help='Sinkhorn backend when using geomloss (e.g. online, multiscale).')
    parser.add_argument('--sinkhorn-scaling', type=float, help='Scaling factor for epsilon annealing in Sinkhorn solvers.')
    parser.add_argument('--uot-regm', type=float, help='Override unbalanced marginal relaxation parameter during validation.')
    parser.add_argument('--gating-mode', choices=['gaussian', 'bayesian'], help='Mixture model type for gating diagnostics.')
    parser.add_argument('--min-cluster-frac', type=float, help='Minimum fraction threshold when analysing gating stability.')
    parser.add_argument('--merge-wass', type=float, help='Wasserstein threshold for merging clusters in validation.')
    parser.add_argument('--weight-concentration', type=float, help='Dirichlet-process concentration prior for Bayesian gating.')
    parser.add_argument('--gmm-max-iter', type=int, help='Maximum EM iterations for mixture fitting during validation.')
    parser.add_argument('--gmm-n-init', type=int, help='Number of initialisations for Gaussian mixture mode during validation.')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--eval-no-projection', action='store_true', help='Disable stability projection checks when recomputing metrics.')
    parser.add_argument('--clip-neg-eps', type=float, help='Numerical floor applied to Sinkhorn distances during evaluation.')

    args = parser.parse_args()

    config = CTOTSUConfig.from_env()
    if args.config and os.path.exists(args.config):
        with open(args.config) as fh:
            overrides = json.load(fh)
        if 'seed' in overrides:
            config.seed = overrides['seed']

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
    if args.clip_neg_eps is not None:
        config.train.clip_neg_dist_eps = args.clip_neg_eps

    with open(args.summary, 'r', encoding='utf-8') as fh:
        summary = json.load(fh)

    stage_map = summary.get('stage_map') or _stage_map_from_args(args)

    raw_adata = ad.read_h5ad(args.h5ad)
    splits, processed, raw_subset = _extract_condition_splits(
        raw_adata,
        config,
        args.stage_key,
        stage_map,
        args.celltype_key,
        args.celltype,
        pct_mt_limit=args.pct_mt_limit,
    )

    if args.verbose:
        print('Validation stage map:', stage_map)
        print('Data shapes:', splits.X0.shape, splits.Xt1.shape, splits.Xt2.shape)

    h1_result = _compute_uot_metrics(splits, config)
    h2h3_result = summary.get('consistency', {})
    if not h2h3_result:
        h2h3_result = {'status': 'SKIP', 'pass': False}
    else:
        if 'pass' not in h2h3_result:
            consistency_pass = h2h3_result.get('improvement', 0.0) >= config.eval_tol.semigroup_improve_min
            stability_info = summary.get('stability', {}) or {}
            margin = stability_info.get('margin', config.stable.max_spectral_margin)
            if margin is None:
                margin = 0.0
            mu2_proj = stability_info.get('max_mu2_proj')
            stability_pass = mu2_proj is not None and mu2_proj <= margin + config.stable.alpha
            h2h3_result = {
                'status': 'PASS' if consistency_pass and stability_pass else 'FAIL',
                'pass': consistency_pass and stability_pass,
                'consistency': h2h3_result,
                'stability': stability_info,
                'stability_margin': margin,
                'clip_eps': config.train.clip_neg_dist_eps,
            }
    h4_result = summary.get('gating', {})
    if h4_result:
        min_frac = float(h4_result.get('min_cluster_fraction', h4_result.get('min_frac_actual', 0.0)))
        k_final = int(h4_result.get('K_final', 0))
        gating_pass = k_final >= 2 and min_frac >= config.gate.min_cluster_frac
        h4_result = {
            'status': 'PASS' if gating_pass else 'FAIL',
            'pass': gating_pass,
            'mode': h4_result.get('mode', config.gate.mode),
            'K_initial': int(h4_result.get('K_initial', 0)),
            'K_final': k_final,
            'min_cluster_fraction': min_frac,
            'required_min_fraction': config.gate.min_cluster_frac,
        }
    else:
        h4_result = {'status': 'SKIP', 'pass': False, 'reason': 'No gating info'}

    gmt_sources = args.msigdb_gmt or summary.get('msigdb', {}).get('gmt_files', [])
    if gmt_sources:
        pathway_result = _pathway_analysis(
            processed,
            args.stage_key,
            stage_map,
            gmt_sources,
            alpha=config.eval_tol.consistency_alpha,
        )
    else:
        pathway_result = {
            'status': 'SKIP',
            'pass': None,
            'reason': 'No MSigDB GMT files provided',
            'top_pathways': [],
        }

    h6_result = _compute_mloy_statistics(raw_subset)

    results = {
        'H1_uot_advantage': h1_result,
        'H2_H3_semigroup_consistency': h2h3_result,
        'H4_cluster_lockdown': h4_result,
        'H5_pathway_enrichment': pathway_result,
        'H6_mloy_diagnostics': h6_result,
    }

    decision = make_final_decision(results)

    report = {
        'decision': decision,
        'hypotheses': results,
        'config_used': config.to_dict(),
        'training_summary_path': args.summary,
        'stage_map': stage_map,
        'msigdb_sources': gmt_sources,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w', encoding='utf-8') as fh:
            json.dump(report, fh, indent=2)
    else:
        print(json.dumps(report, indent=2))

    if args.verbose:
        print('\nFinal decision:', decision)

    return report


if __name__ == '__main__':
    main()
