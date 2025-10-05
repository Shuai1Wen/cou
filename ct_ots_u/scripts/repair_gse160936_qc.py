#!/usr/bin/env python3
"""Augment GSE160936_microglia_only.h5ad with QC metrics and region labels."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

MT_REGEX = re.compile(r"^MT-")
DEFAULT_MAP = Path("data/metadata/GSE160936/donor_region_map.csv")


def load_region_map(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "donor_id" not in df.columns or "region" not in df.columns:
        raise ValueError(f"Region map {path} missing required columns")
    df["region"] = df["region"].astype(str)
    return df.set_index("donor_id")


def assign_region_from_map(adata: sc.AnnData, region_map: pd.DataFrame) -> np.ndarray:
    donors = adata.obs["donor_id"].astype(str)
    region = donors.map(region_map["region"])
    return region.to_numpy()


def infer_region(sample_value: str) -> str:
    if sample_value is None or (isinstance(sample_value, float) and np.isnan(sample_value)):
        return np.nan
    token = str(sample_value).upper()
    if "EC" in token:
        return "EC"
    if "SSC" in token or "SS" in token:
        return "SSC"
    return np.nan


def annotate_cell_type(adata: sc.AnnData) -> None:
    marker_genes = [g for g in ["P2RY12", "CX3CR1", "TMEM119"] if g in adata.var_names]
    if not marker_genes:
        adata.obs["cell_type"] = np.nan
        return
    subset = adata[:, marker_genes].X
    if hasattr(subset, "A1"):
        signal = subset.mean(axis=1).A1
    elif hasattr(subset, "todense"):
        signal = np.asarray(subset.todense()).mean(axis=1)
    else:
        signal = np.asarray(subset).mean(axis=1)
    threshold = np.nanpercentile(signal, 60)
    adata.obs["cell_type"] = np.where(signal >= threshold, "Microglia_like", np.nan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad-in", dest="h5ad_in", default="data/raw-use/GSE160936/GSE160936_microglia_only.h5ad")
    parser.add_argument("--h5ad-out", dest="h5ad_out", default="data/raw-use/GSE160936/GSE160936_microglia_only.qc.h5ad")
    parser.add_argument("--region-map", dest="region_map", default=str(DEFAULT_MAP), help="CSV mapping donor_id to region")
    args = parser.parse_args()

    ad = sc.read_h5ad(args.h5ad_in)

    if "pct_counts_mt" not in ad.obs.columns:
        mt_mask = ad.var_names.str.match(MT_REGEX)
        if mt_mask.sum() > 0:
            mt_counts = ad[:, mt_mask].X.sum(axis=1)
            total = ad.X.sum(axis=1)
            mt_counts = np.asarray(mt_counts).reshape(-1)
            total = np.asarray(total).reshape(-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                pct_mt = np.where(total > 0, (mt_counts / total) * 100.0, np.nan)
            ad.obs["pct_counts_mt"] = pct_mt
        else:
            ad.obs["pct_counts_mt"] = np.nan

    region_series = None
    region_map_path = Path(args.region_map)
    region_map_df = None
    if region_map_path.exists():
        region_map_df = load_region_map(region_map_path)
        try:
            region_series = assign_region_from_map(ad, region_map_df)
        except Exception as exc:
            raise RuntimeError(f"Failed to map regions using {region_map_path}") from exc

    if region_series is not None:
        ad.obs["region"] = region_series
    else:
        if "sample" in ad.obs.columns:
            ad.obs["region"] = ad.obs["sample"].map(infer_region)
        elif "donor_id" in ad.obs.columns:
            ad.obs["region"] = ad.obs["donor_id"].map(infer_region)
        else:
            ad.obs["region"] = np.nan

    if "cell_type" not in ad.obs.columns:
        annotate_cell_type(ad)

    Path(args.h5ad_out).parent.mkdir(parents=True, exist_ok=True)
    ad.write_h5ad(args.h5ad_out, compression="gzip")

    report = {
        "n_obs": int(ad.n_obs),
        "n_vars": int(ad.n_vars),
        "pct_counts_mt_non_null": int(ad.obs["pct_counts_mt"].notna().sum()),
        "region_missing": int(ad.obs["region"].isna().sum()),
        "region_levels": pd.Series(ad.obs["region"], dtype="object").value_counts(dropna=False).to_dict(),
        "cell_type_levels": pd.Series(ad.obs["cell_type"], dtype="object").value_counts(dropna=False).to_dict(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
