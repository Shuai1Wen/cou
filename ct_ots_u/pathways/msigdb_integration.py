"""MSigDB database integration and pathway enrichment analysis."""

from __future__ import annotations

import gzip
import json
import warnings
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Union
import urllib.request

import numpy as np
import pandas as pd

Array = np.ndarray


class MSigDBDownloader:
    """Download and manage MSigDB gene sets."""

    BASE_URL = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2023.2.Hs/"

    COLLECTIONS = {
        "H": "h.all.v2023.2.Hs.symbols.gmt",  # Hallmark gene sets
        "C1": "c1.all.v2023.2.Hs.symbols.gmt",  # Positional gene sets
        "C2": "c2.all.v2023.2.Hs.symbols.gmt",  # Curated gene sets
        "C3": "c3.all.v2023.2.Hs.symbols.gmt",  # Regulatory target gene sets
        "C5": "c5.all.v2023.2.Hs.symbols.gmt",  # Ontology gene sets
        "C6": "c6.all.v2023.2.Hs.symbols.gmt",  # Oncogenic signature gene sets
        "C7": "c7.all.v2023.2.Hs.symbols.gmt",  # Immunologic signature gene sets
        "C8": "c8.all.v2023.2.Hs.symbols.gmt"   # Cell type signature gene sets
    }

    def __init__(self, cache_dir: Union[str, Path] = None):
        """Initialize MSigDB downloader.

        Parameters
        ----------
        cache_dir : str or Path, optional
            Directory to cache downloaded files. If None, uses ./msigdb_cache
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./msigdb_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.gene_sets = {}

    def download_collection(self, collection: str, force_update: bool = False) -> Path:
        """Download a specific MSigDB collection.

        Parameters
        ----------
        collection : str
            Collection name (e.g., 'H', 'C2', 'C5')
        force_update : bool
            Force re-download even if file exists

        Returns
        -------
        file_path : Path
            Path to downloaded GMT file
        """

        if collection not in self.COLLECTIONS:
            raise ValueError(f"Unknown collection: {collection}. Available: {list(self.COLLECTIONS.keys())}")

        filename = self.COLLECTIONS[collection]
        url = self.BASE_URL + filename
        local_path = self.cache_dir / filename

        if not local_path.exists() or force_update:
            print(f"Downloading {collection} collection from MSigDB...")
            try:
                urllib.request.urlretrieve(url, local_path)
                print(f"Downloaded: {local_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download {collection}: {e}")

        return local_path

    def load_gmt(self, gmt_path: Union[str, Path]) -> Dict[str, Set[str]]:
        """Load gene sets from GMT file.

        Parameters
        ----------
        gmt_path : str or Path
            Path to GMT file

        Returns
        -------
        gene_sets : Dict[str, Set[str]]
            Dictionary mapping gene set names to sets of gene symbols
        """

        gene_sets = {}
        gmt_path = Path(gmt_path)

        if not gmt_path.exists():
            raise FileNotFoundError(f"GMT file not found: {gmt_path}")

        with open(gmt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue

                gene_set_name = parts[0]
                genes = set(parts[2:])  # Skip description field
                gene_sets[gene_set_name] = genes

        print(f"Loaded {len(gene_sets)} gene sets from {gmt_path}")
        return gene_sets

    def get_gene_sets(self, collections: List[str] = None, force_update: bool = False) -> Dict[str, Set[str]]:
        """Get gene sets from specified collections.

        Parameters
        ----------
        collections : List[str], optional
            List of collections to load. If None, loads commonly used ones.
        force_update : bool
            Force re-download of files

        Returns
        -------
        all_gene_sets : Dict[str, Set[str]]
            Combined gene sets from all collections
        """

        if collections is None:
            collections = ['H', 'C2', 'C5']  # Hallmark, Curated, GO

        all_gene_sets = {}

        for collection in collections:
            try:
                gmt_path = self.download_collection(collection, force_update)
                gene_sets = self.load_gmt(gmt_path)

                # Add collection prefix to avoid name conflicts
                for name, genes in gene_sets.items():
                    prefixed_name = f"{collection}_{name}" if not name.startswith(collection) else name
                    all_gene_sets[prefixed_name] = genes

            except Exception as e:
                warnings.warn(f"Failed to load collection {collection}: {e}")

        self.gene_sets = all_gene_sets
        return all_gene_sets

    def filter_by_size(self, min_size: int = 15, max_size: int = 500) -> Dict[str, Set[str]]:
        """Filter gene sets by size.

        Parameters
        ----------
        min_size : int
            Minimum number of genes
        max_size : int
            Maximum number of genes

        Returns
        -------
        filtered_gene_sets : Dict[str, Set[str]]
            Filtered gene sets
        """

        filtered = {
            name: genes for name, genes in self.gene_sets.items()
            if min_size <= len(genes) <= max_size
        }

        print(f"Filtered from {len(self.gene_sets)} to {len(filtered)} gene sets "
              f"(size range: {min_size}-{max_size})")

        return filtered


def pathway_enrichment_analysis(
    differential_genes: List[str],
    background_genes: List[str],
    gene_sets: Dict[str, Set[str]],
    min_overlap: int = 3,
    alpha: float = 0.05
) -> pd.DataFrame:
    """Perform pathway enrichment analysis using Fisher's exact test.

    Parameters
    ----------
    differential_genes : List[str]
        List of differentially expressed genes
    background_genes : List[str]
        List of background genes (all genes tested)
    gene_sets : Dict[str, Set[str]]
        Gene sets for enrichment testing
    min_overlap : int
        Minimum overlap required for testing
    alpha : float
        Significance threshold

    Returns
    -------
    results : pd.DataFrame
        Enrichment results with p-values and statistics
    """

    try:
        from scipy.stats import fisher_exact
    except ImportError:
        raise ImportError("scipy is required for pathway enrichment analysis")

    differential_set = set(differential_genes)
    background_set = set(background_genes)

    # Ensure differential genes are subset of background
    differential_set = differential_set.intersection(background_set)

    results = []

    for pathway_name, pathway_genes in gene_sets.items():
        # Intersect with background
        pathway_in_background = pathway_genes.intersection(background_set)

        if len(pathway_in_background) < min_overlap:
            continue

        # Overlap with differential genes
        overlap = differential_set.intersection(pathway_in_background)

        if len(overlap) < min_overlap:
            continue

        # Fisher's exact test
        # Contingency table:
        # | In pathway | Not in pathway |
        # |------------|----------------|
        # | Diff genes | a | b |
        # | Non-diff   | c | d |

        a = len(overlap)  # Differential + in pathway
        b = len(differential_set) - a  # Differential + not in pathway
        c = len(pathway_in_background) - a  # Non-differential + in pathway
        d = len(background_set) - len(pathway_in_background) - b  # Non-differential + not in pathway

        # Fisher's exact test (one-tailed, greater)
        _, p_value = fisher_exact([[a, b], [c, d]], alternative='greater')

        # Fold enrichment
        expected = len(differential_set) * len(pathway_in_background) / len(background_set)
        fold_enrichment = a / max(expected, 1e-10)

        results.append({
            'pathway': pathway_name,
            'overlap_genes': len(overlap),
            'pathway_size': len(pathway_in_background),
            'differential_genes': len(differential_set),
            'p_value': p_value,
            'fold_enrichment': fold_enrichment,
            'genes': ', '.join(sorted(overlap))
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Multiple testing correction (Benjamini-Hochberg)
    try:
        from scipy.stats import false_discovery_control
        df['q_value'] = false_discovery_control(df['p_value'])
    except ImportError:
        # Fallback: simple Bonferroni correction
        df['q_value'] = np.minimum(df['p_value'] * len(df), 1.0)

    # Sort by significance
    df = df.sort_values('p_value')

    # Add significance flags
    df['significant'] = df['q_value'] < alpha

    return df


def extract_differential_genes(
    adata_src,
    adata_tgt,
    method: str = "log_fold_change",
    threshold: float = 0.5,
    min_expression: float = 0.1
) -> List[str]:
    """Extract differential genes between source and target conditions.

    Parameters
    ----------
    adata_src, adata_tgt : AnnData
        Source and target single-cell data
    method : str
        Method for differential analysis
    threshold : float
        Threshold for calling genes differential
    min_expression : float
        Minimum expression level

    Returns
    -------
    differential_genes : List[str]
        List of differential gene symbols
    """

    if not hasattr(adata_src, 'X') or not hasattr(adata_tgt, 'X'):
        raise ValueError("Input must be AnnData objects with expression data")

    # Get expression matrices
    X_src = adata_src.X
    X_tgt = adata_tgt.X

    if hasattr(X_src, 'toarray'):
        X_src = X_src.toarray()
    if hasattr(X_tgt, 'toarray'):
        X_tgt = X_tgt.toarray()

    # Ensure same genes
    common_genes = list(set(adata_src.var_names) & set(adata_tgt.var_names))

    if len(common_genes) == 0:
        raise ValueError("No common genes between source and target")

    # Subset to common genes
    src_mask = adata_src.var_names.isin(common_genes)
    tgt_mask = adata_tgt.var_names.isin(common_genes)

    X_src = X_src[:, src_mask]
    X_tgt = X_tgt[:, tgt_mask]

    # Compute mean expression
    mean_src = np.mean(X_src, axis=0)
    mean_tgt = np.mean(X_tgt, axis=0)

    if method == "log_fold_change":
        # Log fold change
        log_fc = np.log2((mean_tgt + 1e-6) / (mean_src + 1e-6))

        # Filter by expression and fold change
        expressed = (mean_src > min_expression) | (mean_tgt > min_expression)
        differential = np.abs(log_fc) > threshold

        mask = expressed & differential

    elif method == "difference":
        # Simple difference
        diff = mean_tgt - mean_src
        expressed = (mean_src > min_expression) | (mean_tgt > min_expression)
        differential = np.abs(diff) > threshold

        mask = expressed & differential

    else:
        raise ValueError(f"Unknown method: {method}")

    differential_genes = [common_genes[i] for i in np.where(mask)[0]]

    print(f"Found {len(differential_genes)} differential genes (threshold={threshold})")

    return differential_genes


__all__ = [
    "MSigDBDownloader",
    "pathway_enrichment_analysis",
    "extract_differential_genes"
]
