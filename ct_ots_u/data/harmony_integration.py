"""Harmony batch integration wrapper for CT-OTS-U.

Reference:
    Korsunsky et al. "Fast, sensitive and accurate integration of single-cell data with Harmony"
    Nature Methods (2019)
    https://www.nature.com/articles/s41592-019-0619-0

Note:
    This is a Python wrapper. Actual Harmony runs in R via scanpy.external.pp.harmony_integrate
    or harmonypy (pure Python implementation).
"""

from __future__ import annotations

from typing import Optional, Union
import numpy as np


def apply_harmony_integration(
    X_pca: np.ndarray,
    batch_labels: np.ndarray,
    n_components: Optional[int] = None,
    theta: float = 2.0,
    lambda_: float = 1.0,
    max_iter: int = 10,
    backend: str = 'harmonypy',  # 'harmonypy' or 'scanpy'
    random_state: int = 42,
) -> np.ndarray:
    """Apply Harmony batch correction to PCA embeddings.

    Args:
        X_pca: PCA embeddings [n_cells, n_pcs]
        batch_labels: Batch/donor/site labels [n_cells]
        n_components: Number of harmonized components (default: use all)
        theta: Diversity clustering penalty (higher = more aggressive correction)
        lambda_: Ridge regularization parameter
        max_iter: Maximum number of Harmony iterations
        backend: 'harmonypy' (pure Python) or 'scanpy' (via R/reticulate)
        random_state: Random seed

    Returns:
        X_harmony: Harmonized embeddings [n_cells, n_components]
    """

    if n_components is None:
        n_components = X_pca.shape[1]

    if backend == 'harmonypy':
        try:
            from harmonypy import run_harmony

            # HarmonyPy expects input as pandas DataFrame
            import pandas as pd

            # Create DataFrame for PCA
            pca_df = pd.DataFrame(
                X_pca[:, :n_components],
                columns=[f'PC{i+1}' for i in range(n_components)]
            )

            # Create metadata DataFrame
            meta_df = pd.DataFrame({
                'batch': batch_labels,
            })

            # Run Harmony
            harmony_out = run_harmony(
                pca_df,
                meta_df,
                vars_use=['batch'],
                theta=theta,
                lamb=lambda_,
                max_iter_harmony=max_iter,
                random_state=random_state,
                verbose=False,
            )

            X_harmony = harmony_out.Z_corr.T  # Transpose to [n_cells, n_pcs]
            return X_harmony

        except ImportError:
            raise ImportError(
                "harmonypy not installed. Install with: pip install harmonypy"
            )

    elif backend == 'scanpy':
        try:
            import scanpy as sc
            import anndata as ad

            # Create temporary AnnData object
            adata = ad.AnnData(X=X_pca[:, :n_components])
            adata.obs['batch'] = batch_labels
            adata.obsm['X_pca'] = X_pca[:, :n_components]

            # Run Harmony via scanpy
            sc.external.pp.harmony_integrate(
                adata,
                key='batch',
                basis='X_pca',
                adjusted_basis='X_pca_harmony',
                theta=theta,
                max_iter_harmony=max_iter,
            )

            X_harmony = adata.obsm['X_pca_harmony']
            return X_harmony

        except ImportError:
            raise ImportError(
                "Scanpy Harmony not available. Install scanpy and ensure R/harmonypy is configured."
            )

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'harmonypy' or 'scanpy'.")


class HarmonyIntegrator:
    """Harmony batch integration with dual-mode support.

    Maintains both original and harmonized embeddings for analysis.
    """

    def __init__(
        self,
        use_harmony: bool = False,
        theta: float = 2.0,
        lambda_: float = 1.0,
        max_iter: int = 10,
        backend: str = 'harmonypy',
    ):
        """Initialize Harmony integrator.

        Args:
            use_harmony: Whether to apply Harmony correction
            theta: Diversity clustering penalty
            lambda_: Ridge regularization
            max_iter: Maximum iterations
            backend: 'harmonypy' or 'scanpy'
        """
        self.use_harmony = use_harmony
        self.theta = theta
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.backend = backend

        self.X_original: Optional[np.ndarray] = None
        self.X_harmony: Optional[np.ndarray] = None

    def fit_transform(
        self,
        X_pca: np.ndarray,
        batch_labels: np.ndarray,
        n_components: Optional[int] = None,
    ) -> np.ndarray:
        """Fit Harmony and return harmonized embeddings.

        Args:
            X_pca: PCA embeddings
            batch_labels: Batch labels
            n_components: Number of components to use

        Returns:
            X_harmony: Harmonized embeddings (or original if use_harmony=False)
        """
        self.X_original = X_pca.copy()

        if not self.use_harmony:
            self.X_harmony = self.X_original
            return self.X_harmony

        self.X_harmony = apply_harmony_integration(
            X_pca,
            batch_labels,
            n_components=n_components,
            theta=self.theta,
            lambda_=self.lambda_,
            max_iter=self.max_iter,
            backend=self.backend,
        )

        return self.X_harmony

    def get_embeddings(self, mode: str = 'auto') -> np.ndarray:
        """Get embeddings.

        Args:
            mode: 'auto' (harmony if available, else original), 'original', or 'harmony'

        Returns:
            X: Embeddings
        """
        if mode == 'original':
            if self.X_original is None:
                raise ValueError("No embeddings available. Call fit_transform first.")
            return self.X_original

        elif mode == 'harmony':
            if self.X_harmony is None:
                raise ValueError("No Harmony embeddings. Call fit_transform first.")
            return self.X_harmony

        elif mode == 'auto':
            if self.use_harmony and self.X_harmony is not None:
                return self.X_harmony
            elif self.X_original is not None:
                return self.X_original
            else:
                raise ValueError("No embeddings available.")

        else:
            raise ValueError(f"Unknown mode: {mode}")


__all__ = [
    'apply_harmony_integration',
    'HarmonyIntegrator',
]
