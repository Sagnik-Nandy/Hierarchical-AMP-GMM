"""
TCGA-BRCA Multimodal Pipeline
==============================
Downloads RNA-seq, CNV, and DNA Methylation from UCSC Xena,
preprocesses each modality, aligns samples, and produces
low-rank approximations:

    X_i = U_i D_i V_i^T + Z_i,   Z_i ~ N(0, sigma_i^2)

Output matrices X_1, X_2, X_3 share the same row index (samples)
but have different numbers of columns (features).
"""

# ─────────────────────────────────────────────
# 0. DEPENDENCIES
# ─────────────────────────────────────────────
# pip install pandas numpy scipy scikit-learn matplotlib seaborn requests tqdm

import os
import requests
import gzip
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path("tcga_brca_data")
DATA_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# 1. DOWNLOAD FROM UCSC XENA
# ─────────────────────────────────────────────
# All files are from the TCGA BRCA cohort on UCSC Xena.
# Direct download URLs (GDC TCGA Breast Cancer hub).

XENA_FILES = {
    "rna": {
        "url": "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FHiSeqV2_PANCAN.gz",
        "filename": "BRCA_RNAseq.tsv.gz",
        "description": "RNA-seq gene expression (RSEM log2 normalized)"
    },
    "cnv": {
        "url": "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FGistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz",
        "filename": "BRCA_CNV.tsv.gz",
        "description": "CNV GISTIC2 thresholded copy number"
    },
    "meth": {
        "url": "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FHumanMethylation450.gz",
        "filename": "BRCA_Methylation450.tsv.gz",
        "description": "DNA Methylation 450k beta values"
    },
    "survival": {
        "url": "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/survival%2FBRCA_survival.txt",
        "filename": "BRCA_survival.tsv",
        "description": "Survival / clinical labels"
    }
}


def download_file(url: str, dest: Path, description: str = ""):
    """Download a file with progress bar."""
    if dest.exists():
        print(f"  [SKIP] {dest.name} already exists.")
        return
    print(f"  Downloading {description} ...")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"  Saved to {dest}")


def download_all():
    for key, info in XENA_FILES.items():
        dest = DATA_DIR / info["filename"]
        download_file(info["url"], dest, info["description"])


# ─────────────────────────────────────────────
# 2. LOAD RAW DATA
# ─────────────────────────────────────────────
# Xena files are (features x samples) TSVs — we transpose to (samples x features).

def load_xena_tsv(filepath: Path) -> pd.DataFrame:
    """Load a Xena TSV (features x samples) and transpose to (samples x features)."""
    print(f"  Loading {filepath.name} ...")
    df = pd.read_csv(filepath, sep="\t", index_col=0)
    df = df.T  # now: rows=samples, cols=features
    df.index.name = "sample_id"
    print(f"    Shape (samples x features): {df.shape}")
    return df


def load_survival(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep="\t")
    df = df.set_index("sample")
    return df


# ─────────────────────────────────────────────
# 3. SAMPLE ALIGNMENT
# ─────────────────────────────────────────────
# Keep only primary tumor samples (TCGA barcode position 14 = '01')
# and intersect across all three modalities.

def is_primary_tumor(barcode: str) -> bool:
    """TCGA barcode: positions 13-14 (0-indexed) indicate sample type. '01' = primary tumor."""
    try:
        return barcode.split("-")[3][:2] == "01"
    except IndexError:
        return False


def align_samples(*dataframes) -> list:
    """Keep only primary tumor samples present in ALL modalities."""
    filtered = []
    for df in dataframes:
        primary = [s for s in df.index if is_primary_tumor(s)]
        filtered.append(df.loc[primary])

    common = filtered[0].index
    for df in filtered[1:]:
        common = common.intersection(df.index)

    print(f"\n  Shared primary tumor samples across all modalities: {len(common)}")
    return [df.loc[common] for df in filtered]


# ─────────────────────────────────────────────
# 4. MODALITY-SPECIFIC PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_rna(df: pd.DataFrame, var_threshold_pct: float = 20) -> pd.DataFrame:
    """
    RNA-seq (already log2(RSEM+1) normalized by Xena):
    1. Remove low-variance genes (bottom var_threshold_pct%)
    2. Remove genes with >20% missing values
    3. Impute remaining NaNs with gene median
    4. Z-score normalize across samples (per gene)
    """
    print("\n  [RNA] Preprocessing ...")
    # Drop high-missing genes
    missing_frac = df.isnull().mean(axis=0)
    df = df.loc[:, missing_frac < 0.20]

    # Impute with gene median
    df = df.fillna(df.median(axis=0))

    # Remove low-variance genes
    gene_var = df.var(axis=0)
    threshold = gene_var.quantile(var_threshold_pct / 100)
    df = df.loc[:, gene_var >= threshold]

    # Z-score per gene
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

    print(f"    Final shape: {df_scaled.shape}")
    return df_scaled.astype(np.float32)


def preprocess_cnv(df: pd.DataFrame) -> pd.DataFrame:
    """
    CNV GISTIC2 thresholded values: {-2, -1, 0, 1, 2}
    Treat as continuous (already numeric).
    1. Remove genes with >20% missing
    2. Impute with 0 (diploid, neutral copy number)
    3. Remove zero-variance genes (no CNV events)
    4. Cast to float32
    """
    print("\n  [CNV] Preprocessing ...")
    df = df.apply(pd.to_numeric, errors="coerce")

    missing_frac = df.isnull().mean(axis=0)
    df = df.loc[:, missing_frac < 0.20]

    df = df.fillna(0.0)

    # Remove zero-variance features
    gene_var = df.var(axis=0)
    df = df.loc[:, gene_var > 0]

    print(f"    Final shape: {df.shape}")
    return df.astype(np.float32)


def preprocess_methylation(df: pd.DataFrame,
                            var_threshold_pct: float = 20,
                            missing_thresh: float = 0.20) -> pd.DataFrame:
    """
    Methylation beta values in [0, 1].
    1. Remove CpG sites with >20% missing
    2. Impute remaining NaNs with site median
    3. Remove low-variance CpG sites (bottom 20%)
    4. M-value transform: M = log2(beta / (1 - beta))  [more statistically appropriate]
    5. Clip extreme M-values at ±10
    6. Z-score normalize
    """
    print("\n  [Methylation] Preprocessing ...")
    df = df.apply(pd.to_numeric, errors="coerce")

    # Remove high-missing CpGs
    missing_frac = df.isnull().mean(axis=0)
    df = df.loc[:, missing_frac < missing_thresh]

    # Impute with site median
    df = df.fillna(df.median(axis=0))

    # Clip beta to avoid log(0)
    epsilon = 1e-6
    df = df.clip(epsilon, 1 - epsilon)

    # M-value transform
    df = np.log2(df / (1 - df))

    # Remove low-variance CpGs
    cpg_var = df.var(axis=0)
    threshold = cpg_var.quantile(var_threshold_pct / 100)
    df = df.loc[:, cpg_var >= threshold]

    # Clip extreme values
    df = df.clip(-10, 10)

    # Z-score
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

    print(f"    Final shape: {df_scaled.shape}")
    return df_scaled.astype(np.float32)


# ─────────────────────────────────────────────
# 5. LOW-RANK APPROXIMATION
# ─────────────────────────────────────────────
# X_i = U_i D_i V_i^T + Z_i
# We estimate rank r_i via explained variance (e.g., 90% threshold),
# then compute truncated SVD.

def estimate_rank(X: np.ndarray, variance_threshold: float = 0.90) -> int:
    """Estimate rank as number of components explaining variance_threshold of variance."""
    max_components = min(X.shape) - 1
    svd = TruncatedSVD(n_components=max_components, random_state=42)
    svd.fit(X)
    cumvar = np.cumsum(svd.explained_variance_ratio_)
    rank = int(np.searchsorted(cumvar, variance_threshold)) + 1
    return rank


def low_rank_decompose(df: pd.DataFrame,
                        rank: int = None,
                        variance_threshold: float = 0.90,
                        name: str = "X") -> dict:
    """
    Compute truncated SVD: X ≈ U D V^T
    Returns dict with U, D, V, X_approx, Z (residual), sigma_Z.
    """
    X = df.values  # (n_samples x n_features)
    n, p = X.shape

    if rank is None:
        rank = estimate_rank(X, variance_threshold)
        print(f"\n  [{name}] Estimated rank = {rank} "
              f"(explains {variance_threshold*100:.0f}% variance)")

    svd = TruncatedSVD(n_components=rank, random_state=42)
    svd.fit(X)

    # Components
    V = svd.components_.T          # (p x rank)
    D = np.diag(svd.singular_values_)  # (rank x rank)
    U = X @ V @ np.linalg.inv(D)  # (n x rank)

    # Low-rank approximation and residual
    X_approx = U @ D @ V.T        # (n x p)
    Z = X - X_approx               # residual noise

    sigma_Z = Z.std()
    expl_var = svd.explained_variance_ratio_.sum()

    print(f"  [{name}] Explained variance by rank-{rank} approx: {expl_var*100:.2f}%")
    print(f"  [{name}] Residual noise std (sigma_Z): {sigma_Z:.4f}")
    print(f"  [{name}] U: {U.shape}, D: {D.shape}, V^T: {V.T.shape}")

    return {
        "name": name,
        "X_full": pd.DataFrame(X, index=df.index, columns=df.columns),
        "X_approx": pd.DataFrame(X_approx, index=df.index, columns=df.columns),
        "U": pd.DataFrame(U, index=df.index),
        "D": D,
        "V": pd.DataFrame(V, columns=range(rank), index=df.columns),
        "Z": pd.DataFrame(Z, index=df.index, columns=df.columns),
        "sigma_Z": sigma_Z,
        "rank": rank,
        "explained_variance": expl_var,
    }


# ─────────────────────────────────────────────
# 6. DIAGNOSTICS & VISUALIZATION
# ─────────────────────────────────────────────

def plot_singular_values(results: list, save_path: Path = None):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    for ax, res in zip(axes, results):
        singular_vals = np.diag(res["D"])
        ax.plot(singular_vals, "o-", color="steelblue")
        ax.axvline(res["rank"] - 1, color="red", linestyle="--",
                   label=f"rank={res['rank']}")
        ax.set_title(f"{res['name']} Singular Values")
        ax.set_xlabel("Component")
        ax.set_ylabel("Singular Value")
        ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved plot: {save_path}")
    plt.show()


def plot_residual_distribution(results: list, save_path: Path = None):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    for ax, res in zip(axes, results):
        z_flat = res["Z"].values.flatten()
        z_sample = np.random.choice(z_flat, size=min(50000, len(z_flat)), replace=False)
        ax.hist(z_sample, bins=80, color="salmon", edgecolor="none", density=True)
        # Overlay N(0, sigma^2)
        xs = np.linspace(z_sample.min(), z_sample.max(), 300)
        ax.plot(xs, stats.norm.pdf(xs, 0, res["sigma_Z"]),
                "k-", linewidth=2, label=f"N(0,{res['sigma_Z']:.3f}²)")
        ax.set_title(f"{res['name']} Residual Z")
        ax.set_xlabel("Residual value")
        ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved plot: {save_path}")
    plt.show()


def print_summary(results: list, sample_ids: pd.Index):
    print("\n" + "="*60)
    print("FINAL MATRIX SUMMARY")
    print("="*60)
    print(f"  Shared samples (n): {len(sample_ids)}")
    for res in results:
        print(f"\n  {res['name']}:")
        print(f"    Full matrix X:     {res['X_full'].shape}")
        print(f"    Low-rank approx:   rank = {res['rank']}")
        print(f"    Explained var:     {res['explained_variance']*100:.1f}%")
        print(f"    sigma_Z (noise):   {res['sigma_Z']:.4f}")
        print(f"    U shape:           {res['U'].shape}")
        print(f"    D shape:           {res['D'].shape}")
        print(f"    V shape:           {res['V'].shape}")


# ─────────────────────────────────────────────
# 7. SAVE OUTPUTS
# ─────────────────────────────────────────────

def save_matrices(results: list, out_dir: Path):
    out_dir.mkdir(exist_ok=True)
    for res in results:
        name = res["name"].replace(" ", "_")
        res["X_full"].to_csv(out_dir / f"{name}_X_full.csv")
        res["X_approx"].to_csv(out_dir / f"{name}_X_approx.csv")
        res["U"].to_csv(out_dir / f"{name}_U.csv")
        np.savetxt(out_dir / f"{name}_D.csv", res["D"], delimiter=",")
        res["V"].to_csv(out_dir / f"{name}_V.csv")
        res["Z"].to_csv(out_dir / f"{name}_Z_residual.csv")
    print(f"\n  All matrices saved to {out_dir}/")


# ─────────────────────────────────────────────
# 8. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(
    variance_threshold: float = 0.90,
    rna_var_pct: float = 20,
    meth_var_pct: float = 20,
    save_outputs: bool = True,
):
    print("=" * 60)
    print("STEP 1: Downloading data from UCSC Xena")
    print("=" * 60)
    download_all()

    print("\n" + "=" * 60)
    print("STEP 2: Loading raw data")
    print("=" * 60)
    rna_raw  = load_xena_tsv(DATA_DIR / XENA_FILES["rna"]["filename"])
    cnv_raw  = load_xena_tsv(DATA_DIR / XENA_FILES["cnv"]["filename"])
    meth_raw = load_xena_tsv(DATA_DIR / XENA_FILES["meth"]["filename"])

    print("\n" + "=" * 60)
    print("STEP 3: Aligning samples (primary tumors only)")
    print("=" * 60)
    rna_aligned, cnv_aligned, meth_aligned = align_samples(rna_raw, cnv_raw, meth_raw)

    print("\n" + "=" * 60)
    print("STEP 4: Modality-specific preprocessing")
    print("=" * 60)
    X1 = preprocess_rna(rna_aligned, var_threshold_pct=rna_var_pct)
    X2 = preprocess_cnv(cnv_aligned)
    X3 = preprocess_methylation(meth_aligned, var_threshold_pct=meth_var_pct)

    # Confirm same row order
    assert list(X1.index) == list(X2.index) == list(X3.index), \
        "Sample indices are not aligned!"
    sample_ids = X1.index

    print("\n" + "=" * 60)
    print("STEP 5: Low-rank decomposition  X_i = U_i D_i V_i^T + Z_i")
    print("=" * 60)
    res1 = low_rank_decompose(X1, variance_threshold=variance_threshold, name="X1_RNA")
    res2 = low_rank_decompose(X2, variance_threshold=variance_threshold, name="X2_CNV")
    res3 = low_rank_decompose(X3, variance_threshold=variance_threshold, name="X3_Meth")

    results = [res1, res2, res3]
    print_summary(results, sample_ids)

    print("\n" + "=" * 60)
    print("STEP 6: Diagnostics")
    print("=" * 60)
    plot_singular_values(results, save_path=Path("singular_values.png"))
    plot_residual_distribution(results, save_path=Path("residuals.png"))

    if save_outputs:
        print("\n" + "=" * 60)
        print("STEP 7: Saving matrices")
        print("=" * 60)
        save_matrices(results, out_dir=Path("tcga_brca_matrices"))

    return X1, X2, X3, results, sample_ids


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    X1, X2, X3, results, sample_ids = run_pipeline(
        variance_threshold=0.90,   # rank chosen to explain 90% variance
        rna_var_pct=20,            # remove bottom 20% variance genes
        meth_var_pct=20,           # remove bottom 20% variance CpGs
        save_outputs=True,
    )

    # Quick access to decomposition components:
    U1, D1, V1 = results[0]["U"], results[0]["D"], results[0]["V"]
    U2, D2, V2 = results[1]["U"], results[1]["D"], results[1]["V"]
    U3, D3, V3 = results[2]["U"], results[2]["D"], results[2]["V"]

    print("\nDone! Matrices ready for downstream modeling.")
    print(f"  X1 (RNA-seq):       {X1.shape}")
    print(f"  X2 (CNV):           {X2.shape}")
    print(f"  X3 (Methylation):   {X3.shape}")
