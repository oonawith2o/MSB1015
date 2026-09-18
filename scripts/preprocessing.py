import logging
import time
import h5py
import pandas as pd
import scanpy as sc
import numpy as np
import seaborn as sns
from scipy.stats import median_abs_deviation

def is_outlier(adata, metric: str, nmads: int):
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
        np.median(M) + nmads * median_abs_deviation(M) < M
    )
    return outlier

logging.basicConfig(filename=f'../log/{time.strftime("%Y%m%d-%H%M%S")}-preprocessing.log', filemode='w', 
                    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("[0] Starting Preprocessing") 

sc_matrix_path = "../data/GSE225600_sc_matrix.mtx/matrix.mtx"
sc_features_path = "../data/GSE225600_sc_features.tsv/GSE225600_sc_features.tsv"
sc_barcodes_path = "../data/GSE225600_sc_barcodes.tsv/GSE225600_sc_barcodes.tsv"
save_path = '../data/GSE225600_sc_matrix.mtx/matrix_2000.h5'

features = pd.read_csv(
    sc_features_path,
    sep="\t",
    header=None,
    dtype=str
)

barcodes = pd.read_csv(
    sc_barcodes_path,
    sep="\t",
    header=None,
    dtype=str
)

adata = sc.read_mtx(sc_matrix_path).T.copy()

logging.info("[1] Completed Importing the data and creating AnnData object") 

logging.info("-----Data Summary-----")    
logging.info("Shape of the original dataset: %s", adata.shape)  # Cell x Gene
logging.info("Number of variables (genes): %d", adata.shape[1])     # Column: Genes
logging.info("Number of observations (barcodes): %d", adata.shape[0])  # Row: Barcodes
logging.info("Type of adata.X: %s", type(adata.X))
logging.info("Shape of adata.X: %s", adata.X.shape)
logging.info("Dtype of adata.X: %s", adata.X.dtype)
logging.info("Type of features: %s", type(features))
logging.info("Shape of features: %s", features.shape)
logging.info("Head of features: \n%s", features[0].head())
logging.info("Type of barcodes: %s", type(barcodes))
logging.info("Shape of barcodes: %s", barcodes.shape)
logging.info("Head of barcodes: \n%s", barcodes[0].head())

if adata.n_obs != len(barcodes):
    raise ValueError(
        f"Number of barcodes does not match: "
        f"matrix={adata.n_obs}, barcodes={len(barcodes)}"
    )

if adata.n_vars != len(features):
    raise ValueError(
        f"Number of genes does not match: "
        f"matrix={adata.n_vars}, features={len(features)}"
    )

adata.obs_names = barcodes[0].astype(str).values
adata.obs_names_make_unique()

adata.var_names = features[0].astype(str).values
adata.var_names_make_unique()

adata.obs["sample"] = adata.obs_names.to_series().str.split("-").str[-1]

logging.info("[2] Completed Assigning Cell and Gene Names")
logging.info("-----Sample Distribution-----")
logging.info("Sample Counts:\n%s", adata.obs["sample"].value_counts())
logging.info("AnnData Object:\n%s", adata)

# mitochondrial genes
adata.var["mt"] = adata.var_names.str.startswith("MT-")
# ribosomal genes
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
# hemoglobin genes.
adata.var["hb"] = adata.var_names.str.contains(r"^HB[ABDEGMQZ]\d*(?!\w)")

sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, percent_top=[20], log1p=True)

logging.info("[3] Completed Quality Control")
logging.info("-----QC Metrics Summary-----")
logging.info("QC metrics:\n%s", adata)

sns.displot(adata.obs["total_counts"], bins=100, kde=False)
sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True,
)
sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt")

# Outlier Detection
adata.obs["outlier"] = (
    is_outlier(adata, "log1p_total_counts", 5)
    | is_outlier(adata, "log1p_n_genes_by_counts", 5)
    | is_outlier(adata, "pct_counts_in_top_20_genes", 5)
)
logging.info("Outlier distribution:\n%s", adata.obs.outlier.value_counts())

adata.obs["mt_outlier"] = is_outlier(adata, "pct_counts_mt", 3) | (
    adata.obs["pct_counts_mt"] > 8
)
logging.info("Mitochondrial outlier distribution:\n%s", adata.obs.mt_outlier.value_counts())

logging.info("[4] Completed Outlier Detection")
logging.info("-----Outlier Detection Summary-----") 
logging.info(f"Total number of cells: {adata.n_obs}")

adata.raw = adata.copy()
adata = adata[(~adata.obs.outlier) & (~adata.obs.mt_outlier)].copy()

logging.info(f"Number of cells after filtering of low quality cells: {adata.n_obs}")

sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt")

# Filter Cells and Genes
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=200)

logging.info("[5] Completed Filtering Cells and Genes")

# Doublet Detection 
sc.pp.scrublet(adata, batch_key="sample")

logging.info("[6] Completed Doublet Detection")
logging.info("Number of doublets detected: %d", adata.obs["predicted_doublet"].sum())

# Saving count data
adata.layers["counts"] = adata.X.copy()

# Normalizing to median total counts and add Size Factor
sc.pp.normalize_total(adata)
adata.obs['size_factors'] = adata.obs.total_counts / np.median(adata.obs.total_counts)

# Logarithmize the data
sc.pp.log1p(adata)

logging.info("[7] Completed Normalization and Log Transformation")

# Feature Selection
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = 2000, subset=True)
sc.pl.highly_variable_genes(adata)

logging.info("[8] Completed Feature Selection")

# Normalize Input Data
sc.pp.scale(adata)

logging.info("[9] Completed Input Data Normalization")

# Saving the Normalized Data to a New .h5 file
with h5py.File(save_path, 'w') as f_normalized:
    f_normalized.create_dataset('X', data=adata.X, compression="gzip", compression_opts=9)
    y = np.array(adata.obs_names, dtype='S')
    f_normalized.create_dataset('Y', data=y, compression="gzip", compression_opts=9)

'''
# Dimensionality Reduction
sc.tl.pca(adata)
sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True)
sc.pl.pca(
    adata,
    color=["sample", "sample", "pct_counts_mt", "pct_counts_mt"],
    dimensions=[(0, 1), (2, 3), (0, 1), (2, 3)],
    ncols=2,
    size=2,
)
logging.info("[9] Completed Dimensionality Reduction")
'''