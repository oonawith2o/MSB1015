import logging
import time
import math
import h5py

import pandas as pd
import scanpy as sc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import median_abs_deviation

timestamp = time.strftime('%Y%m%d')

#----------- DESIGN SETUP -----------

lab_size = 25
tick_size = 15

main_colors = ["#0E2841", "#4E95D9", "#CBCBCB", "#F0F0F0"]
accent_color = "#FFC000"

#------------------------------------

#----------- LOGGER SETUP -----------

logging.basicConfig(filename=f'../log/{timestamp}-01_load_data.log', filemode='w', 
                    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.category").setLevel(logging.WARNING)

logging.info("------------ Starting quality control ------------") 

#------------------------------------

#----------- FILE NAMES -------------

sc_matrix_path = "../data/GSE225600_sc_matrix.mtx/matrix.mtx"
sc_features_path = "../data/GSE225600_sc_features.tsv/GSE225600_sc_features.tsv"
sc_barcodes_path = "../data/GSE225600_sc_barcodes.tsv/GSE225600_sc_barcodes.tsv"
patients_path = "../data/supplementary_data/ADVS-10-2205395-s008.xlsx"
save_path = '../data/processed_data/adata_raw.h5'

preprocessing_raw_dir = Path("../results/preprocessing") / timestamp / "raw"
preprocessing_raw_dir.mkdir(parents=True, exist_ok=True)

#------------------------------------

#----------- FUNCTIONS --------------

def savefig(path):
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

#------------------------------------

#----------- GLOBAL VARIABLES -------

QC_GROUPS = ["patient", "region", "sample"]

QC_METRICS = [
    "n_genes_by_counts",
    "total_counts",
    "pct_counts_mt",
]

QC_PERCENTILES = [
    0.01, 0.05, 0.10, 0.25, 0.50,
    0.75, 0.90, 0.95, 0.99
]

# PRIMARY QC THRESHOLDS
# - at least 200 detected genes per cell
# - at least 3 detected cells per gene
# - less than 20% mitochondrial counts

MIN_GENES = 200
MIN_CELLS = 3
MT_CUTOFF = 20

MT_CUTOFFS = [15, 20, 25, 30]

SAMPLES_TO_INSPECT = [
    "P6_L",
    "P3_T",
    "P2_T",
    "P7_T",
]

TARGET_SAMPLES = [
    "P6_T",
    "P6_L",
]

#------------------------------------


# ============================================================
# ------------------------- MAIN -----------------------------
# ============================================================

# ----------- LOAD DATA ---------- #

logging.info("[1] Loading data and creating AnnData object") 

# load patient table data
patient_variables = ["age", "sex", "tumor_size", "lymph_nodes", "cancer_type", "grade",
                     "ER", "PR", "HER2","Ki67", "stage", "treatment"]
patients = pd.read_excel(
    patients_path, 
    header=1, 
    names=patient_variables,
    index_col=0
)
patients.index = {"2","3","6","7"}
patients["tumor_volume"] = patients["tumor_size"].apply(
    lambda x: eval(x) if pd.notna(x) else None
)
patient_variables.append("tumor_volume")
logging.info("Patient information table\n%s",patients)

# load single-cell data
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

adata = sc.read_mtx(sc_matrix_path)
adata = adata.transpose()
  
logging.info("Shape of the original dataset: %s", adata.shape)          # Cell x Gene
logging.info("Number of variables (genes): %d", adata.shape[1])         # Column: Genes
logging.info("Number of observations (barcodes): %d", adata.shape[0])   # Row: Barcodes
logging.info("Type of adata.X: %s", type(adata.X))
logging.info("Shape of adata.X: %s", adata.X.shape)
logging.info("Dtype of adata.X: %s", adata.X.dtype)
logging.info("Type of features: %s", type(features))
logging.info("Shape of features: %s", features.shape)
logging.info("Type of barcodes: %s", type(barcodes))
logging.info("Shape of barcodes: %s", barcodes.shape)

# ----------- DATA ANNOTATION ---------- #

logging.info("[2] Annotating cell and gene names")

# load and map barcodes onto adata object 
barcodes.rename(columns={0:'barcode'}, inplace=True)
barcodes.set_index('barcode', inplace=True)
adata.obs = barcodes
adata.obs_names = adata.obs.index.astype(str)                                                   # set barcode as the obs index
adata.obs_names.name = "barcode"                                                                 
barcode_info = adata.obs_names.to_series().str.extract(r"-(?P<region>[LT])(?P<patient>\d+)$")
adata.obs['sample'] = ("P" + barcode_info["patient"] + "_" + barcode_info["region"]).astype("category")
adata.obs['patient'] = barcode_info["patient"].astype("category")
adata.obs['region'] = barcode_info["region"].astype("category")

# map patient data onto adata object 
assert adata.obs["patient"].isin(patients.index).all()

for variable in patient_variables:
    adata.obs[variable] = adata.obs["patient"].map(patients[variable])

numeric_variables = ["age", "Ki67"] 
for variable in numeric_variables:
    adata.obs[variable] = pd.to_numeric(
        adata.obs[variable],
        errors="coerce"
    )

categorical_variables = [ "tumor_size", "lymph_nodes", "sex", "cancer_type", "grade", "ER", "PR", "HER2", "stage", "treatment"]
for variable in categorical_variables:
    adata.obs[variable] = (
        adata.obs[variable].astype("string").astype("category")
    )

check = adata.obs[["patient"] + patient_variables].head(10)

logging.info("Patient data mapping check\n%s", check.to_string())

# load and map features onto adata object 
features.rename(columns={0:'id', 1:'gene_symbol'}, inplace=True)
features.set_index('id', inplace=True)
adata.var = features
adata.var_names = adata.var.index.astype(str) 
adata.var_names.name = "gene_symbol"    
adata.var_names_make_unique()

logging.info("obs_names name: %s", adata.obs_names.name)
logging.info("var_names name: %s", adata.var_names.name)
logging.info("First barcodes: %s", adata.obs_names[:5].tolist())
logging.info("First genes: %s", adata.var_names[:5].tolist())
logging.info("obs columns: %s", adata.obs.columns.tolist())
logging.info("var columns: %s", adata.var.columns.tolist())
logging.info("Patient counts:\n%s", adata.obs['patient'].value_counts())
logging.info("Region counts:\n%s", adata.obs['region'].value_counts())
logging.info("Sample counts:\n%s", adata.obs['sample'].value_counts())
logging.info("AnnData object:\n%s", adata)

# ----------- COMPUTE QC METRICS ---------- #

logging.info("[3] Quality control metrics computation")

# mitochondrial Genes
adata.var["mt"] = adata.var_names.str.startswith("MT-")
# ribosomal Genes
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
# hemoglobin Genes
adata.var["hb"] = adata.var_names.str.contains(r"^HB[ABDEGMQZ]\d*(?!\w)")

sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, percent_top=[20], log1p=True)

# n_genes_by_counts : number of genes expressed in the count matrix 
# total_counts : total counts per cell
# pct_counts_mt : percentage of counts in mitochondrial genes

logging.info("QC metrics:\n%s", adata)
logging.info("Overall QC statistics\n%s",adata.obs[["total_counts", "n_genes_by_counts", "pct_counts_mt", "pct_counts_ribo"]].describe(percentiles=[.01, .5, .99]))
logging.info("Tumor QC statistics\n%s", adata[adata.obs["region"] == "T"].obs[["total_counts", "n_genes_by_counts", "pct_counts_mt", "pct_counts_ribo"]].describe(percentiles=[.01, .5, .99]))
logging.info("Lymph QC statistics\n%s", adata[adata.obs["region"] == "L"].obs[["total_counts", "n_genes_by_counts", "pct_counts_mt", "pct_counts_ribo"]].describe(percentiles=[.01, .5, .99]))

'''
1. Metadata / Sample Composition
   ├── patient_cell_counts
   ├── region_cell_counts
   └── sample_cell_counts

2. QC Metric Distributions
   ├── overall distributions
   └── log-scale distributions

3. QC Metrics by Group
   ├── patient violins
   ├── region violins
   └── sample violins

4. QC Relationships
   ├── all cells: counts vs genes
   ├── all cells: counts vs MT
   └── all cells: genes vs MT

5. QC Relationships by Group
   ├── patient_qc_scatter.jpg
   ├── region_qc_scatter.jpg
   └── sample_qc_scatter.jpg

6. Sample-level QC
   ├── counts by sample
   └── MT by sample

7. Tumor / Lymph QC
   ├── tumor by patient
   └── lymph by patient

8. Filtering Diagnostics
   ├── genes per cell
   ├── cells per gene
   ├── mitochondrial percentage
   └── filtering QC space

'''

# ----------- METADATA/SAMPLE COMPOSITION ---------- #

logging.info("[4] Generating metadata QC plots")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), dpi=300, sharey=True)
x = adata.obs['n_genes']
x_lowerbound = 1500
x_upperbound = 2000
nbins=100

sns.histplot(x, ax=ax1, norm_hist=True, bins=nbins)
sns.histplot(x, ax=ax2, norm_hist=True, bins=nbins)
sns.histplot(x, ax=ax3, norm_hist=True, bins=nbins)

ax2.set_xlim(0,x_lowerbound)
ax3.set_xlim(x_upperbound, adata.obs['n_genes'].max() )

for ax in (ax1,ax2,ax3): 
  ax.set_xlabel('')

ax1.title.set_text('n_genes')
ax2.title.set_text('n_genes, lower bound')
ax3.title.set_text('n_genes, upper bound')

fig.text(-0.01, 0.5, 'Frequency', ha='center', va='center', rotation='vertical', size='x-large')
fig.text(0.5, 0.0, 'Genes expressed per cell', ha='center', va='center', size='x-large')

fig.tight_layout()

for column in QC_GROUPS:

    counts = adata.obs[column].value_counts()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x=counts.index.astype(str),
        y=counts.values,
        legend=False, 
        color=main_colors[0],
        ax=ax
    )
    ax.set_xlabel(column.capitalize())
    ax.set_ylabel("Number of barcodes")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    savefig(
        preprocessing_raw_dir / f"{column}_barcode_counts.jpg"
    )

# ----------- QC METRIC DISTRIBUTION ---------- #

logging.info("[5] Generating QC metric distribution plots")

# overall distribution
fig, axes = plt.subplots(1,3,figsize=(15, 4))
for ax, metric in zip(axes, QC_METRICS):
    sns.histplot(
        adata.obs[metric],
        bins=100,
        color=main_colors[0],
        ax=ax,
    )
    ax.set_title(metric)
    ax.set_xlabel(metric)
    ax.set_ylabel("Number of barcodes")
fig.tight_layout()
savefig(
    preprocessing_raw_dir/ "qc_metric_distributions.jpg"
)

# log-scale distribution
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, metric in zip(axes, ["total_counts", "n_genes_by_counts"]):
    sns.histplot(
        adata.obs[metric],
        bins=100,
        log_scale=True,
        color=main_colors[0],
        ax=ax,
    ) 
    ax.set_title(f"{metric} (log scale)")
    ax.set_xlabel(metric)
    ax.set_ylabel("Number of barcodes")
fig.tight_layout()
savefig(
    preprocessing_raw_dir / "qc_metric_distributions_log.jpg"
)

# ----------- QC METRICS BY GROUP ---------- #

logging.info("[6] Generating grouped QC plots")

for group in QC_GROUPS:

    # violin plots
    sc.pl.violin(
        adata,
        QC_METRICS,
        groupby=group,
        stripplot=False,
        multi_panel=True,
        rotation=45,
        show=False
    )
    plt.savefig(
        preprocessing_raw_dir / f"qc_violin_{group}.jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")

    # log-scale count distribution
    sc.pl.violin(
        adata,
        ["n_genes_by_counts", "total_counts"],
        groupby=group,
        stripplot=False,
        multi_panel=True,
        log=True,
        rotation=45,
        show=False
    )
    plt.savefig(
        preprocessing_raw_dir / f"qc_violin_{group}_log.jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")

# ----------- QC RELATIONSHIPS ---------- #

logging.info("[7] Generating QC relationship plots")

# counts vs detected genes
sc.pl.scatter(
    adata,
    x="total_counts",
    y="n_genes_by_counts",
    color="pct_counts_mt",
    alpha=0.6,
    size=2,
    show=False
)
savefig(
    preprocessing_raw_dir / "qc_scatter_counts_vs_genes.jpg"
)

# counts vs mitochondrial percentage
sc.pl.scatter(
    adata,
    x="total_counts",
    y="pct_counts_mt",
    color="n_genes_by_counts",
    alpha=0.6,
    size=2,
    show=False
)
savefig(
    preprocessing_raw_dir / "qc_scatter_counts_vs_mt.jpg"
)

# genes vs. mitochondrial percentage
sc.pl.scatter(
    adata,
    x="n_genes_by_counts",
    y="pct_counts_mt",
    color="total_counts",
    alpha=0.6,
    size=2,
    show=False
)
savefig(
    preprocessing_raw_dir / "qc_scatter_genes_vs_mt.jpg"
)

# ----------- QC RELATIONSHIPS BY GROUP ---------- #

logging.info("[8] Generating grouped QC relationship plots")

for group in ["patient", "region", "sample"]:

    sc.pl.scatter(
        adata,
        x="total_counts",
        y="n_genes_by_counts",
        color=group,
        alpha=0.6,
        size=2,
        show=False
    )
    savefig(
        preprocessing_raw_dir / f"qc_scatter_counts_genes_{group}.jpg"
    )

    values = adata.obs[column].dropna().unique()
    n_values = len(values)

    ncols = 2
    nrows = math.ceil(n_values / ncols)

    fig, axes = plt.subplots(
        ncols, nrows,
        figsize=(nrows * 5, 12),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    axes = np.atleast_1d(axes).ravel()
    scatter = None

    for ax, value in zip(axes, values):

        mask = adata.obs[column] == value

        scatter = ax.scatter(
            adata.obs.loc[mask, "total_counts"],
            adata.obs.loc[mask, "n_genes_by_counts"],
            c=adata.obs.loc[mask, "pct_counts_mt"],
            s=2,
            alpha=0.5
        )

        ax.set_title(str(value))
        ax.set_xlabel("Total counts")
        ax.set_ylabel("Detected genes")
        ax.grid(alpha=0.2)

    for ax in axes[n_values:]:
        ax.set_visible(False)

    if scatter is not None:
        fig.colorbar(
            scatter,
            ax=axes[:n_values].tolist(),
            pad=0.04,
            fraction=0.025,
            label="Mitochondrial counts (%)",
        )

    fig.savefig(
        preprocessing_raw_dir / f"qc_scatter_counts_genes_{group}_individual.jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

# ----------- SAMPLE LEVEL QC ---------- # 

logging.info("[9] Generating sample-level QC plots")  

# counts per sample
sc.pl.violin(
    adata,
    "total_counts",
    groupby="sample",
    stripplot=False,
    cut=0,
    rotation=45,
    log=True,
    show=False
)
savefig(
    preprocessing_raw_dir / "qc_total_counts_by_sample.jpg"
)

# mitochondrial percentage per sample
sc.pl.violin(
    adata,
    "pct_counts_mt",
    groupby="sample",
    stripplot=False,
    cut=0,
    rotation=45,
    show=False
)
savefig(
    preprocessing_raw_dir / "qc_mt_by_sample.jpg"
)

# ----------- TUMOR & LYMPH COMPARISON ---------- # 

logging.info("[10] Generating region-specific QC plots")

for region, name in [("T", "tumor"), ("L", "lymph")]:

    subset = adata[adata.obs["region"] == region]

    sc.pl.violin(
        subset,
        QC_METRICS,
        groupby="patient",
        stripplot=False,
        cut=0,
        rotation=45,
        multi_panel=True,
        show=False
    )
    plt.savefig(
        preprocessing_raw_dir / f"qc_metrics_{name}_by_patient.jpg",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close("all")

# ----------- FILTERING DIAGNOSTICS ---------- # 

logging.info("[11] Generating filtering diagnostics")

# detected genes per cell
logging.info(
    "Gene distribution:\n%s",
    adata.obs["n_genes_by_counts"].describe(percentiles=QC_PERCENTILES)
)

fig, ax = plt.subplots(figsize=(7, 5))
sns.histplot(
    adata.obs["n_genes_by_counts"],
    bins=100,
    color=main_colors[0],
    ax=ax
)
ax.axvline(
    MIN_GENES,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Threshold: {MIN_GENES}"
)
ax.set_xlabel("Number of detected genes")
ax.set_ylabel("Number of cells")
ax.legend()
fig.tight_layout()
savefig(
    preprocessing_raw_dir / "filter_genes_per_cell.jpg"
)

fig, ax = plt.subplots(figsize=(7, 5))
xmin = 0
xmax = MIN_GENES + 1000
sns.histplot(
    adata.obs["n_genes_by_counts"],
    bins=np.arange(xmin, xmax + 10, 10),
    color=main_colors[0],
    ax=ax
)
ax.axvline(
    MIN_GENES,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Threshold: {MIN_GENES}"
)
ax.set_xlabel("Number of detected genes")
ax.set_ylabel("Number of cells")
ax.set_xlim(xmin, xmax)
ax.legend()
fig.tight_layout()
savefig(
    preprocessing_raw_dir / "filter_genes_per_cell_zoomed.jpg"
)

# cells expressing each gene
logging.info(
    "Gene detection distribution:\n%s",
    adata.var["n_cells_by_counts"].describe(percentiles=QC_PERCENTILES)
)

fig, ax = plt.subplots(figsize=(7, 5))
sns.histplot(
    adata.var["n_cells_by_counts"],
    bins=100,
    color=main_colors[0],
    ax=ax
)
ax.axvline(
    MIN_CELLS,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Threshold: {MIN_CELLS}"
)
ax.set_xlabel("Number of cells expressing gene")
ax.set_ylabel("Number of genes")
ax.legend()
fig.tight_layout()
savefig(
    preprocessing_raw_dir / "filter_cells_per_gene.jpg"
)

fig, ax = plt.subplots(figsize=(7, 5))
xmin = max(0, MIN_CELLS - 5)
xmax = MIN_CELLS + 10
sns.histplot(
    adata.var["n_cells_by_counts"],    
    bins=np.arange(xmin, xmax + 10, 1),
    color=main_colors[0],
    ax=ax
)
ax.axvline(
    MIN_CELLS,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Threshold: {MIN_CELLS}"
)
ax.set_xlabel("Number of cells expressing gene")
ax.set_ylabel("Number of genes")
ax.set_xlim(xmin, xmax) 
fig.tight_layout()
savefig(
    preprocessing_raw_dir / "filter_cells_per_gene_zoomed.jpg"
)

# mitochondrial percentage
logging.info(
    "Mitochondrial percentage distribution:\n%s",
    adata.obs["pct_counts_mt"].describe(percentiles=QC_PERCENTILES)
)

fig, ax = plt.subplots(figsize=(7, 5))
sns.histplot(
    adata.obs["pct_counts_mt"],
    bins=100,
    color=main_colors[0],
    ax=ax
)
ax.axvline(
    MT_CUTOFF,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Threshold: {MT_CUTOFF}%"
)
ax.set_xlabel("Mitochondrial counts (%)")
ax.set_ylabel("Number of cells")
ax.set_xlim(0, 100)
ax.legend()
fig.tight_layout()
savefig(
    preprocessing_raw_dir / "filter_mitochondrial_percentage.jpg"
)

fig, ax = plt.subplots(figsize=(7, 5))
xmin = max(0, MT_CUTOFF - 10)
xmax = min(MT_CUTOFF + 10, 100)
sns.histplot(
    adata.obs["pct_counts_mt"],    
    bins=np.arange(xmin, xmax + 10, 1),
    color=main_colors[0],
    ax=ax
)
plt.axvline(
    MT_CUTOFF,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Threshold: {MT_CUTOFF}%"
)
ax.set_xlabel("Mitochondrial counts (%)")
ax.set_ylabel("Number of cells") 
ax.set_xlim(xmin, xmax) 
plt.tight_layout()
fig.tight_layout()
savefig(
    preprocessing_raw_dir / "filter_mitochondrial_percentage_zoomed.jpg"
)

# ----------- FILTERING SPACE ---------- # 

logging.info("[12] Generating filtering space")

# show where the proposed thresholds fall in QC space

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    adata.obs["n_genes_by_counts"],
    adata.obs["pct_counts_mt"],
    c=adata.obs["total_counts"],
    s=2,
    alpha=0.5
)
ax.axvline(
    MIN_GENES,
    color="red",
    linestyle="--",
    label=f"Min genes = {MIN_GENES}"
)
ax.axhline(
    MT_CUTOFF,
    color="red",
    linestyle="--",
    label=f"MT cutoff = {MT_CUTOFF}%",
)
ax.set_xlabel("Number of detected genes")
ax.set_ylabel("Mitochondrial counts (%)")
fig.colorbar(
    scatter,
    ax=ax,
    label="Total counts",
)
ax.legend()
fig.tight_layout()
savefig(
    preprocessing_raw_dir / "filtering_qc_space.jpg"
)

# ----------- OVERALL QC FAILURE ANALYSIS ---------- #

logging.info("[13] Overall QC threshold analysis")

# cell-level QC criteria
low_genes = adata.obs["n_genes_by_counts"] < MIN_GENES
high_mt = adata.obs["pct_counts_mt"] >= MT_CUTOFF

# cells failing either criterion
qc_fail = low_genes | high_mt

# cells passing both criteria
qc_pass = ~qc_fail

logging.info(
    "Low genes (< %d): %d",
    MIN_GENES,
    low_genes.sum()
)

logging.info(
    "High mitochondrial (>= %d%%): %d",
    MT_CUTOFF, 
    high_mt.sum()
)

logging.info(
    "Failing both criteria: %d",
    (low_genes & high_mt).sum()
)

logging.info(
    "Failing either criterion: %d",
    qc_fail.sum()
)

logging.info(
    "Retained: %d cells (%.2f%%)",
    qc_pass.sum(),
    qc_pass.mean() * 100
)

logging.info(
    "Removed: %d cells (%.2f%%)",
    qc_fail.sum(),
    qc_fail.mean() * 100
)

# breakdown of failure modes

qc_failure_modes = pd.Series(
    "pass",
    index=adata.obs.index,
    name="qc_failure"
)

qc_failure_modes.loc[low_genes & ~high_mt] = "low_genes"
qc_failure_modes.loc[~low_genes & high_mt] = "high_mt"
qc_failure_modes.loc[low_genes & high_mt] = "low_genes_and_high_mt"

logging.info(
    "QC failure modes:\n%s",
    qc_failure_modes.value_counts().to_string()
)

# ----------- QC FAILURE BY SAMPLE / PATIENT / REGION ---------- #

logging.info("[14] QC failure rates by sample")

qc_df = adata.obs[
    ["patient", "region", "sample"]
].copy()

qc_df["low_genes"] = low_genes.to_numpy()
qc_df["high_mt"] = high_mt.to_numpy()
qc_df["qc_fail"] = qc_fail.to_numpy()
qc_df["qc_pass"] = qc_pass.to_numpy()

# sample-level QC summary
sample_qc = (
    qc_df.groupby(
        ["patient", "region", "sample"],
        observed=True
    ).agg(
        total_cells=("qc_fail", "size"),
        failed_cells=("qc_fail", "sum"),
        low_gene_cells=("low_genes", "sum"),
        high_mt_cells=("high_mt", "sum")
    )
)

sample_qc["failure_rate"] = sample_qc["failed_cells"] / sample_qc["total_cells"]
sample_qc["retention_rate"] = 1 - sample_qc["failure_rate"]

logging.info(
    "QC summary by sample:\n%s",
    sample_qc.to_string()
)

logging.info(
    "Samples sorted by QC failure rate:\n%s",
    sample_qc
    .sort_values("failure_rate", ascending=False)
    .to_string()
)

# patient-level QC summary
patient_qc = (
    qc_df.groupby(
        "patient",
        observed=True
    ).agg(
        total_cells=("qc_fail", "size"),
        failed_cells=("qc_fail", "sum"),
        low_gene_cells=("low_genes", "sum"),
        high_mt_cells=("high_mt", "sum")
    )
)

patient_qc["failure_rate"] = patient_qc["failed_cells"] / patient_qc["total_cells"]
patient_qc["retention_rate"] = 1 - patient_qc["failure_rate"]

logging.info(
    "QC summary by patient:\n%s",
    patient_qc.to_string()
)

# region-level QC summary
region_qc = (
    qc_df.groupby(
        "region",
        observed=True
    ).agg(
        total_cells=("qc_fail", "size"),
        failed_cells=("qc_fail", "sum"),
        low_gene_cells=("low_genes", "sum"),
        high_mt_cells=("high_mt", "sum")
    )
)

region_qc["failure_rate"] = region_qc["failed_cells"] / region_qc["total_cells"]
region_qc["retention_rate"] = 1 - region_qc["failure_rate"]


logging.info(
    "QC summary by region:\n%s",
    region_qc.to_string()
)

# ----------- HIGH-MITOCHONDRIAL CELLS ---------- # 

logging.info("[15] Cells with highest mitochondrial percentage")

highest_mt = (
    adata.obs
    .sort_values("pct_counts_mt", ascending=False)
    [
        ["patient", "region", "sample", "n_genes_by_counts", 
         "total_counts", "pct_counts_mt",]
    ].head(10)
)

logging.info(
    "Top 10 cells by mitochondrial percentage:\n%s",
    highest_mt.to_string()
)

# ----------- HIGH-MITOCHONDRIAL CELLS BY SAMPLE ---------- #

logging.info("[16] Investigating high-mitochondrial cells by sample")

high_mt_by_sample = (
    qc_df.groupby(
        ["patient", "region", "sample"],
        observed=True
    ).agg(
        total_cells=("high_mt", "size"),
        high_mt_cells=("high_mt", "sum")
    )
)

high_mt_by_sample["high_mt_rate"] = high_mt_by_sample["high_mt_cells"] / high_mt_by_sample["total_cells"]

logging.info(
    "High-mitochondrial cells by sample:\n%s",
    high_mt_by_sample.sort_values("high_mt_rate", ascending=False).to_string()
)

# ----------- MITOCHONDRIAL THRESHOLD SENSITIVITY ---------- #

logging.info("[17] Mitochondrial threshold sensitivity analysis") 

threshold_results = []

for cutoff in MT_CUTOFFS:

    pass_mask = (
        (adata.obs["n_genes_by_counts"] >= MIN_GENES) &
        (adata.obs["pct_counts_mt"] < cutoff)
    )

    retained = pass_mask.sum()
    removed = (~pass_mask).sum()

    threshold_results.append(
        {
            "mt_cutoff": cutoff,
            "retained_cells": retained,
            "removed_cells": removed,
            "retention_rate": pass_mask.mean(),
            "removal_rate": (~pass_mask).mean(),
        }
    )

threshold_sensitivity = pd.DataFrame(threshold_results)

logging.info(
    "MT threshold sensitivity:\n%s",
    threshold_sensitivity.to_string(index=False)
)

# sample-level threshold sensitivity
sample_threshold_results = []

for cutoff in MT_CUTOFFS:

    pass_mask = (
        (adata.obs["n_genes_by_counts"] >= MIN_GENES) &
        (adata.obs["pct_counts_mt"] < cutoff)
    )

    temp = adata.obs[["patient", "region", "sample"]].copy()

    temp["qc_pass"] = pass_mask.to_numpy()

    summary = (
        temp.groupby(
            ["patient", "region", "sample"],
            observed=True,
        )["qc_pass"].agg(
            retained="sum",
            total="size",
        )
    )

    summary["retention_rate"] = summary["retained"] / summary["total"]

    summary["mt_cutoff"] = cutoff

    sample_threshold_results.append(summary.reset_index())

sample_threshold_sensitivity = pd.concat(
    sample_threshold_results,
    ignore_index=True
)

logging.info(
    "Sample-level MT threshold sensitivity:\n%s",
    sample_threshold_sensitivity.to_string(index=False)
)

# ----------- MITOCHONDRIAL DISTRIBUTIONS FOR SELECTED SAMPLES ---------- #

logging.info("[18] Generating mitochondrial distributions for selected samples")

for sample in SAMPLES_TO_INSPECT:

    sample_mask = adata.obs["sample"] == sample

    if not sample_mask.any():
        logging.warning(
            "Sample %s not found; skipping",
            sample
        )
        continue

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.histplot(
        adata.obs.loc[sample_mask, "pct_counts_mt"],
        bins=100,
        color=main_colors[0],
        ax=ax
    )
    ax.axvline(
        MT_CUTOFF,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold: {MT_CUTOFF}%",
    )
    plt.xlim(0, 100)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Mitochondrial counts (%)")
    ax.set_ylabel("Number of cells")
    ax.set_title(sample)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        preprocessing_raw_dir / f"mt_distribution_{sample}.jpg",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

# ----------- MITOCHONDRIAL BINS ----------------- #

logging.info("[19] Investigating QC metrics across mitochondrial bins")

mt_bins = pd.cut(
    adata.obs["pct_counts_mt"],
    bins=[0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100],
    include_lowest=True
)

mt_bin_summary = (
    adata.obs.groupby(
        mt_bins,
        observed=True,
    ).agg(
        cells=("pct_counts_mt", "size"),
        median_genes=("n_genes_by_counts", "median"),
        median_counts=("total_counts", "median"),
        median_mt=("pct_counts_mt", "median"),
        median_ribo=("pct_counts_ribo", "median")
    )
)

logging.info(
    "QC metrics by mitochondrial bin:\n%s",
    mt_bin_summary.to_string()
)

# mitochondrial bins by sample
mt_bin_by_sample = pd.crosstab(
    adata.obs["sample"],
    mt_bins,
    normalize="index"
) * 100

logging.info(
    "Mitochondrial bins by sample (%%):\n%s",
    mt_bin_by_sample.to_string()
)

# ----------- TARGET SAMPLE INVESTIGATION ----------------- #

logging.info("[20] Detailed QC statistics for selected samples")

for sample in TARGET_SAMPLES:

    sample_mask = adata.obs["sample"] == sample

    if not sample_mask.any():
        logging.warning(
            "Sample %s not found; skipping",
            sample,
        )
        continue

    logging.info(
        "QC statistics: %s \n%s",
        sample,
        adata.obs.loc[
            sample_mask,
            QC_METRICS,
        ].describe(
            percentiles=QC_PERCENTILES,
        ).to_string()
    )

# ----------- MITOCHONDRIAL QC GROUPS ----------------- #

logging.info("[21] Generating mitochondrial QC groups")

mt_qc_group = pd.cut(
    adata.obs["pct_counts_mt"],
    bins=[-np.inf, 20, 30, np.inf],
    labels=["low", "intermediate","high"],
    right=False
)

mt_group_summary = (
    adata.obs.assign(mt_qc_group=mt_qc_group).groupby(
        "mt_qc_group",
        observed=True
    )[QC_METRICS].describe()
)

logging.info(
    "QC metrics by mitochondrial group:\n%s",
    mt_group_summary.to_string()
)

# distribution by sample
mt_group_by_sample = (
    pd.crosstab(
        adata.obs["sample"],
        mt_qc_group,
        normalize="index"
    )* 100
)

logging.info(
    "Mitochondrial QC groups by sample (%%):\n%s",
    mt_group_by_sample.to_string()
)

# ----------- PCA INVESTIGATION ----------------- #

logging.info("[21] Investigate Confounders using PCA")

adata_pca = adata.copy()

# Normalizing to median total counts and add Size Factor
sc.pp.normalize_total(adata_pca)

# Logarithmize the data
sc.pp.log1p(adata_pca)

sc.tl.pca(adata_pca)

sc.pl.pca_variance_ratio(
    adata_pca, show=False
)
savefig(preprocessing_raw_dir / "pca_variance_ratio.jpg")
sc.pl.pca_loadings(
    adata_pca, components='1,2,3', 
    show=False
)
savefig(preprocessing_raw_dir / "pca_loadings_ratio.jpg")
sc.pl.pca(
    adata_pca, annotate_var_explained=True, 
    components=['1,2','3,4','1,4', '2,3'], 
    color="pct_counts_mt", show=False
)
savefig(preprocessing_raw_dir / "pca_pct_mt.jpg")
sc.pl.pca(
    adata_pca, annotate_var_explained=True, 
    components=['1,2','3,4','1,4', '2,3'], 
    color="pct_counts_ribo", show=False
)
savefig(preprocessing_raw_dir / "pca_pct_ribo.jpg")


# ============================================================
# ----------------------- SAVE DATA --------------------------
# ============================================================

logging.info("[22] Saving adata object as .h5 file")

adata.write_h5ad(
    save_path,
    compression="gzip",
)

logging.info("------------ Completed quality control ------------") 