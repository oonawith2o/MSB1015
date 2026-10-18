"""
Project:    Metabolic Reprogramming of Immune Response during Early Breast Cancer Dessimination
Script:     01_load_data.py 

Author:     Oona Kintscher
Created:    17.10.2026

"""

# =============================================================================
# IMPORTS
# =============================================================================

import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import config.plotting as plot

from config.plotting import (
    MAIN_COLOR,
    ACCENT_COLOR,
    COLOR_PALETTE,
    COLOR_PALETTE_2,
    NAVY_WHITE
)

from config.settings import (
    SC_EXP_FNAME,
    SC_FEATURES_FNAME,
    SC_BARCODES_FNAME,
    PATIENT_INFO_FNAME,
    SC_EXP_RAW_FNAME,
    QC_DIR,
    setup_logging
)

# =============================================================================
# SETUP LOGGER
# =============================================================================

logger = setup_logging("01_load_data")

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

QC_GROUPS = ["patient", "region", "sample"]

QC_METRICS = [
    "n_genes_by_counts",
    "n_cells_by_counts",
    "pct_counts_mt",
]

QC_OBS_METRICS = [
    "n_genes_by_counts",
    "total_counts",
    "pct_counts_mt",
]

QC_METRICS_NAMES = {
    "n_genes_by_counts": "Number of Expressed Genes per Barcode",
    "n_cells_by_counts": "Number of Barcodes Expressing Gene",
    "total_counts": "Number Counts per Barcode",
    "pct_counts_mt": "Percentage of Mitochondrial Counts",
}

QC_PERCENTILES = [
    0.01, 0.05, 0.10, 0.25, 0.50,
    0.75, 0.90, 0.95, 0.99,
]

# Minimum of 200 detected genes per cell
# Minimum of 3 detected cells per gene
# Maximum of 20% mitochondrial counts

ST_MIN_CELLS = 3
ST_MIN_GENES = 200
ST_MT_CUTOFF = 20.0

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

PCA_COMPONENTS = ["1,2", "3,4", "1,4", "2,3"]

# =============================================================================
# MAIN WORKFLOW
# =============================================================================

# -----------------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------------

logger.info("[1] Loading data and creating AnnData object") 

# load patient table data
patient_variables = ["age", "sex", "tumor_size", "lymph_nodes", "cancer_type", "grade",
                     "ER", "PR", "HER2","Ki67", "stage", "treatment"]
patients = pd.read_excel(
    PATIENT_INFO_FNAME, 
    header=1, 
    names=patient_variables,
    index_col=0
)
patients.index = {"2","3","6","7"}
patients["tumor_volume"] = patients["tumor_size"].apply(
    lambda x: eval(x) if pd.notna(x) else None
)
patient_variables.append("tumor_volume")
logger.info("Patient information table\n%s",patients)

# load single-cell data
features = pd.read_csv(
    SC_FEATURES_FNAME,
    sep="\t",
    header=None,
    dtype=str
)

barcodes = pd.read_csv(
    SC_BARCODES_FNAME,
    sep="\t",
    header=None,
    dtype=str
)

adata = sc.read_mtx(SC_EXP_FNAME)
adata = adata.transpose()

logger.info("AnnData shape (cells × genes): %s", adata.shape)
logger.info("Expression matrix: %s, dtype=%s", type(adata.X).__name__, adata.X.dtype)
logger.info("Features shape: %s", features.shape)
logger.info("Barcodes shape: %s", barcodes.shape)

X = adata.X
n_obs, n_vars = adata.shape
total_entries = n_obs * n_vars
nnz = X.nnz
missing_X = int(np.isnan(X.data).sum()) if X.dtype.kind in "fc" else 0
zeros = total_entries - nnz
nonzero_entries = total_entries - zeros - missing_X
sparsity = zeros / total_entries if total_entries else 0.0

logger.info(
    "UMI Count Matrix General Metrics\n" 
    "Total number of entries: %d\n" 
    "Zero entries: %d\n" 
    "Non-zero entries: %d\n" 
    "Missing values: %d\n" 
    "Sparsity percent: %d",
    total_entries,
    zeros,
    nonzero_entries,
    missing_X,
    sparsity * 100
)  

# -----------------------------------------------------------------------------
# 2. DATA ANNOTATION
# -----------------------------------------------------------------------------

logger.info("[2] Annotating cell and gene names")

# load and map barcodes onto adata object 
barcodes.rename(columns={0:'barcode'}, inplace=True)
barcodes.set_index('barcode', inplace=True)
logger.info(
    "Barcode Set General Metrics\n" \
    "Number of barcodes: %d\n" \
    "Missing values: %d\n" \
    "Duplicate barcode IDs: %d",
    len(barcodes),
    barcodes.isna().sum().sum(),
    barcodes.index.duplicated().sum()
)  
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

logger.info("Patient data mapping check\n%s", check.to_string())

# load and map features onto adata object 
features = features.rename(columns={0: "gene_symbol"}).copy()
features["gene_symbol"] = features["gene_symbol"].astype("string")

date_pattern = re.compile(
    r"^\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$",
    re.IGNORECASE
)
date_mask = features["gene_symbol"].str.match(date_pattern, na=False)
suspicious_gene_symbols = features.loc[date_mask, ["gene_symbol"]]

features.set_index("gene_symbol", inplace=True)

logger.info(
    "Feature Set General Metrics\n"
    "Number of features: %d\n"
    "Missing values: %d\n"
    "Duplicate gene symbols: %d\n"
    "Duplicates: %s\n"
    "Potentially converted gene symbols: %d\n"
    "Suspicious entries:\n%s",
    len(features),
    features.isna().sum().sum(),
    features.index.duplicated().sum(),
    features.loc[features.index.duplicated()].to_string(),
    len(suspicious_gene_symbols),
    suspicious_gene_symbols.to_string()
)

adata.var = features
adata.var_names = adata.var.index.astype(str) 
adata.var_names.name = "gene_symbol"  
adata.var_names_make_unique()

logger.info("obs_names name: %s", adata.obs_names.name)
logger.info("var_names name: %s", adata.var_names.name)
logger.info("First barcodes: %s", adata.obs_names[:5].tolist())
logger.info("First genes: %s", adata.var_names[:5].tolist())
logger.info("obs columns: %s", adata.obs.columns.tolist())
logger.info("var columns: %s", adata.var.columns.tolist())
logger.info("Patient counts:\n%s", adata.obs['patient'].value_counts())
logger.info("Region counts:\n%s", adata.obs['region'].value_counts())
logger.info("Sample counts:\n%s", adata.obs['sample'].value_counts())
logger.info("AnnData object:\n%s", adata)  

# -----------------------------------------------------------------------------
# 3. COMPUTE QC METRICS
# -----------------------------------------------------------------------------

logger.info("[3] Quality control metrics computation")

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

logger.info("QC metrics:\n%s", adata)
logger.info("Overall QC statistics\n%s",adata.obs[["total_counts", "n_genes_by_counts", "pct_counts_mt", "pct_counts_ribo"]].describe(percentiles=[.01, .5, .99]))
logger.info("Tumor QC statistics\n%s", adata[adata.obs["region"] == "T"].obs[["total_counts", "n_genes_by_counts", "pct_counts_mt", "pct_counts_ribo"]].describe(percentiles=[.01, .5, .99]))
logger.info("Lymph QC statistics\n%s", adata[adata.obs["region"] == "L"].obs[["total_counts", "n_genes_by_counts", "pct_counts_mt", "pct_counts_ribo"]].describe(percentiles=[.01, .5, .99]))

# -----------------------------------------------------------------------------
# 4. METADATA/SAMPLE COMPOSITION
# -----------------------------------------------------------------------------

logger.info("[4] Generating metadata QC plots")

for column in QC_GROUPS:

    counts = adata.obs[column].value_counts()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x=counts.index.astype(str),
        y=counts.values,
        hue=counts.index.astype(str),
        palette=COLOR_PALETTE[:counts.size],
        alpha=0.95,
        legend=False, 
        ax=ax
    )
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.0f",
            padding=4,
            fontsize=14,
            color="#263746"
        )
    ax.set_xlabel(column.replace("_", " ").capitalize(), fontsize=14, labelpad=10)
    ax.set_ylabel("Number of Barcodes", fontsize=14, labelpad=10)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    plot.savefig(fig, QC_DIR / f"barcode_distribution_by_{column}.jpg")

# -----------------------------------------------------------------------------
# 5. QC METRIC DISTRIBUTION
# -----------------------------------------------------------------------------

logger.info("[5] Generating QC metric distribution plots")

# overall distribution
fig, axes = plt.subplots(1,3,figsize=(15, 4))
for ax, metric in zip(axes, QC_METRICS):
    if metric == "n_cells_by_counts":
        sns.histplot(
                adata.var[metric],
                bins=100,
                color=MAIN_COLOR,
                alpha=0.95,
                ax=ax
        )
    else:
        sns.histplot(
                adata.obs[metric],
                bins=100,
                color=MAIN_COLOR,
                alpha=0.95,
                ax=ax
        )
    ax.set_xlabel(QC_METRICS_NAMES[metric], fontsize=14, labelpad=10)
    ax.set_ylabel("Number of Barcodes", fontsize=14, labelpad=10)
    ax.tick_params(axis="x", bottom=True, labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", alpha=0.7)
    ax.grid(axis="x", visible=False)
fig.tight_layout()
plot.savefig(fig, QC_DIR / "qc_metric_distribution_summary.jpg")

# log-scale distribution
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, metric in zip(axes, ["total_counts", "n_genes_by_counts"]):
    sns.histplot(
        adata.obs[metric],
        bins=100,
        log_scale=True, 
        color=MAIN_COLOR,
        alpha=0.95,
        ax=ax
    ) 
    ax.set_xlabel(QC_METRICS_NAMES[metric], fontsize=14, labelpad=10)
    ax.set_ylabel("Number of Barcodes", fontsize=14, labelpad=10)
    ax.tick_params(axis="x", bottom=True, labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", alpha=0.7)
    ax.grid(axis="x", visible=False)
fig.tight_layout()
plot.savefig(fig, QC_DIR / "qc_metric_distributions_summary_log.jpg")

# three-panel summary plots - genes expressed
x = adata.obs["n_genes_by_counts"].dropna()
nbins = 1000

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
axes = (ax1, ax2, ax3)
sns.histplot(
    x=x,
    bins=nbins,
    kde=True,
    color=MAIN_COLOR,
    alpha=0.7,
    ax=ax1
)
sns.histplot(
    x=x,
    bins=nbins,
    kde=True,
    color=MAIN_COLOR,
    alpha=0.7,
    ax=ax2,
)
ax2.set_xlim(0, 400)
sns.histplot(
    x=x,
    bins=nbins,
    kde=True,
    color=MAIN_COLOR,
    alpha=0.7,
    ax=ax3,
)
ax3.set_xlim(3000, 6000)
titles = [
    "Full Distribution",
    "Lower Range (0-400)",
    "Upper Range (3,000-6,000)",
]
for ax, title in zip(axes, titles):
    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        color=MAIN_COLOR,
        loc="center",
        pad=12,
    )
    ax.set_xlabel("")
    ax.tick_params(axis="x", bottom=True, labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", alpha=0.7)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax, left=True, bottom=False)
fig.supxlabel(
    "Genes Expressed per Barcode",
    fontsize=14,
    color="#263746",
    y=-0.04
)
plot.savefig(fig, QC_DIR / f"qc_metric_distributions_n_genes_by_counts_summary.jpg")

# three-panel summary plots - cells expressed
x = adata.var["n_cells_by_counts"].dropna()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), dpi=300, sharey=True)
axes = (ax1, ax2, ax3)
sns.histplot(
    x=x,
    bins=100,
    kde=True,
    color=MAIN_COLOR,
    alpha=0.7,
    ax=ax1
)
sns.histplot(
    x=x,
    binwidth=1,
    kde=True,
    color=MAIN_COLOR,
    alpha=0.7,
    ax=ax2,
)
ax2.set_xlim(0, 10)
sns.histplot(
    x=x,
    bins=100,
    kde=True,
    color=MAIN_COLOR,
    alpha=0.7,
    ax=ax3,
)
ax3.set_xlim(3000, x.max())
titles = [
    "Full Distribution",
    "Lower Range (0-10)",
    f"Upper Range (3,000-{x.max()})",
]
for ax, title in zip(axes, titles):
    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        color=MAIN_COLOR,
        loc="center",
        pad=12,
    )
    ax.set_xlabel("")
    ax.tick_params(axis="x", bottom=True, labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", alpha=0.7)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax, left=True, bottom=False)
fig.supxlabel(
    "Barcodes Expressing Gene",
    fontsize=14,
    color="#263746",
    y=-0.04
)
plot.savefig(fig, QC_DIR / f"qc_metric_distributions_n_cells_by_counts_summary.jpg")

# three-panel summary plots - mitochondrial percentage
x = adata.obs["pct_counts_mt"].dropna()
nbins = 1000

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), dpi=300, sharey=True)
axes = (ax1, ax2, ax3)
sns.histplot(
    x=x,
    bins=nbins,
    kde=True,
    color=MAIN_COLOR,
    alpha=0.7,
    ax=ax1
)
sns.histplot(
    x=x,
    bins=nbins,
    kde=True,
    color=MAIN_COLOR,
    alpha=0.7,
    ax=ax2,
)
ax2.set_xlim(0, 10)
sns.histplot(
    x=x,
    bins=nbins,
    kde=True,
    color=MAIN_COLOR,
    alpha=0.7,
    ax=ax3,
)
ax3.set_xlim(30, 50)
titles = [
    "Full Distribution",
    "Lower Range (0-10%)",
    "Upper Range (30-50%)",
]
for ax, title in zip(axes, titles):
    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        color=MAIN_COLOR,
        loc="center",
        pad=12,
    )
    ax.set_xlabel("")
    ax.tick_params(axis="x", bottom=True, labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", alpha=0.7)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax, left=True, bottom=False)
fig.supxlabel(
    "Mitochondrial Read Fraction per Barcode",
    fontsize=14,
    color="#263746",
    y=-0.02
)
fig.tight_layout()
plot.savefig(fig, QC_DIR / f"qc_metric_distributions_pct_counts_mt_summary.jpg")

# -----------------------------------------------------------------------------
# 6. QC METRICS BY GROUP
# -----------------------------------------------------------------------------

logger.info("[6] Generating grouped QC plots")

plot.plot_qc_violinplots(adata, QC_DIR)

# -----------------------------------------------------------------------------
# 7. QC RELATIONSHIPS
# -----------------------------------------------------------------------------

logger.info("[7] Generating QC relationship plots")

scatter_configs = [
    {
        "x": "total_counts",
        "y": "n_genes_by_counts",
        "color": "pct_counts_mt",
        "filename": "qc_scatter_counts_vs_genes.jpg",
    },
    {
        "x": "total_counts",
        "y": "pct_counts_mt",
        "color": "n_genes_by_counts",
        "filename": "qc_scatter_counts_vs_mt.jpg",
    },
    {
        "x": "n_genes_by_counts",
        "y": "pct_counts_mt",
        "color": "total_counts",
        "filename": "qc_scatter_genes_vs_mt.jpg",
    }
]

for config in scatter_configs:
    fig, ax = plt.subplots(figsize=(8, 6))

    x = adata.obs[config["x"]]
    y = adata.obs[config["y"]]
    c = adata.obs[config["color"]]

    valid = x.notna() & y.notna() & c.notna()

    # Scatter plot
    scatter = ax.scatter(
        x[valid],
        y[valid],
        c=c[valid],
        cmap=NAVY_WHITE,
        s=2,
        linewidths=0,
    )

    ax.set_xlabel(
        QC_METRICS_NAMES.get(config["x"], config["x"]),
        fontsize=14,
        labelpad=10,
    )

    ax.set_ylabel(
        QC_METRICS_NAMES.get(config["y"], config["y"]),
        fontsize=14,
        labelpad=10,
    )

    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    ax.grid(axis="y", alpha=0.7)
    ax.grid(axis="x", alpha=0.7)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cbar = fig.colorbar(
        scatter,
        ax=ax,
        location="right",
        pad=0.12,
        fraction=0.05,
    )
    cbar.set_label(
        QC_METRICS_NAMES.get(config["color"], config["color"]),
        fontsize=14,
        labelpad=8,
    )
    cbar.ax.tick_params(labelsize=14)

    fig.tight_layout()
    plot.savefig(fig, QC_DIR / config["filename"])

# -----------------------------------------------------------------------------
# 8. QC RELATIONSHIPS BY GROUP
# -----------------------------------------------------------------------------

logger.info("[8] Generating grouped QC relationship plots")

for group in ["patient", "region", "sample"]:

    x = adata.obs["total_counts"]
    y = adata.obs["n_genes_by_counts"]
    c = adata.obs[group].astype("category")

    valid = x.notna() & y.notna() & c.notna()
    categories = c[valid].cat.categories

    fig, ax = plt.subplots(figsize=(8, 4))

    for category, color in zip(categories, COLOR_PALETTE):
        mask = valid & (c == category)

        ax.scatter(
            x[mask],
            y[mask],
            color=color,
            s=2,
            linewidths=0,
            label=str(category)
        )

    ax.set_xlabel(
        QC_METRICS_NAMES.get("total_counts", "Number of Barcodes"),
        fontsize=14,
        labelpad=10,
        y=-0.02
    )

    ax.set_ylabel(
        QC_METRICS_NAMES.get("n_genes_by_counts", "Genes Expressed per Barcode"),
        fontsize=14,
        labelpad=10
    )

    ax.tick_params(axis="both", labelsize=14)
    ax.grid(axis="both", alpha=0.7)

    ax.legend(
        title=group.replace("_", " ").title(),
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=14,
        markerscale=3,
        frameon=True
    )

    fig.tight_layout()
    plot.savefig(fig, QC_DIR / f"qc_scatter_counts_vs_genes_{group}.jpg")

    values = c[valid].cat.categories
    n_values = len(values)

    ncols = 2
    nrows = math.ceil(n_values / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 4.5, nrows * 3.5),
        sharex=True,
        sharey=True,
        squeeze=False
    )

    axes = axes.ravel()
    scatter = None

    for ax, value in zip(axes, values):

        mask = adata.obs[group] == value

        scatter = ax.scatter(
            x[mask],
            y[mask],
            c=adata.obs.loc[mask, "pct_counts_mt"],
            cmap=NAVY_WHITE,
            s=3,
            alpha=0.65,
            linewidths=0,
            rasterized=True
        )

        ax.tick_params(axis="both", labelsize=14)
        ax.grid(axis="both", alpha=0.7)

        ax.set_title(
            value,
            fontsize=14,
            fontweight="bold",
            color=MAIN_COLOR,
            loc="center",
            pad=12
        )

    for ax in axes[n_values:]:
        ax.set_visible(False)

    fig.supxlabel(
        QC_METRICS_NAMES.get("total_counts", "Number of Barcodes"),
        fontsize=14,
        y=-0.02
    )

    fig.supylabel(
        QC_METRICS_NAMES.get("n_genes_by_counts", "Genes Expressed per Barcode"),
        fontsize=14
    )

    fig.suptitle(
        f"QC Metrics by {group.replace('_', ' ').title()}",
        fontsize=14,
        fontweight="bold",
        color=MAIN_COLOR,
        y=1.03
    )

    if scatter is not None:
        cbar = fig.colorbar(
            scatter,
            ax=axes[:n_values].tolist(),
            location="right",
            pad=0.12,
            fraction=0.05,
            label="Mitochondrial Read Fraction per Barcode"
        )
        cbar.set_label(
            QC_METRICS_NAMES.get(config["color"], config["color"]),
            fontsize=14,
            labelpad=8
        )
        
    plot.savefig(fig, QC_DIR / f"qc_scatter_counts_vs_genes_{group}_individual.jpg")

# -----------------------------------------------------------------------------
# 9. FILTERING DIAGNOSTICS
# -----------------------------------------------------------------------------

logger.info("[9] Generating filtering diagnostics")

# detected genes per cell
logger.info(
    "Gene distribution:\n%s",
    adata.obs["n_genes_by_counts"].describe(percentiles=QC_PERCENTILES)
)

fig, ax = plt.subplots(figsize=(7, 5))
xmin = 0
xmax = 800
sns.histplot(
    adata.obs["n_genes_by_counts"],
    bins=1000,
    color=MAIN_COLOR,
    alpha=0.95,
    ax=ax
)
ax.set_xlabel("Number of Expressed Genes", fontsize=14, labelpad=10)
ax.set_ylabel("Number of Barcodes", fontsize=14, labelpad=10)
ax.tick_params(axis="x", bottom=True, labelsize=14)
ax.tick_params(axis="y", labelsize=14)
ax.grid(axis="y", alpha=0.7)
ax.grid(axis="x", visible=False)
ax.set_xlim(xmin, xmax)
fig.tight_layout()
plot.savefig(fig, QC_DIR / "filter_genes_per_cell_zoomed.jpg")

# cells expressing each gene
logger.info(
    "Gene detection distribution:\n%s",
    adata.var["n_cells_by_counts"].describe(percentiles=QC_PERCENTILES)
)

fig, ax = plt.subplots(figsize=(7, 5))
xmin = 0
xmax = 12
sns.histplot(
    adata.var["n_cells_by_counts"],
    binwidth=1,
    color=MAIN_COLOR,
    alpha=0.95,
    ax=ax
)
ax.set_xlabel("Number of Barcodes", fontsize=14, labelpad=10)
ax.set_ylabel("Number of Expressed Genes", fontsize=14, labelpad=10)
ax.tick_params(axis="x", bottom=True, labelsize=14)
ax.tick_params(axis="y", labelsize=14)
ax.grid(axis="y", alpha=0.7)
ax.grid(axis="x", visible=False)
ax.set_xlim(xmin, xmax)
fig.tight_layout()
plot.savefig(fig, QC_DIR / "filter_cells_per_gene_zoomed.jpg")

# mitochondrial percentage
logger.info(
    "Mitochondrial percentage distribution:\n%s",
    adata.obs["pct_counts_mt"].describe(percentiles=QC_PERCENTILES)
)

fig, ax = plt.subplots(figsize=(7, 5))
xmin = 15
xmax = 100
sns.histplot(
    adata.obs["pct_counts_mt"],    
    bins=1000,
    color=MAIN_COLOR,
    alpha=0.95,
    ax=ax
)
ax.set_xlabel("Mitochondrial Read Fraction per Barcode", fontsize=14, labelpad=10)
ax.set_ylabel("Number of Barcodes", fontsize=14, labelpad=10)
ax.tick_params(axis="x", bottom=True, labelsize=14)
ax.tick_params(axis="y", labelsize=14)
ax.grid(axis="y", alpha=0.7)
ax.grid(axis="x", visible=False)
ax.set_xlim(xmin, xmax) 
plot.savefig(fig, QC_DIR / "filter_mitochondrial_percentage_zoomed.jpg")

# -----------------------------------------------------------------------------
# 10. OVERALL QC FAILURE ANALYSIS
# -----------------------------------------------------------------------------

logger.info("[10] Overall QC threshold analysis")

# cell-level QC criteria
low_genes = adata.obs["n_genes_by_counts"] < ST_MIN_GENES
high_mt = adata.obs["pct_counts_mt"] >= ST_MT_CUTOFF

# cells failing either criterion
qc_fail = low_genes | high_mt

# cells passing both criteria
qc_pass = ~qc_fail

logger.info(
    "Low genes (< %d): %d",
    ST_MIN_GENES,
    low_genes.sum()
)

logger.info(
    "High mitochondrial (>= %d%%): %d",
    ST_MT_CUTOFF, 
    high_mt.sum()
)

logger.info(
    "Failing both criteria: %d",
    (low_genes & high_mt).sum()
)

logger.info(
    "Failing either criterion: %d",
    qc_fail.sum()
)

logger.info(
    "Retained: %d cells (%.2f%%)",
    qc_pass.sum(),
    qc_pass.mean() * 100
)

logger.info(
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

logger.info(
    "QC failure modes:\n%s",
    qc_failure_modes.value_counts().to_string()
)

# -----------------------------------------------------------------------------
# 11. QC FAILURE BY SAMPLE / PATIENT / REGION
# -----------------------------------------------------------------------------

logger.info("[11] QC failure rates by sample")

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

logger.info(
    "QC summary by sample:\n%s",
    sample_qc.to_string()
)

logger.info(
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

logger.info(
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

logger.info(
    "QC summary by region:\n%s",
    region_qc.to_string()
)

# -----------------------------------------------------------------------------
# 12. HIGH-MITOCHONDRIAL CELLS
# -----------------------------------------------------------------------------

logger.info("[12] Cells with highest mitochondrial percentage")

highest_mt = (
    adata.obs
    .sort_values("pct_counts_mt", ascending=False)
    [
        ["patient", "region", "sample", "n_genes_by_counts", 
         "total_counts", "pct_counts_mt",]
    ].head(10)
)

logger.info(
    "Top 10 cells by mitochondrial percentage:\n%s",
    highest_mt.to_string()
)

# -----------------------------------------------------------------------------
# 13. HIGH-MITOCHONDRIAL CELLS BY SAMPLE
# -----------------------------------------------------------------------------

logger.info("[13] Investigating high-mitochondrial cells by sample")

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

logger.info(
    "High-mitochondrial cells by sample:\n%s",
    high_mt_by_sample.sort_values("high_mt_rate", ascending=False).to_string()
)

# -----------------------------------------------------------------------------
# 14. MITOCHONDRIAL THRESHOLD SENSITIVITY 
# -----------------------------------------------------------------------------

logger.info("[14] Mitochondrial threshold sensitivity analysis") 

threshold_results = []

for cutoff in MT_CUTOFFS:

    pass_mask = (
        (adata.obs["n_genes_by_counts"] >= ST_MIN_GENES) &
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
            "removal_rate": (~pass_mask).mean()
        }
    )

threshold_sensitivity = pd.DataFrame(threshold_results)

logger.info(
    "MT threshold sensitivity:\n%s",
    threshold_sensitivity.to_string(index=False)
)

# sample-level threshold sensitivity
sample_threshold_results = []

for cutoff in MT_CUTOFFS:

    pass_mask = (
        (adata.obs["n_genes_by_counts"] >= ST_MIN_GENES) &
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
            total="size"
        )
    )

    summary["retention_rate"] = summary["retained"] / summary["total"]

    summary["mt_cutoff"] = cutoff

    sample_threshold_results.append(summary.reset_index())

sample_threshold_sensitivity = pd.concat(
    sample_threshold_results,
    ignore_index=True
)

logger.info(
    "Sample-level MT threshold sensitivity:\n%s",
    sample_threshold_sensitivity.to_string(index=False)
)

# -----------------------------------------------------------------------------
# 15. MMITOCHONDRIAL DISTRIBUTIONS FOR SELECTED SAMPLES
# -----------------------------------------------------------------------------

logger.info("[15] Generating mitochondrial distributions for selected samples")

for sample in SAMPLES_TO_INSPECT:

    sample_mask = adata.obs["sample"] == sample

    if not sample_mask.any():
        logger.warning(
            "Sample %s not found; skipping",
            sample
        )
        continue

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(
        adata.obs.loc[sample_mask, "pct_counts_mt"],
        binwidth=1,
        color=MAIN_COLOR,
        alpha=0.95,
        ax=ax
    )
    ax.set_xlabel("Number of Barcodes", fontsize=14, labelpad=10)
    ax.set_ylabel("Number of Expressed Genes", fontsize=14, labelpad=10)
    ax.tick_params(axis="x", bottom=True, labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", alpha=0.7)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    plot.savefig(fig, QC_DIR / f"mt_distribution_{sample}.jpg")

# -----------------------------------------------------------------------------
# 16. MITOCHONDRIAL BINS
# -----------------------------------------------------------------------------

logger.info("[16] Investigating QC metrics across mitochondrial bins")

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

logger.info(
    "QC metrics by mitochondrial bin:\n%s",
    mt_bin_summary.to_string()
)

# mitochondrial bins by sample
mt_bin_by_sample = pd.crosstab(
    adata.obs["sample"],
    mt_bins,
    normalize="index"
) * 100

logger.info(
    "Mitochondrial bins by sample (%%):\n%s",
    mt_bin_by_sample.to_string()
)

# -----------------------------------------------------------------------------
# 17. TARGET SAMPLE INVESTIGATION 
# -----------------------------------------------------------------------------

logger.info("[17] Detailed QC statistics for selected samples")

for sample in TARGET_SAMPLES:

    sample_mask = adata.obs["sample"] == sample

    if not sample_mask.any():
        logger.warning(
            "Sample %s not found; skipping",
            sample,
        )
        continue

    logger.info(
        "QC statistics: %s \n%s",
        sample,
        adata.obs.loc[
            sample_mask,
            QC_OBS_METRICS,
        ].describe(
            percentiles=QC_PERCENTILES,
        ).to_string()
    )

# -----------------------------------------------------------------------------
# 18. MITOCHONDRIAL QC GROUPS 
# -----------------------------------------------------------------------------

logger.info("[18] Generating mitochondrial QC groups")

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
    )[QC_OBS_METRICS].describe()
)

logger.info(
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

logger.info(
    "Mitochondrial QC groups by sample (%%):\n%s",
    mt_group_by_sample.to_string()
)

# -----------------------------------------------------------------------------
# 19. PCA INVESTIGATION
# -----------------------------------------------------------------------------

logger.info("[19] Investigate Confounders using PCA")

adata_pca = adata.copy()

# Normalizing to median total counts and add Size Factor
sc.pp.normalize_total(adata_pca)

# Logarithmize the data
sc.pp.log1p(adata_pca)

sc.tl.pca(adata_pca)

sc.pl.pca_variance_ratio(
    adata_pca, show=False
)
plot.savefig(fig, QC_DIR / "pca_variance_ratio.jpg")

sc.pl.pca_loadings(
    adata_pca, components='1,2,3', 
    show=False
)
plot.savefig(fig, QC_DIR / "pca_loadings_ratio.jpg")

fig_region = sc.pl.pca(
    adata_pca,
    annotate_var_explained=False,
    components=PCA_COMPONENTS,
    color="region",
    palette=["#440154", "#FDE725"],
    show=False,
    return_fig=True
)

pca_axes = plot.style_pca_figure(fig_region, adata_pca)

handles, labels = [], []

for ax in pca_axes:
    legend = ax.get_legend()

    if legend is not None:
        h, l = ax.get_legend_handles_labels()

        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

        legend.remove()

if handles:
    fig_region.legend(
        handles,
        labels,
        title="Region",
        loc="center left",
        bbox_to_anchor=(0.88, 0.5),
        fontsize=14,
        title_fontsize=14,
        frameon=False
    )

plot.savefig(fig, QC_DIR / "pca_adata.jpg")

fig_region = sc.pl.pca(
    adata_pca,
    annotate_var_explained=False,
    components=PCA_COMPONENTS,
    color="Ki67",
    cmap=NAVY_WHITE,
    show=False,
    return_fig=True
)

pca_axes = plot.style_pca_figure(fig_region, adata_pca)

handles, labels = [], []

for ax in pca_axes:
    legend = ax.get_legend()

    if legend is not None:
        h, l = ax.get_legend_handles_labels()

        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

        legend.remove()

if handles:
    fig_region.legend(
        handles,
        labels,
        title="Ki67",
        loc="center left",
        bbox_to_anchor=(0.88, 0.5),
        fontsize=14,
        title_fontsize=14,
        frameon=False
    )

plot.savefig(fig, QC_DIR / "pca_Ki67.jpg")

# pca colored by pct_counts_mt
fig_mt = sc.pl.pca(
    adata_pca,
    annotate_var_explained=False,
    components=PCA_COMPONENTS,
    color="pct_counts_mt",
    cmap="viridis",
    colorbar_loc=None,
    show=False,
    return_fig=True
)

pca_axes = plot.style_pca_figure(fig_mt, adata_pca)

scatter = next(
    collection
    for ax in pca_axes
    for collection in ax.collections
    if collection.get_array() is not None
)

cbar = fig_mt.colorbar(
    scatter,
    ax=pca_axes,
    location="right",
    pad=0.025,
    fraction=0.025
)

cbar.set_label(
    "Mitochondrial Read Fraction per Barcode",
    fontsize=14,
    labelpad=8
)
cbar.ax.tick_params(labelsize=14)
plot.savefig(fig, QC_DIR / "pca_pct_mt.jpg")

# pca colored by pct_counts_ribo
fig_mt = sc.pl.pca(
    adata_pca,
    annotate_var_explained=False,
    components=PCA_COMPONENTS,
    color="pct_counts_ribo",
    cmap="viridis",
    colorbar_loc=None,
    show=False,
    return_fig=True
)

pca_axes = plot.style_pca_figure(fig_mt, adata_pca)

scatter = next(
    collection
    for ax in pca_axes
    for collection in ax.collections
    if collection.get_array() is not None
)

cbar = fig_mt.colorbar(
    scatter,
    ax=pca_axes,
    location="right",
    pad=0.025,
    fraction=0.025
)

cbar.set_label(
    "Ribosomal Read Fraction per Barcode",
    fontsize=14,
    labelpad=8
)
cbar.ax.tick_params(labelsize=14)
plot.savefig(fig, QC_DIR / "pca_pct_ribo.jpg")

# ============================================================
# ----------------------- SAVE DATA --------------------------
# ============================================================

logger.info("[20] Saving adata object as .h5 file")

adata.write_h5ad(
    SC_EXP_SAVE_FNAME,
    compression="gzip",
)

logger.info("------------ Completed quality control ------------") 