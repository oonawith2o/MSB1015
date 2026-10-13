"""
Project:    Metabolic Reprogramming of Immune Response during Early Breast Cancer Dessimination
Script:     01_load_data.py 

Author:     Oona Kintscher
Created:    10.10.2026
Version:    1.0.0

"""

#----------- LIBRARY IMPORT -----------

import logging
import time, os
import harmonypy as hm
import pandas as pd
import scanpy as sc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from kennard_stone import train_test_split
from itertools import combinations
from pathlib import Path
from adjustText import adjust_text
from scipy.stats import median_abs_deviation
from matplotlib.colors import LinearSegmentedColormap

#------------------------------------

#----------- TIME STAMP -----------

timestamp = time.strftime('%Y%m%d')

#----------- DESIGN SETUP -----------

MAIN_COLOR = "#0E2841"
ACCENT_COLOR = "#4E95D9"

NAVY_WHITE = LinearSegmentedColormap.from_list(
    "navy_white",
    ["#0E2841", "#FFFFFF"]
)

COLOR_PALETTE = sns.color_palette(
    [NAVY_WHITE(x) for x in np.linspace(0,1,10)],
    as_cmap=True
)

COLOR_PALETTE_2 = [
    "#0072B2",  # Blue
    "#E69F00",  # Orange
    "#009E73",  # Bluish green
    "#CC79A7",  # Reddish purple
    "#56B4E9",  # Sky blue
    "#D55E00",  # Vermilion
    "#4D4D4D",  # Dark grey
    "#332288",  # Indigo
    "#88CCEE",  # Light cyan
    "#44AA99",  # Teal
    "#117733",  # Dark green
    "#999933",  # Olive
    "#DDCC77",  # Sand
    "#AA4499",  # Purple
    "#882255",  # Wine
]

sns.set_theme(
    style="whitegrid",
    context="notebook",
    font_scale=1.05,
    rc={
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#263746",
        "xtick.color": "#465563",
        "ytick.color": "#465563",
        "grid.color": "#D9E1E8",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
    }
)

#------------------------------------

#----------- LOGGER SETUP -----------

logging.basicConfig(filename=f'../log/{timestamp}-02_preprocessing.log', filemode='w', 
                    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.category").setLevel(logging.WARNING)

logging.info("------------ Starting preprocessing ------------") 

#------------------------------------

#----------- FILE NAMES -------------

DATA_FOLDER = "../data"

RESOURCE_FOLDER = Path("../results/preprocessing") / timestamp / "qc"
RESOURCE_FOLDER.mkdir(parents=True, exist_ok=True)

SC_EXP_FNAME = os.path.join(DATA_FOLDER, "processed_data/adata_raw.h5")
SC_EXP_SAVE_FNAME = os.path.join(DATA_FOLDER, "processed_data/adata_preprocessed.h5")

#------------------------------------

#----------- GLOBAL VARIABLES -------

SEED = 123
SAMPLING_PCT = 0.8

MIN_GENES = 300     # minimum number of genes detected per cell
MAX_GENES = 6000    # maximum number of genes detected per cell
MIN_CELLS = 5       # minimum number of cells detected specific gene
MT_CUTOFF = 30.0

QC_GROUPS = ["patient", "region", "sample"]

QC_OBS_METRICS = [
    "n_genes_by_counts",
    "total_counts",
    "pct_counts_mt",
]

QC_METRICS = [
    "n_genes_by_counts",
    "n_cells_by_counts",
    "pct_counts_mt",
]

QC_METRICS_NAMES = {
    "n_genes_by_counts": "Number of Expressed Genes per Barcode",
    "n_cells_by_counts": "Number of Barcodes Expressing Gene",
     "total_counts": "Number Counts per Barcode",
    "pct_counts_mt": "Percentage of Mitochondrial Counts"
}

NUMBER_OF_VARIABLE_GENES = 4000

PCA_COMPONENTS = ["1,2", "3,4", "1,4", "2,3"]
COMPONENT_PAIRS = [(0, 1), (2, 3), (0, 3), (1, 2)]

FIGSIZE = (15, 4.5)

#------------------------------------

#----------- FUNCTIONS --------------

def savefig(path):
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

def is_outlier(adata, metric: str, nmads: int):
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
        np.median(M) + nmads * median_abs_deviation(M) < M
    )
    return outlier

def style_pca_figure(fig, adata_pca):
    fig.set_size_inches(*FIGSIZE)

    variance = adata_pca.uns["pca"]["variance_ratio"] * 100

    pca_axes = [
        ax for ax in fig.axes
        if ax.get_visible()
        and ax.get_xlabel().startswith("PC")
        and ax.get_ylabel().startswith("PC")
    ]

    if len(pca_axes) != len(COMPONENT_PAIRS):
        pca_axes = [
            ax for ax in fig.axes
            if ax.get_visible() and ax.has_data()
        ][:len(COMPONENT_PAIRS)]

    for ax, (pc_x, pc_y) in zip(pca_axes, COMPONENT_PAIRS):
        ax.set_title("")

        ax.set_xlabel(
            f"PC{pc_x + 1} ({variance[pc_x]:.2f}%)",
            fontsize=12,
            labelpad=8,
        )
        ax.set_ylabel(
            f"PC{pc_y + 1} ({variance[pc_y]:.2f}%)",
            fontsize=12,
            labelpad=8,
        )

        ax.tick_params(axis="both", labelsize=9)
        ax.grid(axis="both", alpha=0.35, linewidth=0.6)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    return pca_axes

#------------------------------------

#####################################
#------------- MAIN -----------------
#####################################

# ----------- LOAD DATA ---------- #

logging.info("[1] Loading adata object") 

adata = sc.read_h5ad(SC_EXP_FNAME)
logging.info("AnnData object:\n%s", adata)

logging.info(
    "Initial dimensions: %d cells x %d genes",
    adata.n_obs,
    adata.n_vars,
)

#---------- THREE-PANEL SUMMARY PLOTS  ----------

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
    ax.set_xlabel(QC_METRICS_NAMES[metric], fontsize=12, labelpad=10)
    ax.set_ylabel("Number of Barcodes", fontsize=12, labelpad=10)
    ax.tick_params(axis="x", bottom=True, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", alpha=0.7)
    ax.grid(axis="x", visible=False)
axes[0].axvspan(MIN_GENES, MAX_GENES, alpha=0.1, color=ACCENT_COLOR)
axes[1].axvspan(MIN_CELLS, adata.var["n_cells_by_counts"].max(), alpha=0.1, color=ACCENT_COLOR)
axes[2].axvspan(0, MT_CUTOFF, alpha=0.1, color=ACCENT_COLOR)
fig.tight_layout()
savefig(RESOURCE_FOLDER / "filtering_panel_histplot_pre_filter.jpg")

#---------- FILTERING CELLS AND GENES  ----------

logging.info("[2] Filtering Cells and Genes") 

n_cells_before = adata.n_obs
n_genes_before = adata.n_vars

sc.pp.filter_genes(adata, min_cells=MIN_CELLS)

n_genes_after = adata.n_vars
n_genes_removed = (
    n_genes_before - n_genes_after
)

removed_pct = (
    100.0
    * n_genes_removed
    / n_genes_before
)

logging.info(
    "Genes: %d --> %d "
    "(removed: %d, %.3f%%)",
    n_genes_before,
    n_genes_after,
    n_genes_removed,
    removed_pct
)

sc.pp.filter_cells(adata, min_genes=MIN_GENES)
sc.pp.filter_cells(adata, max_genes=MAX_GENES)

n_cells_after = adata.n_obs
n_cells_removed = (
    n_cells_before - n_cells_after
)

removed_pct = (
    100.0
    * n_cells_removed
    / n_cells_before
)

logging.info(
    "Cells: %d --> %d "
    "(removed: %d, %.3f%%)",  
    n_cells_before,
    n_cells_after,
    n_cells_removed,
    removed_pct
)

#---------- MITOCHONDRIAL PERCENTAGE CUTOFF  ----------

logging.info("[3] Filtering cells with mitochondrial percentage")

n_cells_before_mt = adata.n_obs

mt_mask = adata.obs["pct_counts_mt"] <= MT_CUTOFF

adata = adata[mt_mask].copy()

n_cells_after_mt = adata.n_obs
n_removed_mt = (
    n_cells_before_mt - n_cells_after_mt
)

removed_pct = (
    100.0
    * n_removed_mt
    / n_cells_before_mt
)

logging.info(
    "Cells after mitochondrial filtering: %d --> %d "
    "(removed: %d, %.3f%%)",
    n_cells_before_mt,
    n_cells_after_mt,
    n_removed_mt,
    removed_pct
)

#---------- RECOMPUTE QC METRICS ----------

# mitochondrial Genes
adata.var["mt"] = adata.var_names.str.startswith("MT-")
# ribosomal Genes
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
# hemoglobin Genes
adata.var["hb"] = adata.var_names.str.contains(r"^HB[ABDEGMQZ]\d*(?!\w)")

sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, percent_top=[20], log1p=True)

#---------- THREE-PANEL SUMMARY PLOTS ----------

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
    ax.set_xlabel(QC_METRICS_NAMES[metric], fontsize=12, labelpad=10)
    ax.set_ylabel("Number of Barcodes", fontsize=12, labelpad=10)
    ax.tick_params(axis="x", bottom=True, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", alpha=0.7)
    ax.grid(axis="x", visible=False)
fig.tight_layout()
savefig(RESOURCE_FOLDER / "filtering_panel_histplot_post_filter.jpg")

COUNT_METRICS = ["total_counts", "n_genes_by_counts"]

for group in QC_GROUPS:

    n_metrics = len(QC_OBS_METRICS)
    fig, axes = plt.subplots(
        1,
        n_metrics,
        figsize=(5 * n_metrics, 4),
        squeeze=False,
    )
    axes = axes.ravel()
    fig.set_dpi(300)

    for ax, metric in zip(axes, QC_OBS_METRICS):

        plot_data = adata.obs[[group, metric]].dropna().copy()

        if metric in COUNT_METRICS:
            upper = plot_data[metric].quantile(0.99)
            plot_data = plot_data[plot_data[metric] <= upper]

        sns.violinplot(
            data=plot_data,
            x=group,
            y=metric,
            palette=COLOR_PALETTE,
            linewidth=1, 
            linecolor="k",
            inner = None,
            ax=ax
        )

        if metric in COUNT_METRICS:
            ax.set_yscale("log")
            ax.set_ylim(bottom=1)

        ax.set_xlabel("")
        ax.set_ylabel(
            QC_METRICS_NAMES.get(metric, "Value"),
            fontsize=12,
            labelpad=10
        )
        ax.tick_params(axis="x", bottom=True, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", alpha=0.7)
        ax.grid(axis="x", visible=False)

    fig.suptitle(
        f"QC Metrics by {group.replace('_', ' ').title()}",
        fontsize=14,
        fontweight="bold",
        color=MAIN_COLOR,
        y=1.03,
    )
    fig.tight_layout()
    savefig(RESOURCE_FOLDER / f"qc_violin_{group}_post_filter.jpg")

#---------- DOUBLET DETECTION AND FILTERING  ----------

logging.info("[4] Doublet detection") 

n_cells_before_doublet = adata.n_obs

sc.pp.scrublet(adata, batch_key="sample")

logging.info(
    "Scrublet score distribution:\n%s",
    adata.obs["doublet_score"].describe(
        percentiles=[0.50, 0.90, 0.95, 0.99, 0.999]
    ).to_string()
)

fig, ax = plt.subplots(figsize=(7, 5))
sns.histplot(
    adata.obs["doublet_score"],
    color=MAIN_COLOR,
    alpha=0.95,
    ax=ax
)
ax.tick_params(axis="x", bottom=True, labelsize=9)
ax.tick_params(axis="y", labelsize=9)
ax.grid(axis="y", alpha=0.7)
ax.grid(axis="x", visible=False)
ax.set_xlabel("Doublet Score", fontsize=12, labelpad=10)
ax.set_ylabel("Number of Barcodes", fontsize=12, labelpad=10)
fig.tight_layout()
savefig(RESOURCE_FOLDER / "doublet_score_distribution.jpg")

# overall doublet summary
n_doublets = int(adata.obs["predicted_doublet"].sum())
doublet_pct = 100.0 * n_doublets / n_cells_before_doublet

logging.info(
    "Predicted doublets: %d / %d (%.3f%%)",
    n_doublets,
    n_cells_before_doublet,
    doublet_pct,
)

# pre-sample doublet summary
doublet_summary = (
    adata.obs
    .groupby("sample", observed=True)["predicted_doublet"]
    .agg(
        n_doublets="sum",
        n_cells="count",
    )
)

doublet_summary["doublet_pct"] = (
    100.0
    * doublet_summary["n_doublets"]
    / doublet_summary["n_cells"]
)

logging.info(
    "Doublet summary by sample:\n%s",
    doublet_summary.to_string(),
)

# filter predicted doublets
adata = adata[
    ~adata.obs["predicted_doublet"]
].copy()

n_cells_after_doublet = adata.n_obs
n_removed_doublets = (
    n_cells_before_doublet - n_cells_after_doublet
)

removed_pct = (
    100.0
    * n_removed_doublets
    / n_cells_before_doublet
)

logging.info(
    "Cells after doublet filtering: %d --> %d "
    "(removed: %d, %.3f%%)",
    n_cells_before_doublet,
    n_cells_after_doublet,
    n_removed_doublets,
    removed_pct,
)

#---------- PRESERVE RAW COUNTS ---------------

adata.layers["counts"] = adata.X.copy()
adata.raw = adata.copy()

#---------- NORMALIZATION ---------------

logging.info("[5] Normalization and log transformation") 

# normalizing to median total counts and add size factor
# - makes data comparable across cells and mitigate the influence of cell-specific biases
# - total-count normalize (library-size correct) to 10,000 reads/cell
sc.pp.normalize_total(adata, target_sum=1e4)
adata.obs['size_factors'] = adata.obs.total_counts / np.median(adata.obs.total_counts)

# logarithmize the data
sc.pp.log1p(adata)

#---------- HIGHLY VARIABLE GENES ---------------

logging.info("[6] Highly variable genes") 

strategies = {
    "pooled": None,
    "region": "region",
    "patient": "patient",
    "sample": "sample",
}

for name, batch_key in strategies.items():

    sc.pp.highly_variable_genes(
        adata, 
        layer="counts",
        n_top_genes=NUMBER_OF_VARIABLE_GENES,
        flavor="seurat_v3",
        batch_key=batch_key
    )

    gene_labels_rank = (
        adata.var.loc[adata.var["highly_variable"]]
        .sort_values("highly_variable_rank", ascending=True)
        .head(5).index
    )

    sc.pl.highly_variable_genes(adata, show=False)
    fig = plt.gcf()
    ax1, ax2 = fig.axes[:2]

    texts1, texts2 = [], []
    x = []
    y = []
    
    for gene in gene_labels_rank:
        row = adata.var.loc[gene]
        texts1.append(
            ax1.text(
                row["means"], row["variances_norm"], gene,
                fontsize=10, color=ACCENT_COLOR,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor=ACCENT_COLOR,
                    linewidth=0.8,
                    alpha=0.9
                )
            )
        )
        texts2.append(
            ax2.text(
                row["means"], row["variances"], gene,
                fontsize=10, color=ACCENT_COLOR,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor=ACCENT_COLOR,
                    linewidth=0.8,
                    alpha=0.9
                )
            )
        )

    adjust_text(
        texts1,
        ax=ax1,
        arrowprops=dict(arrowstyle="-", color=ACCENT_COLOR, lw=0.8)
    )
    
    adjust_text(
        texts2,
        ax=ax2,
        arrowprops=dict(arrowstyle="-", color=ACCENT_COLOR, lw=0.8)
    )

    ax1.get_legend().remove()
    ax2.legend(
        title="HVG",
        loc="center left",
        title_fontproperties={"weight": "bold"},
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        fontsize=10
    )

    for ax in (ax1, ax2):
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(axis="both", alpha=0.7)
        ax.set_xlabel("Mean Expression of Genes", fontsize=12, labelpad=10)
    ax1.set_ylabel("Variances of Genes (normalized)", fontsize=12, labelpad=10)
    ax2.set_ylabel("Variances of Genes (not normalized)", fontsize=12, labelpad=10)
    fig.tight_layout()
    savefig(RESOURCE_FOLDER / f"highly_variable_genes_by_{name}.jpg")

    gene_labels_var = (
        adata.var.loc[adata.var["highly_variable"]]
        .sort_values("variances", ascending=False)
        .head(5).index
    )

    gene_labels_var_norm = (
        adata.var.loc[adata.var["highly_variable"]]
        .sort_values("variances_norm", ascending=False)
        .head(5).index
    )

    sc.pl.highly_variable_genes(adata, show=False)
    fig = plt.gcf()
    ax1, ax2 = fig.axes[:2]

    texts1, texts2 = [], []
    x = []
    y = []

    for gene in gene_labels_var_norm:
        row = adata.var.loc[gene]
        texts1.append(
            ax1.text(
                row["means"], row["variances_norm"], gene,
                fontsize=10, color=ACCENT_COLOR,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor=ACCENT_COLOR,
                    linewidth=0.8,
                    alpha=0.9
                )
            )
        )

    for gene in gene_labels_var:
        row = adata.var.loc[gene]
        texts2.append(
            ax2.text(
                row["means"], row["variances"], gene,
                fontsize=10, color=ACCENT_COLOR,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor=ACCENT_COLOR,
                    linewidth=0.8,
                    alpha=0.9
                )
            )
        )
    
    adjust_text(
        texts1,
        ax=ax1,
        avoid_self=True,
        prevent_crossings=True,
        arrowprops=dict(arrowstyle="-", color=ACCENT_COLOR, lw=0.8)
    )

    adjust_text(
        texts2,
        ax=ax2,
        avoid_self=True,
        prevent_crossings=True,
        arrowprops=dict(arrowstyle="-", color=ACCENT_COLOR, lw=0.8)
    )
    ax1.get_legend().remove()
    ax2.legend(
        title="HVG",
        loc="center left",
        title_fontproperties={"weight": "bold"},
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        fontsize=10
    )
    for ax in (ax1, ax2):
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(axis="both", alpha=0.7)
        ax.set_xlabel("Mean Expression of Genes", fontsize=12, labelpad=10)
    ax1.set_ylabel("Variances of Genes (normalized)", fontsize=12, labelpad=10)
    ax2.set_ylabel("Variances of Genes (not normalized)", fontsize=12, labelpad=10)
    plt.tight_layout()
    savefig(RESOURCE_FOLDER / f"highly_variable_genes_by_{name}_with_labels.jpg")

    cols = [
        "highly_variable",
        "highly_variable_rank",
        "means",
        "variances",
        "highly_variable_nbatches",
    ]
    
    logging.info(
        "========== HVG by %s ==========",
        name
    )

    logging.info(
        "Overview of HVG \n%s", 
        adata.var[
            ["means", "variances", "variances_norm", "highly_variable"]
        ].describe().to_string()
    )
    logging.info(
        "Overview of Top-Ranked HVGs \n%s",
        adata.var[
            [c for c in cols if c in adata.var.columns]
        ].sort_values("highly_variable_rank").head(20).to_string()
    )
    logging.info(
        "Genes w/ largest Variance \n%s",
        adata.var.nlargest(10, "variances")[
            ["means", "variances", "highly_variable"]
        ].to_string()
    )
    adata.var[f"hvg_{name}"] = adata.var["highly_variable"].copy()

# Investigate Overlap Between HVG

hvg_cols = [
    "hvg_pooled",
    "hvg_region",
    "hvg_sample",
    "hvg_patient"
]

for col1, col2 in combinations(hvg_cols, 2):
    set1 = set(adata.var_names[adata.var[col1]])
    set2 = set(adata.var_names[adata.var[col2]])

    intersection = set1 & set2
    union = set1 | set2

    logging.info(f"--- {col1} vs {col2} ---")
    logging.info(f"Shared genes: {len(intersection)}")
    logging.info(f"Jaccard similarity: {len(intersection) / len(union):.3f}")
    logging.info(f"Fraction of overlapping: {len(set1 & set2) / len(set1):.1%}")

overlap = pd.DataFrame(
    index=hvg_cols,
    columns=hvg_cols,
    dtype=float,
)

for col1 in hvg_cols:
    for col2 in hvg_cols:
        set1 = set(adata.var_names[adata.var[col1]])
        set2 = set(adata.var_names[adata.var[col2]])

        overlap.loc[col1, col2] = len(set1 & set2) / len(set1)

sns.heatmap(
    overlap,
    annot=True,
    fmt=".2f",
    cmap=NAVY_WHITE.reversed(),
    vmin=0,
    vmax=1
)
fig = plt.gcf()
labels = [col.replace("hvg_", "").capitalize() for col in hvg_cols]
fig.axes[0].set_xticklabels(labels)
fig.axes[0].set_yticklabels(labels, rotation=0)
fig.axes[0].set_xlabel("HVG Selection Method by Batch Key", fontsize=12, labelpad=10)
plt.title("HVG Set Overlap")
plt.tight_layout()
savefig(RESOURCE_FOLDER / "hvg_overlap_heatmap.jpg")

#---------- MOST ABUNDANT GENES ---------------

X = adata.layers["counts"]
gene_counts = np.asarray(adata.layers["counts"].sum(axis=0)).ravel()

abundance = pd.DataFrame({
    "gene": adata.var_names,
    "total_counts": gene_counts,
})

abundance["pct_total_counts"] = (
    abundance["total_counts"] / abundance["total_counts"].sum() * 100
)

abundance = abundance.sort_values(
    "total_counts", ascending=False
).reset_index(drop=True)

logging.info(
    "Highly Abundant Genes \n%s",
    abundance.head(15).to_string(index=False)
)

top_genes = abundance.head(20).copy()

plt.figure(figsize=(10, 7))

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(
    data=top_genes,
    x="pct_total_counts",
    y="gene",
    color=MAIN_COLOR,
    alpha=0.95,
    legend=False
)
for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.2f",
        padding=4,
        fontsize=10,
        color="#263746"
    )
ax.set_xlabel("Percentage of Total Counts (%)", fontsize=12, labelpad=10)
ax.set_ylabel("Gene", fontsize=12, labelpad=10)
ax.tick_params(axis="x", labelsize=9)
ax.tick_params(axis="y", labelsize=9)
ax.grid(axis="x", linestyle="--", alpha=0.7)
ax.grid(axis="y", visible=False)
plt.title("Top 20 Most Abundant Genes")
plt.tight_layout()
savefig(RESOURCE_FOLDER / "most_abundant_genes.jpg")

sample_ids = adata.obs["sample"].astype(str).to_numpy()
unique_samples = pd.unique(sample_ids)

sample_abundance = []

for sample in unique_samples:
    mask = sample_ids == sample
    sample_X = X[mask]

    totals = np.asarray(sample_X.sum(axis=0)).ravel()
    total_counts = totals.sum()

    if total_counts == 0:
        continue

    sample_abundance.append(
        pd.DataFrame({
            "sample": sample,
            "gene": adata.var_names,
            "total_counts": totals,
            "pct_sample_counts": totals / total_counts * 100,
        })
    )

sample_abundance = pd.concat(
    sample_abundance, ignore_index=True
)

top_gene_names = abundance.head(15)["gene"]

plot_data = sample_abundance[
    sample_abundance["gene"].isin(top_gene_names)
]

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(
    data=plot_data,
    x="sample",
    y="pct_sample_counts",
    hue="gene",
    palette=COLOR_PALETTE_2,
    ax=ax
)
ax.legend(
    title="Gene",
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0
)
ax.set_xlabel("Samples", fontsize=12, labelpad=10)
ax.set_ylabel("Percent of Gene by Sample Counts", fontsize=12, labelpad=10)
ax.tick_params(axis="x", labelsize=9)
ax.tick_params(axis="y", labelsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.7)
ax.grid(axis="x", visible=False)
plt.xticks(rotation=90)
savefig(RESOURCE_FOLDER / "most_abundant_genes_by_sample.jpg")

#---------- PCA FOR SAMPLING ---------------

logging.info("[7] PCA for random sampling")

sc.tl.pca(
    adata,
    n_comps=50,
    svd_solver="arpack"
)

#---------- RANDOM SAMPLING ---------------

logging.info("[8] Random sampling") 

selected = []

sample_counts = adata.obs["sample"].value_counts()
n_per_sample = int(sample_counts.min() * SAMPLING_PCT)

logging.info(f"Selecting {n_per_sample:,} barcodes per sample")

for sample, group in adata.obs.groupby("sample", observed=True):

    idx = group.index
    X = adata[idx].obsm["X_pca"][:, :20]
    X = np.asarray(X)

    X_train, X_test, idx_train, idx_test = train_test_split(
        X,
        idx.to_numpy(),
        test_size=n_per_sample
    )

    selected.extend(idx_test)

adata_balanced = adata[selected].copy()

logging.info(
    "Check balanced data\n%s",
    adata_balanced.obs["sample"].value_counts().sort_index().to_string()
)

#---------- DATA SCALING ---------------

# scale the data 
# - ensures that the expression levles of genes across cells to have a mean 0 and a variance of 1
# - high abundance genes do not dominate the signal simply due to their larger numerical values
sc.pp.scale(adata_balanced)

#---------- DIMENSIONALITY REDUCTION PRE BATCH CORRECTION ---------------

logging.info("[9] PCA")

# UMAP
sc.pp.neighbors(adata_balanced, n_neighbors=15, n_pcs=30)
sc.tl.umap(adata_balanced)
sc.pl.umap(
    adata_balanced,
    color="sample",
    palette=COLOR_PALETTE_2,
    size=2,
    show=False
)
savefig(RESOURCE_FOLDER / "umap_by_sample_pre_batch_correction.jpeg")
sc.pl.umap(
    adata_balanced,
    color="region",
    palette=COLOR_PALETTE_2,
    size=2,
    show=False
)
savefig(RESOURCE_FOLDER / "umap_by_region_pre_batch_correction.jpeg")

#---------- BATCH CORRECTION ---------------

logging.info("[10] Batch correction")

ho = hm.run_harmony(
    adata_balanced.obsm["X_pca"],
    adata_balanced.obs,
    "sample"
)

adata_balanced.obsm["X_pca_harmony"] = ho.Z_corr

# get pca embeddings after batch correction
pca_embs = adata_balanced.obsm["X_pca_harmony"]

# ----------- METADATA/SAMPLE COMPOSITION ---------- #

logging.info("[11] Generating metadata QC plots")

for column in QC_GROUPS:

    counts = adata_balanced.obs[column].value_counts()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x=counts.index.astype(str),
        y=counts.values,
        legend=False, 
        color=MAIN_COLOR,
        ax=ax
    )
    ax.set_xlabel(column.capitalize())
    ax.set_ylabel("Number of barcodes")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    savefig(
        RESOURCE_FOLDER / f"{column}_barcode_counts.jpg"
    )

# ----------- QC METRICS BY GROUP ---------- #

logging.info("[12] Generating grouped QC plots")

COUNT_METRICS = ["total_counts", "n_genes_by_counts"]

for group in QC_GROUPS:

    n_metrics = len(QC_OBS_METRICS)
    fig, axes = plt.subplots(
        1,
        n_metrics,
        figsize=(5 * n_metrics, 4),
        squeeze=False,
    )
    axes = axes.ravel()
    fig.set_dpi(300)

    for ax, metric in zip(axes, QC_OBS_METRICS):

        plot_data = adata.obs[[group, metric]].dropna().copy()

        if metric in COUNT_METRICS:
            upper = plot_data[metric].quantile(0.99)
            plot_data = plot_data[plot_data[metric] <= upper]

        sns.violinplot(
            data=plot_data,
            x=group,
            y=metric,
            palette=COLOR_PALETTE,
            linewidth=1, 
            linecolor="k",
            inner = None,
            ax=ax
        )

        if metric in COUNT_METRICS:
            ax.set_yscale("log")
            ax.set_ylim(bottom=1)

        ax.set_xlabel("")
        ax.set_ylabel(
            QC_METRICS_NAMES.get(metric, "Value"),
            fontsize=12,
            labelpad=10
        )
        ax.tick_params(axis="x", bottom=True, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", alpha=0.7)
        ax.grid(axis="x", visible=False)

    fig.suptitle(
        f"QC Metrics by {group.replace('_', ' ').title()}",
        fontsize=14,
        fontweight="bold",
        color=MAIN_COLOR,
        y=1.03,
    )
    fig.tight_layout()
    savefig(RESOURCE_FOLDER / f"qc_violin_{group}.jpg")

# ----------- QC RELATIONSHIPS ---------- #

logging.info("[13] Generating QC relationship plots")

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
    fig, ax = plt.subplots(figsize=(5, 4))

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
        fontsize=12,
        labelpad=10,
    )

    ax.set_ylabel(
        QC_METRICS_NAMES.get(config["y"], config["y"]),
        fontsize=12,
        labelpad=10,
    )

    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)

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
        fontsize=10,
        labelpad=8,
    )
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    savefig(RESOURCE_FOLDER / config["filename"])

#---------- DIMENSIONALITY REDUCTION ---------------

logging.info("[14] Dimensionality reduction")

sc.pp.neighbors(adata_balanced, n_neighbors=15, n_pcs=30, use_rep="X_pca_harmony")

sc.tl.pca(adata_balanced)

sc.pl.pca_variance_ratio(
    adata_balanced, show=False
)
savefig(RESOURCE_FOLDER / "pca_variance_ratio_post_batch_correction.jpg")

sc.pl.pca_loadings(
    adata_balanced, components='1,2,3', 
    show=False
)
savefig(RESOURCE_FOLDER / "pca_loadings_ratio_post_batch_correction.jpg")

fig_region = sc.pl.pca(
    adata_balanced,
    annotate_var_explained=False,
    components=PCA_COMPONENTS,
    color="region",
    palette=[MAIN_COLOR, ACCENT_COLOR],
    show=False,
    return_fig=True
)

pca_axes = style_pca_figure(fig_region, adata_balanced)

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

fig_region.subplots_adjust(
    left=0.06,
    right=0.86,
    bottom=0.20,
    top=0.96,
    wspace=0.35
)

if handles:
    fig_region.legend(
        handles,
        labels,
        title="Region",
        loc="center left",
        bbox_to_anchor=(0.88, 0.5),
        fontsize=9,
        title_fontsize=10,
        frameon=False
    )

savefig(RESOURCE_FOLDER / "pca_adata_post_batch_correction.jpg")

# pca colored by pct_counts_mt
fig_mt = sc.pl.pca(
    adata_balanced,
    annotate_var_explained=False,
    components=PCA_COMPONENTS,
    color="pct_counts_mt",
    cmap=NAVY_WHITE,
    colorbar_loc=None,
    show=False,
    return_fig=True
)

pca_axes = style_pca_figure(fig_mt, adata_balanced)

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
    fontsize=10,
    labelpad=8
)
cbar.ax.tick_params(labelsize=8)
savefig(RESOURCE_FOLDER / "pca_pct_mt_post_batch_correction.jpg")

# pca colored by pct_counts_ribo
fig_mt = sc.pl.pca(
    adata_balanced,
    annotate_var_explained=False,
    components=PCA_COMPONENTS,
    color="pct_counts_ribo",
    cmap=NAVY_WHITE,
    colorbar_loc=None,
    show=False,
    return_fig=True
)

pca_axes = style_pca_figure(fig_mt, adata_balanced)

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
    fontsize=10,
    labelpad=8
)
cbar.ax.tick_params(labelsize=8)
savefig(RESOURCE_FOLDER / "pca_pct_ribo_post_batch_correction.jpg")

# UMAP
sc.tl.umap(adata_balanced)
sc.pl.umap(
    adata_balanced,
    color="sample",
    palette=COLOR_PALETTE_2,
    size=2,
    show=False
)
savefig(RESOURCE_FOLDER / "umap_by_sample_post_batch_correction.jpeg")
sc.pl.umap(
    adata_balanced,
    color="region",
    palette=COLOR_PALETTE_2,
    size=2,
    show=False
)
savefig(RESOURCE_FOLDER / "umap_by_region_post_batch_correction.jpeg")
sc.pl.umap(
    adata_balanced,
    color="Ki67",
    palette=COLOR_PALETTE_2,
    size=2,
    show=False
)
savefig(RESOURCE_FOLDER / "umap_by_Ki67_post_batch_correction.jpeg")

#---------- FEATURE SELECTION & STABILITY ---------------

'''
data_matrix = pd.DataFrame(
        adata_balanced.X,
        index=adata_balanced.obs_names,
        columns=adata_balanced.var_names
    )

feature_stability_HV = ca.assess_feature_stability(data_matrix = data_matrix, feature_set = highly_var_genes_sorted, steps = [500, 1000, 1500, 2000, 2500, 3000], feature_type = 'HV', resolution = [0.3, 0.5, 0.7], n_repetitions=50, algorithm='leiden', ncores=1)
feature_stability_MA = ca.assess_feature_stability(data_matrix = data_matrix, feature_set = most_abundant_genes_sorted, steps = [500, 1000, 1500, 2000, 2500, 3000], feature_type = 'MA', resolution = [0.3, 0.5, 0.7], n_repetitions=50, algorithm='leiden', ncores=1)
ca.plot_feature_overall_stability_boxplot([feature_stability_HV, feature_stability_MA])
ca.plot_feature_overall_stability_incremental([feature_stability_HV, feature_stability_MA])
'''

# ============================================================
# ----------------------- SAVE DATA --------------------------
# ============================================================

logging.info("[15] Saving adata object as .h5 file")

adata_balanced.write_h5ad(
    SC_EXP_SAVE_FNAME,
    compression="gzip",
)

logging.info("------------ Completed Preprocessing ------------") 