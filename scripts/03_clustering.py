import logging
import time, os
import pandas as pd
import scanpy as sc
import numpy as np
import seaborn as sns
import decoupler as dc
import matplotlib.pyplot as plt
from itertools import combinations
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

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

logging.basicConfig(filename=f'../log/{timestamp}-03_clustering.log', filemode='w', 
                    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.category").setLevel(logging.WARNING)

logging.info("----------- Starting clustering ------------") 

#------------------------------------

#----------- FILE NAMES -------------

DATA_FOLDER = "../data"

RESOURCE_FOLDER = Path("../results/clustering") / timestamp
RESOURCE_FOLDER.mkdir(parents=True, exist_ok=True)

SC_EXP_FNAME = os.path.join(DATA_FOLDER, "processed_data/adata_preprocessed.h5")
SC_EXP_SAVE_FNAME_LEGACY = os.path.join(DATA_FOLDER, "processed_data/adata_clustered_legacy.h5")
SC_EXP_SAVE_FNAME = os.path.join(DATA_FOLDER, "processed_data/adata_clustered.h5")

#------------------------------------

#----------- FUNCTIONS --------------

def savefig(path):
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

def remove_none_values(obj):
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if obj[key] is None:
                del obj[key]
            else:
                remove_none_values(obj[key])


def write_legacy_h5ad(adata, path):
    adata_export = adata.copy()

    remove_none_values(adata_export.uns)

    adata_export.write_h5ad(
        path,
        compression="gzip"
    )

def compare_clusterings(adata, resolutions):
    rows = []

    for r1, r2 in zip(resolutions[:-1], resolutions[1:]):
        c1 = adata.obs[f"leiden_res_{r1:.2f}"].astype(str)
        c2 = adata.obs[f"leiden_res_{r2:.2f}"].astype(str)

        rows.append({
            "res_low": r1,
            "res_high": r2,
            "n_clusters_low": c1.nunique(),
            "n_clusters_high": c2.nunique(),
            "ARI": adjusted_rand_score(c1, c2),
            "NMI": normalized_mutual_info_score(c1, c2)
        })

    return pd.DataFrame(rows)

def compute_seed_stability(adata, TEST_RESOLUTIONS):

    SEEDS=[0,1,2,3,4,10,21,42,123,2026]

    for res in TEST_RESOLUTIONS:
        for seed in SEEDS:
            key = f"leiden_r{res:.2f}_s{seed}"

            sc.tl.leiden(
                adata,
                resolution=res, 
                key_added=key,
                flavor="igraph",
                random_state=seed
            )

    seed_rows = []
    for res in TEST_RESOLUTIONS:
        scores = []

        for s1, s2 in combinations(SEEDS, 2):
            c1 = adata.obs[f"leiden_r{res:.2f}_s{s1}"]
            c2 = adata.obs[f"leiden_r{res:.2f}_s{s2}"]
            scores.append(adjusted_rand_score(c1,c2))

        seed_rows.append({
            "resolutions": res,
            "ARI_mean": np.mean(scores),
            "ARI_sd": np.std(scores),
            "ARI_min": np.min(scores),
            "n_clusters_mean": np.mean([
                adata.obs[f"leiden_r{res:.2f}_s{s}"].nunique() for s in SEEDS
            ])
        })

    return pd.DataFrame(seed_rows)



#------------------------------------

#----------- GLOBAL VARIABLES -------

SEED = 42

RESOLUTIONS = np.arange(0.2, 1.21, step=0.1)

MARKERS = ["PTPRC", "CD3D", "CD8A", "CD4", "CD79A", "LYZ", "PLVAP", "ACTA2", "KRT19"]
IMMUNE_MARKERS = ["PTPRC", "CD3D", "CD3E", "MS4A1", "CD79A", "NKG7", "GNLY", "LST1", "LYZ"]
NON_IMMUNE_MARKERS = ["EPCAM", "PECAM1", "COL1A1", "KRT19", "DCN", "VWF"]
IMMUNE_CELLS = {
    'Plasma cells',
    'Plasmacytoid dendritic cells',
    'B cells',
    'T cells',
    'NK cells',
    'Dendritic cells',
    'Macrophages',
    'Neutrophils'
}
IMMUNE_CELL_TYPE_MARKERS = {
    "B cells": ["MS4A1", "CD79A", "CD79B", "CD19", "BANK1", "CD22", "PAX5"],
    "Plasma cells": [
        "MZB1", "JCHAIN", "XBP1", "SDC1", "CD38", "TNFRSF17",
        "DERL3", "IGHG1", "IGHA1"
    ],
    "T cells": ["CD3D", "CD3E", "CD3G", "TRAC", "CD2", "IL7R"],
    "CD4 T cells": ["CD4", "IL7R", "CCR7", "TCF7", "SELL", "LEF1"],
    "CD8 T cells": ["CD8A", "CD8B", "GZMK", "GZMB", "PRF1", "NKG7"],
    "Tregs": ["FOXP3", "IL2RA", "CTLA4", "TIGIT", "IKZF2"],
    "NK cells": [
        "NCAM1", "NKG7", "GNLY", "KLRD1", "KLRF1", "KLRC1",
        "XCL1", "PRF1"
    ],
    "Monocytes": ["LYZ", "S100A8", "S100A9", "VCAN", "FCN1", "CD14", "LST1"],
    "Macrophages": ["CD68", "CD163", "MSR1", "MRC1", "FOLR2", "APOE", "C1QA", "C1QB"],
    "Dendritic cells": ["FCER1A", "CD1C", "CLEC10A", "HLA-DQA1", "HLA-DPB1", "CD1E"],
    "pDCs": ["IL3RA", "CLEC4C", "LILRA4", "TCF4", "SERPINF1", "GZMB", "DERL3"],
    "Mast cells": ["TPSB2", "TPSAB1", "CPA3", "MS4A2", "KIT", "CMA1"],
    "Fibroblasts": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "FAP", "PDGFRA"],
    "Endothelial cells": ["PECAM1", "VWF", "CLDN5", "CDH5", "EGFL7"],
    "Epithelial cells": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT5"],
}

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

#---------- MARKER GENE CURATION ---------------

logging.info("[2] Maker gene curation")

markers_raw = dc.op.resource(
    name='PanglaoDB',
    organism='human',
    license='academic'
)

markers_raw["human_sensitivity"] = pd.to_numeric(
    markers_raw["human_sensitivity"], errors="coerce"
)

human = markers_raw["human"].astype("boolean").fillna(False)
canonical = (
    markers_raw["canonical_marker"]
    .astype("boolean")
    .fillna(False)
)

markers = markers_raw.loc[
    human
    & canonical
    & (markers_raw["human_sensitivity"] > 0.5)
].copy()

markers = (
    markers.rename(
        columns={"cell_type": "source", "genesymbol": "target"}
    )[["source", "target"]]
    .dropna()
    .drop_duplicates(["source", "target"])
)

available_genes = set(adata.var_names)
markers = markers[markers["target"].isin(available_genes)].copy()

coverage = (
    markers.groupby("source")["target"]
    .nunique()
    .sort_values(ascending=False)
)

valid_sources = coverage[coverage >= 3].index
markers = markers[markers["source"].isin(valid_sources)].copy()

logging.info("Marker overlap by cell type:\n%s", coverage)

# ----------- CLUSTERING - PHASE I ---------- #

sc.pp.neighbors(
    adata,
    n_neighbors=15,
    use_rep="X_pca_harmony"
)

for res in RESOLUTIONS:
    sc.tl.leiden(
        adata, 
        key_added=f"leiden_res_{res:4.2f}", 
        resolution=res,
        flavor="igraph",
        random_state=SEED
    )

sc.tl.umap(adata, random_state=SEED)

sc.pl.umap(
    adata,
    color=[f"leiden_res_{res:4.2f}" for res in RESOLUTIONS],
    legend_loc="on data",
    ncols=4,
    wspace=0.3,
    show=False
)
savefig(RESOURCE_FOLDER / "leiden_umap_by_resolution.jpg")

stability = compare_clusterings(adata, RESOLUTIONS)
logging.info(
    "Clustering Phase I - Stability \n%s",
    stability
)

plt.figure(figsize=(8,4))
plt.plot(
    stability["res_high"],
    stability["ARI"],
    color=ACCENT_COLOR,
    marker="o"
)
plt.plot(
    stability["res_high"],
    stability["NMI"],
    color=MAIN_COLOR,
    marker="o"
)
plt.axvspan(0.4, 0.6, alpha=0.1, color=COLOR_PALETTE_2[1])
plt.xlabel("Higher Resolution", fontsize=12, labelpad=10)
plt.ylabel("Adjusted Rand Index (ARI)", fontsize=12, labelpad=10)
plt.title("Leiden Resolution Sensitivity")
plt.ylim(-0.05,1.05)
plt.grid(alpha=0.7)
plt.tick_params(axis="both", labelsize=9)
plt.tight_layout()
savefig(RESOURCE_FOLDER / "leiden_resolution_stability.jpg")

seed_stability = compute_seed_stability(adata, RESOLUTIONS)
logging.info(
    "Clustering Phase I - Seed Stability \n%s",
    seed_stability
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.lineplot(
    data=seed_stability,
    x="resolutions",
    y="ARI_mean",
    marker="o",
    linewidth=2,
    label="Mean ARI",
    ax=axes[0]
)
axes[0].fill_between(
    seed_stability["resolutions"],
    seed_stability["ARI_mean"] - seed_stability["ARI_sd"],
    seed_stability["ARI_mean"] + seed_stability["ARI_sd"],
    alpha=0.2,
    label="±1 SD"
)
sns.lineplot(
    data=seed_stability,
    x="resolutions",
    y="ARI_min",
    marker="s",
    linestyle="--",
    label="Minimum ARI",
    ax=axes[0]
)
axes[0].set(
    title="Seed Stability across Resolutions",
    xlabel="Resolution",
    ylabel="Adjusted Rand Index (ARI)",
)
axes[0].legend()
sns.lineplot(
    data=seed_stability,
    x="resolutions",
    y="n_clusters_mean",
    marker="o",
    linewidth=2,
    color="darkorange",
    ax=axes[1]
)
axes[1].set(
    title="Clustering Granularity",
    xlabel="Resolution",
    ylabel="Mean Number of Clusters"
)
plt.tick_params(axis="both", labelsize=9)
plt.grid(alpha=0.7)
plt.tight_layout()
savefig(RESOURCE_FOLDER / "leiden_resolution_seed_stability.jpg")

# select leiden resolution 
selected_resolution = 'leiden_res_0.50'

# immune and non-immune cell annotation 
adata.var["immune_marker"] = (adata.var_names.isin(IMMUNE_MARKERS))
adata.var["non_immune_marker"] = (adata.var_names.isin(NON_IMMUNE_MARKERS))

immune_markers = adata.var_names[adata.var["immune_marker"]].tolist()
non_immune_markers = adata.var_names[adata.var["non_immune_marker"]].tolist()

X_immune = adata[:, immune_markers].X
adata.obs["immune_marker_detected"] = (np.asarray(X_immune > 0).any(axis=1))

X_non_immune = adata[:, non_immune_markers].X
adata.obs["non_immune_marker_detected"] = (np.asarray(X_non_immune > 0).any(axis=1))

fig, axes = plt.subplots(1, 3, figsize=(21, 6))
sc.pl.umap(
    adata,
    color=selected_resolution,
    title=f"Leiden Clustering ({selected_resolution})",
    legend_loc="on data",
    ax=axes[0],
    show=False
)

sc.pl.umap(
    adata,
    color="immune_marker_detected",
    palette=["lightgrey", MAIN_COLOR],
    title="Immune marker detected",
    ax=axes[1],
    show=False
)

sc.pl.umap(
    adata,
    color="non_immune_marker_detected",
    palette=["lightgrey", MAIN_COLOR],
    title="Non-immune marker detected",
    ax=axes[2],
    show=False
)
plt.tight_layout()
savefig(RESOURCE_FOLDER / "leiden_umap_by_marker.jpg")

# marker gene scoring 
dc.mt.ulm(data=adata, net=markers, tmin=3)

score_object = dc.pp.get_obsm(adata, key="score_ulm")

logging.info(
    "Marker gene score object \n%s",
    score_object
)

# cell annotation 
ranked_groups = dc.tl.rankby_group(
    adata=score_object, 
    groupby=selected_resolution, 
    reference="rest", 
    method="t-test_overestim_var"
)
ranked_groups = (
    ranked_groups.loc[ranked_groups["stat"] > 0]
    .sort_values(["group", "stat"], ascending=[True, False])
)

n_ctypes = 3
ctypes_dict = ranked_groups.groupby("group", observed=True).head(n_ctypes).groupby("group", observed=True)["name"].apply(lambda x: list(x)).to_dict()

logging.info(
    "Cell annotation \n%s",
    ctypes_dict
)

immune_clusters = {
    cluster: cells
    for cluster, cells in ctypes_dict.items()
    if any(cell in IMMUNE_CELLS for cell in cells)
}

logging.info(
    "Immune-specific clusters \n%s",
    immune_clusters
)

sc.pl.matrixplot(
    adata=score_object,
    var_names=ctypes_dict,
    groupby=selected_resolution,
    standard_scale="var",
    colorbar_title="Z-scaled scores",
    cmap=NAVY_WHITE.reversed(),
    show=False
)
savefig(RESOURCE_FOLDER / "leiden_matrixplot_gene_annotation.jpg")

# ----------- CLUSTERING - PHASE II ---------- #

selected_clusters = set(map(str, immune_clusters.keys()))

adata_subset = adata[adata.obs[selected_resolution].astype(str).isin(map(str, selected_clusters))].copy()

sc.pp.neighbors(
    adata_subset,
    n_neighbors=15,
    use_rep="X_pca_harmony",
)

for res in RESOLUTIONS:
    sc.tl.leiden(
        adata_subset, 
        key_added=f"leiden_refined_res_{res:4.2f}", 
        resolution=res, 
        flavor="igraph",
        random_state=SEED
    )

sc.tl.umap(adata_subset, random_state=SEED)

sc.pl.umap(
    adata_subset,
    color=[f"leiden_refined_res_{res:4.2f}" for res in RESOLUTIONS],
    legend_loc="on data",
    ncols=4,
    wspace=0.3,
    show=False
)
savefig(RESOURCE_FOLDER / "leiden_refined_umap_by_resolution.jpg")

stability = compare_clusterings(adata_subset, RESOLUTIONS)
logging.info(
    "Clustering Phase II - Stability \n%s",
    stability
)

plt.figure(figsize=(8,4))
plt.plot(
    stability["res_high"],
    stability["ARI"],
    color=ACCENT_COLOR,
    marker="o"
)
plt.plot(
    stability["res_high"],
    stability["NMI"],
    color=MAIN_COLOR,
    marker="o"
)
plt.axvspan(0.4, 0.6, alpha=0.1, color=COLOR_PALETTE_2[1])
plt.xlabel("Higher Resolution", fontsize=12, labelpad=10)
plt.ylabel("Adjusted Rand Index (ARI)", fontsize=12, labelpad=10)
plt.title("Leiden Resolution Sensitivity")
plt.ylim(-0.05,1.05)
plt.grid(alpha=0.7)
plt.tick_params(axis="both", labelsize=9)
plt.tight_layout()
savefig(RESOURCE_FOLDER / "leiden_refined_resolution_stability.jpg")

seed_stability = compute_seed_stability(adata_subset, RESOLUTIONS)
logging.info(
    "Clustering Phase I - Seed Stability \n%s",
    seed_stability
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.lineplot(
    data=seed_stability,
    x="resolutions",
    y="ARI_mean",
    marker="o",
    linewidth=2,
    label="Mean ARI",
    ax=axes[0]
)
axes[0].fill_between(
    seed_stability["resolutions"],
    seed_stability["ARI_mean"] - seed_stability["ARI_sd"],
    seed_stability["ARI_mean"] + seed_stability["ARI_sd"],
    alpha=0.2,
    label="±1 SD"
)
sns.lineplot(
    data=seed_stability,
    x="resolutions",
    y="ARI_min",
    marker="s",
    linestyle="--",
    label="Minimum ARI",
    ax=axes[0]
)
axes[0].set(
    title="Seed Stability across Resolutions",
    xlabel="Resolution",
    ylabel="Adjusted Rand Index (ARI)",
)
axes[0].legend()
sns.lineplot(
    data=seed_stability,
    x="resolutions",
    y="n_clusters_mean",
    marker="o",
    linewidth=2,
    color="darkorange",
    ax=axes[1]
)
axes[1].set(
    title="Clustering Granularity",
    xlabel="Resolution",
    ylabel="Mean Number of Clusters"
)
plt.tick_params(axis="both", labelsize=9)
plt.grid(alpha=0.7)
plt.tight_layout()
savefig(RESOURCE_FOLDER / "leiden_refined_resolution_seed_stability.jpg")

# select leiden resolution 
selected_resolution = 'leiden_refined_res_0.60'

# immune and non-immune cell annotation 
adata_subset.var["immune_marker"] = (adata_subset.var_names.isin(IMMUNE_MARKERS))
adata_subset.var["non_immune_marker"] = (adata_subset.var_names.isin(NON_IMMUNE_MARKERS))

immune_markers = adata_subset.var_names[adata_subset.var["immune_marker"]].tolist()
non_immune_markers = adata_subset.var_names[adata_subset.var["non_immune_marker"]].tolist()

X_immune = adata_subset[:, immune_markers].X
adata_subset.obs["immune_marker_detected"] = (np.asarray(X_immune > 0).any(axis=1))

X_non_immune = adata_subset[:, non_immune_markers].X
adata_subset.obs["non_immune_marker_detected"] = (np.asarray(X_non_immune > 0).any(axis=1))

fig, axes = plt.subplots(1, 3, figsize=(21, 6))
sc.pl.umap(
    adata_subset,
    color=selected_resolution,
    title=f"Leiden Clustering ({selected_resolution})",
    legend_loc="on data",
    ax=axes[0],
    show=False
)

sc.pl.umap(
    adata_subset,
    color="immune_marker_detected",
    palette=["lightgrey", MAIN_COLOR],
    title="Immune marker detected",
    ax=axes[1],
    show=False
)

sc.pl.umap(
    adata_subset,
    color="non_immune_marker_detected",
    palette=["lightgrey", MAIN_COLOR],
    title="Non-immune marker detected",
    ax=axes[2],
    show=False
)
plt.tight_layout()
savefig(RESOURCE_FOLDER / "leiden_refined_umap_by_marker.jpg")

# marker gene scoring 
dc.mt.ulm(data=adata_subset, net=markers, tmin=3)

score_object = dc.pp.get_obsm(adata_subset, key="score_ulm")

logging.info(
    "Marker gene score object \n%s",
    score_object
)

# cell annotation 
ranked_groups = dc.tl.rankby_group(
    adata=score_object, 
    groupby=selected_resolution, 
    reference="rest", 
    method="t-test_overestim_var"
)
ranked_groups = (
    ranked_groups.loc[ranked_groups["stat"] > 0]
    .sort_values(["group", "stat"], ascending=[True, False])
)

n_ctypes = 3
ctypes_dict = ranked_groups.groupby("group").head(n_ctypes).groupby("group")["name"].apply(lambda x: list(x)).to_dict()

logging.info(
    "Cell annotation \n%s",
    ctypes_dict
)

immune_clusters = {
    cluster: cells
    for cluster, cells in ctypes_dict.items()
    if any(cell in IMMUNE_CELLS for cell in cells)
}

logging.info(
    "Immune-specific clusters \n%s",
    immune_clusters
)

sc.pl.matrixplot(
    adata=score_object,
    var_names=ctypes_dict,
    groupby=selected_resolution,
    standard_scale="var",
    colorbar_title="Z-scaled scores",
    cmap=NAVY_WHITE.reversed(),
    show=False
)
savefig(RESOURCE_FOLDER / "leiden_refined_matrixplot_gene_annotation.jpg")

adata_deg = adata_subset.copy()

adata_deg.X = adata_deg.layers["counts"].copy()

sc.pp.normalize_total(adata_deg, target_sum=1e4)
sc.pp.log1p(adata_deg)

deg_key = "rank_genes_refined"

sc.tl.rank_genes_groups(
    adata_deg,
    groupby=selected_resolution,
    method="wilcoxon",
    pts=True,
    use_raw=False,
    key_added=deg_key
)

sc.pl.rank_genes_groups(
    adata_deg,
    key=deg_key,
    n_genes=15,
    sharey=False,
    show=False,
)
savefig(RESOURCE_FOLDER / "leiden_refined_cluster_ranked_genes.jpg")

markers_df = sc.get.rank_genes_groups_df(
    adata_deg,
    group=None,
    key=deg_key,
)

markers_df = markers_df.rename(
    columns={
        "group": "cluster",         
        "names": "gene",           
        "logfoldchanges": "logFC",  
        "pvals_adj": "padj",
    }
)

positive_markers = markers_df.loc[
    (markers_df["padj"] < 0.05)
    & (markers_df["logFC"] > 0)
].copy()

positive_markers = positive_markers.sort_values(
    ["cluster", "padj", "logFC"],
    ascending=[True, True, False],
)

positive_markers.to_csv(
    RESOURCE_FOLDER / "leiden_refined_cluster_positive_markers.csv",
    index=False,
)

top_markers = (
    positive_markers
    .groupby("cluster", observed=True)
    .head(5)
    .reset_index(drop=True)
).to_csv(
    RESOURCE_FOLDER / "leiden_refined_cluster_top5_markers.csv",
    index=False
)

#---------- Re-Assess QC ---------------

adata_deg.var["mt"] = (
    adata_deg.var_names.str.startswith("MT-")
)

sc.pp.calculate_qc_metrics(
    adata_deg,
    qc_vars=["mt"],
    inplace=True
)

# umap: metadata
sc.pl.umap(
    adata_subset,
    color=["region", "patient", "sample"],
    ncols=4,
    wspace=0.2,
    palette=COLOR_PALETTE_2,
    show=False
)
savefig(RESOURCE_FOLDER / "leiden_refined_umap_by_group.jpg")

# umap: qc metrics
sc.pl.umap(
    adata_subset,
    color=[
        "leiden_refined_res_0.90",
        "log1p_total_counts",
        "pct_counts_mt",
        "log1p_n_genes_by_counts",
    ],
    cmap=NAVY_WHITE.reversed(),
    ncols=4,
    wspace=0.2,
    show=False
)
savefig(RESOURCE_FOLDER / "leiden_refine_umap_by_metric.jpg")

#---------- MARKER PATTERN VALIDATION ---------------

marker_genes = {
    cell_type: [
        gene for gene in genes
        if gene in adata_subset.var_names
    ]
    for cell_type, genes in IMMUNE_CELL_TYPE_MARKERS.items()
}

sc.pl.matrixplot(
    adata_deg,
    var_names=marker_genes,
    groupby=selected_resolution,
    standard_scale="var",
    colorbar_title="Z-scaled scores",
    cmap=NAVY_WHITE.reversed(),
    show=False
)
savefig(RESOURCE_FOLDER / "leiden_refine_cluster_marker_gene_dotplot.jpg")

#---------- ASSIGN CELL TYPES ---------------

cell_type_map = {
    "0": "Plasma cells",
    "1": "Activated immune cells",
    "2": "Antigen-presenting cells",
    "3": "B cells",
    "4": "Mitochondrial-high cells",
    "5": "Activated lymphocytes",
    "6": "Myeloid / APC-like cells",
    "7": "Activated T cells",
}

adata_deg.obs[f"{selected_resolution}_original"] = (
    adata_deg.obs[selected_resolution].astype(str)
)

adata_deg.obs["cell_type"] = (
    adata_deg.obs[f"{selected_resolution}_original"]
    .map(cell_type_map)
    .fillna("Unknown")
)

adata_deg.obs[
    [f"{selected_resolution}_original", "cell_type"]
].to_csv(
    RESOURCE_FOLDER / "leiden_refined_cell_type_annotations.csv"
)

logging.info(
    "Cell Count of Cell Types \n%s",
    adata_deg.obs.groupby(
        [f"{selected_resolution}_original", "cell_type"],
        observed=True,
    ).size()
)

sc.pl.umap(
    adata_deg,
    color="cell_type",
    frameon=False,
    ncols=2,
    wspace=0.4,
    palette=COLOR_PALETTE_2,
    show=False
)
savefig(RESOURCE_FOLDER / "umap_with_cell_type.jpg")

cell_type_counts = (
    adata_deg.obs.groupby(["region", "cell_type"], observed=True)
    .size()
    .reset_index(name="n_cells")
)

cell_type_counts["abundance_pct"] = (
    cell_type_counts["n_cells"]
    / cell_type_counts.groupby("region")["n_cells"].transform("sum")
    * 100
)

plt.figure(figsize=(12, 6))

ax = sns.barplot(
    data=cell_type_counts,
    x="cell_type",
    y="abundance_pct",
    hue="region",
    errorbar=None,
    palette=[MAIN_COLOR, ACCENT_COLOR],
)

plt.ylabel("Relative abundance (%)")
plt.xlabel("")
plt.xticks(rotation=45, ha="right")
handles, _ = ax.get_legend_handles_labels()

ax.legend(
    handles=handles,
    labels=["Lymph node", "Primary tumor"],
    title="Tissue",
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0,
)
plt.tight_layout()
savefig(RESOURCE_FOLDER / "regions_specific_relative_abundance.jpg")

cell_type_counts = (
    adata_deg.obs.groupby(
        ["sample", "patient", "region", "cell_type"],
        observed=True,
    )
    .size()
    .reset_index(name="n_cells")
)

cell_type_counts["abundance_pct"] = (
    100 * cell_type_counts["n_cells"]
    / cell_type_counts.groupby("sample")["n_cells"].transform("sum")
)

cell_types = cell_type_counts["cell_type"].dropna().unique()

ncols = 4
nrows = (len(cell_types) + ncols - 1) // ncols

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(4 * ncols, 3 * nrows),
    squeeze=False,
)
for ax, cell_type in zip(axes.flat, cell_types):
    subset = cell_type_counts[cell_type_counts["cell_type"] == cell_type]

    for patient, group in subset.groupby("patient", observed=True):
        group = group.set_index("region")

        if {"L", "T"}.issubset(group.index):
            ax.plot(
                [0, 1],
                [
                    group.loc["L", "abundance_pct"],
                    group.loc["T", "abundance_pct"],
                ],
                color="gray",
                alpha=0.7,
                linewidth=1.5,
            )

            ax.scatter(
                [0, 1],
                [
                    group.loc["L", "abundance_pct"],
                    group.loc["T", "abundance_pct"],
                ],
                color=[MAIN_COLOR, ACCENT_COLOR],
                s=45,
                zorder=3,
            )

    ax.set_xticks([0, 1], ["Lymph node", "Primary tumor"])
    ax.set_title(cell_type)
    ax.set_ylabel("Relative abundance (%)")

for ax in axes.flat[len(cell_types):]:
    ax.set_visible(False)

plt.tight_layout()
savefig(RESOURCE_FOLDER / "matched_pair_comparison.jpg")

# ============================================================
# ----------------------- SAVE DATA --------------------------
# ============================================================

adata_export = adata_deg.copy()

write_legacy_h5ad(adata_export, SC_EXP_SAVE_FNAME_LEGACY)

adata_export.write_h5ad(
    SC_EXP_SAVE_FNAME,
    compression="gzip",
)

logging.info("------------ Completed Clustering ------------") 

