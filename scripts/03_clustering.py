import logging
import time
import os

import pandas as pd
import scanpy as sc
import numpy as np
import seaborn as sns
import decoupler as dc
import matplotlib.pyplot as plt

from pathlib import Path
timestamp = time.strftime('%Y%m%d')

#----------- DESIGN SETUP -----------

lab_size = 25
tick_size = 15

main_colors = ["#0E2841", "#4E95D9", "#CBCBCB", "#F0F0F0"]
accent_color = "#FFC000"

#------------------------------------

#----------- LOGGER SETUP -----------

logging.basicConfig(filename=f'../log/{timestamp}-03_clustering.log', filemode='w', 
                    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.category").setLevel(logging.WARNING)

logging.info("----------- Starting clustering ------------") 

#------------------------------------

#----------- FILE NAMES -------------

adata_preprocessed_path = Path('../data/processed_data/adata_preprocessed.h5')
save_path = '../data/processed_data/adata_clustered.h5'

clustering_dir = Path("../results/clustering") / timestamp
clustering_dir.mkdir(parents=True, exist_ok=True) 
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

#------------------------------------

#----------- GLOBAL VARIABLES -------

RESOLUTIONS = [0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2.0]

#------------------------------------

#####################################
#------------- MAIN -----------------
#####################################

# ----------- LOAD DATA ---------- #

logging.info("[1] Loading adata object") 

adata = sc.read_h5ad(adata_preprocessed_path)
logging.info("AnnData object:\n%s", adata)

logging.info(
    "Initial dimensions: %d cells x %d genes",
    adata.n_obs,
    adata.n_vars,
)

# ----------- CLUSTERING ---------- #

sc.pp.neighbors(
    adata,
    n_neighbors=15,
    n_pcs=30,
    use_rep="X_pca_harmony"
)

for res in RESOLUTIONS:
    sc.tl.leiden(
        adata, key_added=f"leiden_res_{res:4.2f}", resolution=res, flavor="igraph"
    )

sc.pl.umap(
    adata,
    color=[f"leiden_res_{res:4.2f}" for res in RESOLUTIONS],
    legend_loc="on data",
    ncols=2,
    wspace=0.3,
    show=False
)
savefig(clustering_dir / "umap_leiden_by_resolution.jpg")

'''
sc.tl.leiden(
    adata,
    resolution=0.6, 
    flavor="igraph",
    n_iterations=2,
    key_added="leiden",
)

cluster_counts = (
    adata.obs["leiden"]
    .value_counts()
    .sort_index()
)

logging.info(
    "Leiden clustering completed: %d clusters",
    cluster_counts.size,
)
logging.info(
    "Cluster counts:\n%s",
    cluster_counts.to_string(),
)

# Plot UMAP colored by Leiden clusters
sc.tl.umap(
    adata,
    random_state=42,
)

sc.pl.umap(
    adata,
    color="leiden",
    show=False,
    title="Leiden Clusters", 
    legend_loc = 'on data'
)
savefig(clustering_dir / "umap_leiden.jpg")
'''

#---------- Re-Assess QC ---------------

logging.info("[3] Re-assessing QC")

# umap: metadata
sc.pl.umap(
    adata,
    color=["region", "patient", "total_counts"],
    ncols=3,
    wspace=0.4,
    show=False
)
savefig(clustering_dir / "umap_region_counts.jpg")

# umap: qc metrics
sc.pl.umap(
    adata,
    color=["log1p_total_counts", "pct_counts_mt"],
    ncols=2,
    wspace=0.4,
    show=False
)
savefig(clustering_dir / "umap_qc_metrics.jpg")

sc.pl.umap(
    adata,
    color=[
        "leiden_res_0.50",
        "log1p_total_counts",
        "pct_counts_mt",
        "log1p_n_genes_by_counts",
    ],
    ncols=2,
    wspace=0.5,
    show=False
)
savefig(clustering_dir / "umap_cluster_qc.jpg")

sc.pl.umap(
    adata, 
    color=['CD8A', 'CD4', 'NKG7', 'GNLY', 'CCR7','FOXP3','CXCL13','CD3D'],
    ncols=2,
    wspace=0.3,
    hspace=0.3,
    show=False
)
savefig(clustering_dir / "umap_cluster_genes.jpg")

#---------- MARKER GENE CURATION ---------------

logging.info("[4] Maker gene curation")

human_gene_db = dc.op.resource(
    name='PanglaoDB',
    organism='human',
    license='academic'
)

# filter by canonical_marker and human
markers = human_gene_db[
    human_gene_db["human"].astype(bool)
    & human_gene_db["canonical_marker"].astype(bool)
    & (human_gene_db["human_sensitivity"].astype(float) > 0.5)
]

# remove duplicated entries
markers = markers[~markers.duplicated(["cell_type", "genesymbol"])]

# format
markers = markers.rename(columns={"cell_type": "source", "genesymbol": "target"})
markers = markers[["source", "target"]]

logging.info(
    "Marker gene set \n%s",
    markers.head()
)

#---------- MARKER GENE SCORING ---------------

dc.mt.ulm(data=adata, net=markers, tmin=3)

score_object = dc.pp.get_obsm(adata, key="score_ulm")

logging.info(
    "Marker gene score object \n%s",
    score_object
)

#---------- CELL ANNOTATION ---------------

selected_resolution = 'leiden_res_0.05'

ranked_groups = dc.tl.rankby_group(adata=score_object, groupby=selected_resolution, reference="rest", method="t-test_overestim_var")
ranked_groups = ranked_groups[ranked_groups["stat"] > 0]

n_ctypes = 3
ctypes_dict = ranked_groups.groupby("group").head(n_ctypes).groupby("group")["name"].apply(lambda x: list(x)).to_dict()

logging.info(
    "Cell annotation \n%s",
    ctypes_dict
)

sc.pl.matrixplot(
    adata=score_object,
    var_names=ctypes_dict,
    groupby=selected_resolution,
    standard_scale="var",
    colorbar_title="Z-scaled scores",
    cmap="RdBu_r",
    show=False
)
savefig(clustering_dir / "matrixplot_gene_annotation.jpg")

sc.pl.violin(
    adata=score_object,
    keys=["T cells", "B cells", "Platelets", "Monocytes", "NK cells"],
    groupby=selected_resolution,
    rotation=90,
    multi_panel=True,
    stripplot=False,
    show=False
)
savefig(clustering_dir / "violin_cell_distributions.jpg")

ann = (
    ranked_groups[ranked_groups["stat"] > 0]
    .groupby("group")
    .head(1)
    .set_index("group")["name"]
    .astype(str)
)

# Make duplicate annotation names unique
counts = ann.groupby(ann).cumcount()

ann_unique = ann.copy()
ann_unique[:] = [
    name if count == 0 else f"{name}_{count + 1}"
    for name, count in zip(ann, counts)
]

dict_ann = ann_unique.to_dict()

logging.info(
    "Cluster annotation \n%s",
    dict_ann
)

adata.obs[selected_resolution] = adata.obs[selected_resolution].cat.rename_categories(dict_ann)

sc.pl.umap(
    adata=adata,
    color=selected_resolution,
    ncols=1,
    show=False
)
savefig(clustering_dir / "umap_with_cell_type.jpg")

# ============================================================
# ----------------------- SAVE DATA --------------------------
# ============================================================

logging.info("[5] Saving adata object as .h5 file")

adata_export = adata.copy()

write_legacy_h5ad(adata_export, save_path)

logging.info("------------ Completed Clustering ------------") 

