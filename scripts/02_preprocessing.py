import logging
import time
import inspect
import h5py

import pandas as pd
import scanpy as sc
import numpy as np
import seaborn as sns
from kennard_stone import train_test_split
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

logging.basicConfig(filename=f'../log/{timestamp}-02_preprocessing.log', filemode='w', 
                    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.category").setLevel(logging.WARNING)

logging.info("[0] Starting preprocessing") 

#------------------------------------

#----------- FILE NAMES -------------

adata_raw_path = Path('../data/processed_data/adata_raw.h5')

preprocessing_qc_dir = Path("../results/preprocessing") / timestamp preprocessing_qc_dir.mkdir(parents=True, exist_ok=True)

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

#------------------------------------

#----------- GLOBAL VARIABLES -------

SEED = 123
SAMPLING_PCT = 0.7

MIN_GENES = 300
MIN_CELLS = 5
MT_CUTOFF = 30.0

#------------------------------------

#----------- MARKER GENES -----------
marker_genes = dict()
marker_genes['Stem'] = ['Lgr5', 'Ascl2', 'Slc12a2', 'Axin2', 'Olfm4', 'Gkn3']
marker_genes['Enterocyte (Proximal)'] = ['Gsta1','Rbp2','Adh6a','Apoa4','Reg3a','Creb3l3','Cyp3a13','Cyp2d26','Ms4a10','Ace','Aldh1a1','Rdh7','H2-Q2', 'Hsd17b6','Gstm3','Gda','Apoc3','Gpd1','Fabp1','Slc5a1','Mme','Cox7a1','Gsta4','Lct','Khk','Mttp','Xdh','Sult1b1', 'Treh','Lpgat1','Dhrs1','Cyp2c66','Ephx2','Cyp2c65','Cyp3a25','Slc2a2','Ugdh','Gstm6','Retsat','Ppap2a','Acsl5', 'Cyb5r3','Cyb5b','Ckmt1','Aldob','Ckb','Scp2','Prap1']
marker_genes['Enterocyte (Distal)'] = ['Tmigd1','Fabp6','Slc51b','Slc51a','Mep1a','Fam151a','Naaladl1','Slc34a2','Plb1','Nudt4','Dpep1','Pmp22','Xpnpep2','Muc3','Neu1','Clec2h','Phgr1','2200002D01Rik','Prss30','Cubn','Plec','Fgf15','Crip1','Krt20','Dhcr24','Myo15b','Amn','Enpep','Anpep','Slc7a9','Ocm','Anxa2','Aoc1','Ceacam20','Arf6','Abcb1a','Xpnpep1','Vnn1','Cndp2','Nostrin','Slc13a1','Aspa','Maf','Myh14']
marker_genes['Goblet'] = ['Agr2', 'Fcgbp', 'Tff3', 'Clca1', 'Zg16', 'Tpsg1', 'Muc2', 'Galnt12', 'Atoh1', 'Rep15', 'S100a6', 'Pdia5', 'Klk1', 'Pla2g10', 'Spdef', 'Lrrc26', 'Ccl9', 'Bace2', 'Bcas1', 'Slc12a8', 'Smim14', 'Tspan13', 'Txndc5', 'Creb3l4', 'C1galt1c1', 'Creb3l1', 'Qsox1', 'Guca2a', 'Scin', 'Ern2', 'AW112010', 'Fkbp11', 'Capn9', 'Stard3nl', 'Slc50a1', 'Sdf2l1', 'Hgfa', 'Galnt7', 'Hpd', 'Ttc39a', 'Tmed3', 'Pdia6', 'Uap1', 'Gcnt3', 'Tnfaip8', 'Dnajc10', 'Ergic1', 'Tsta3', 'Kdelr3', 'Foxa3', 'Tpd52', 'Tmed9', 'Spink4', 'Nans', 'Cmtm7', 'Creld2', 'Tm9sf3', 'Wars', 'Smim6', 'Manf', 'Oit1', 'Tram1', 'Kdelr2', 'Xbp1', 'Serp1', 'Vimp', 'Guk1', 'Sh3bgrl3', 'Cmpk1', 'Tmsb10', 'Dap', 'Ostc', 'Ssr4', 'Sec61b', 'Pdia3', 'Gale', 'Klf4', 'Krtcap2', 'Arf4', 'Sep15', 'Ssr2', 'Ramp1', 'Calr', 'Ddost']
marker_genes['Paneth'] = ['Gm15284', 'AY761184', 'Defa17', 'Gm14851', 'Defa22', 'Defa-rs1', 'Defa3', 'Defa24', 'Defa26', 'Defa21', 'Lyz1', 'Gm15292', 'Mptx2', 'Ang4']
marker_genes['Enteroendocrine'] = ['Chgb', 'Gfra3', 'Cck', 'Vwa5b2', 'Neurod1', 'Fev', 'Aplp1', 'Scgn', 'Neurog3', 'Resp18', 'Trp53i11', 'Bex2', 'Rph3al', 'Scg5', 'Pcsk1', 'Isl1', 'Maged1', 'Fabp5', 'Celf3', 'Pcsk1n', 'Fam183b', 'Prnp', 'Tac1', 'Gpx3', 'Cplx2', 'Nkx2-2', 'Olfm1', 'Vim', 'Rimbp2', 'Anxa6', 'Scg3', 'Ngfrap1', 'Insm1', 'Gng4', 'Pax6', 'Cnot6l', 'Cacna2d1', 'Tox3', 'Slc39a2', 'Riiad1']
marker_genes['Tuft'] = ['Alox5ap', 'Lrmp', 'Hck', 'Avil', 'Rgs13', 'Ltc4s', 'Trpm5', 'Dclk1', 'Spib', 'Fyb', 'Ptpn6', 'Matk', 'Snrnp25', 'Sh2d7', 'Ly6g6f', 'Kctd12', '1810046K07Rik', 'Hpgds', 'Tuba1a', 'Pik3r5', 'Vav1', 'Tspan6', 'Skap2', 'Pygl', 'Ccdc109b', 'Ccdc28b', 'Plcg2', 'Ly6g6d', 'Alox5', 'Pou2f3', 'Gng13', 'Bmx', 'Ptpn18', 'Nebl', 'Limd2', 'Pea15a', 'Tmem176a', 'Smpx', 'Itpr2', 'Il13ra1', 'Siglecf', 'Ffar3', 'Rac2', 'Hmx2', 'Bpgm', 'Inpp5j', 'Ptgs1', 'Aldh2', 'Pik3cg', 'Cd24a', 'Ethe1', 'Inpp5d', 'Krt23', 'Gprc5c', 'Reep5', 'Csk', 'Bcl2l14', 'Tmem141', 'Coprs', 'Tmem176b', '1110007C09Rik', 'Ildr1', 'Galk1', 'Zfp428', 'Rgs2', 'Inpp5b', 'Gnai2', 'Pla2g4a', 'Acot7', 'Rbm38', 'Gga2', 'Myo1b', 'Adh1', 'Bub3', 'Sec14l1', 'Asah1', 'Ppp3ca', 'Agt', 'Gimap1', 'Krt18', 'Pim3', '2210016L21Rik', 'Tmem9', 'Lima1', 'Fam221a', 'Nt5c3', 'Atp2a3', 'Mlip', 'Vdac3', 'Ccdc23', 'Tmem45b', 'Cd47', 'Lect2', 'Pla2g16', 'Mocs2', 'Arpc5', 'Ndufaf3']

#------------------------------------

#####################################
#------------- MAIN -----------------
#####################################

# ----------- LOAD DATA ---------- #

logging.info("[1] Loading adata object") 

adata = sc.read_h5ad(adata_raw_path)
logging.info("AnnData object:\n%s", adata)

logging.info(
    "Initial dimensions: %d cells x %d genes",
    adata.n_obs,
    adata.n_vars,
)

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
    bins=100,
    color=main_colors[0],
    ax=ax
)
ax.set_xlabel("Doublet score")
ax.set_ylabel("Number of cells")
fig.tight_layout()
savefig(
    preprocessing_qc_dir / "doublet_score_distribution.jpg"
)

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

#---------- NORMALIZATION ---------------

logging.info("[5] Normalization and log transformation") 

# Saving count data
adata.layers["counts"] = adata.X.copy()

# Normalizing to median total counts and add Size Factor
sc.pp.normalize_total(adata)
adata.obs['size_factors'] = adata.obs.total_counts / np.median(adata.obs.total_counts)

# Logarithmize the data
sc.pp.log1p(adata)

#---------- PCA FOR SAMPLING ---------------

logging.info("[6] PCA for random sampling")

sc.tl.pca(
    adata,
    n_comps=50,
    svd_solver="arpack"
)

#---------- RANDOM SAMPLING ---------------

logging.info("[5] Random sampling") 

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

#---------- HIGHLY VARIABLE GENES ---------------

logging.info("[6] Highly variable genes") 

sc.pp.highly_variable_genes(
    adata_balanced,
    n_top_genes=4000,
    flavor="seurat",
    batch_key="sample"
)

logging.info(
    "Using %d highly variable genes",
    adata_balanced.var['highly_variable'].sum()
)

sc.pl.highly_variable_genes(adata_balanced, show=False)
savefig(preprocessing_qc_dir / "highly_variable_genes.jpg")

# keep highly variable genes
adata_balanced = adata_balanced[:,adata_balanced.var["highly_variable"]].copy()

#---------- SCALING ------------------

logging.info("[7] Scaling")

sc.pp.scale(
    adata_balanced,
    max_value=10
)

#---------- DIMENSIONALITY REDUCTION ---------------

logging.info("[8] PCA")

sc.tl.pca(
    adata_balanced,
    n_comps=50,
    svd_solver="arpack"
)

sc.pl.pca_variance_ratio(
    adata_balanced,
    n_pcs=50,
    log=True,
    show=False
)

plt.savefig(
    preprocessing_qc_dir / "pca_variance_ratio_pre_batch_correction.jpg",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

sc.pl.pca_overview(
    adata_balanced,
    color="sample",
    show=False
)

plt.savefig(
    preprocessing_qc_dir / "pca_overview_pre_batch_correction_sample.jpg",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

sc.pl.pca(
    adata_balanced,
    color="sample",
    components=["1,2", "3,4"],
    ncols=2,
    show=False
)

plt.savefig(
    preprocessing_qc_dir / "pca_by_sample_pre_batch_correction.jpg",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

sc.pl.pca_overview(
    adata_balanced,
    color="tumor_size",
    show=False
)

plt.savefig(
    preprocessing_qc_dir / "pca_overview_pre_batch_correction_tumor_size.jpg",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

#---------- Batch Correction ---------------

logging.info("[9] Batch correction")

sc.pp.combat(adata_balanced, key='sample')

#---------- Dimensionality Reduction ---------------

# PCA
sc.pp.pca(adata_balanced, n_comps=50, mask_var="highly_variable", svd_solver="arpack")
sc.pl.pca_variance_ratio(adata_balanced, n_pcs=50, log=True, show=False)
plt.savefig(
    preprocessing_qc_dir / "dimensionality_reduction_pca_variance.jpeg",
    dpi=300,
    bbox_inches="tight"
)

sc.pl.pca(
    adata_balanced,
    color="sample",
    size=1,
    show=False
)
plt.savefig(
    preprocessing_qc_dir / "dimensionality_reduction_pca.jpeg",
    dpi=300,
    bbox_inches="tight"
)
logging.info("[11A] Completed PCA")

# UMAP
sc.pp.neighbors(adata_balanced)
sc.tl.umap(adata_balanced)
sc.pl.umap(
    adata_balanced,
    color="sample",
    size=1,
    show=False
)
plt.savefig(
    preprocessing_qc_dir / "dimensionality_reduction_umap.jpeg",
    dpi=300,
    bbox_inches="tight"
)
logging.info("[11B] Completed UMAP")

# TSNE
sc.tl.tsne(adata_balanced)
sc.pl.tsne(
    adata_balanced,
    color="sample",
    size=1,
    show=False
) 
plt.savefig(
    preprocessing_qc_dir / "dimensionality_reduction_tsne.jpeg",
    dpi=300,
    bbox_inches="tight"
)
logging.info("[11C] Completed TSNE")

# Diffusion Map
sc.tl.diffmap(adata_balanced)
sc.pl.diffmap(
    adata_balanced,
    color="sample",
    size=1,
    show=False
) 
plt.savefig(
    preprocessing_qc_dir / "dimensionality_reduction_diffusion_map.jpeg",
    dpi=300,
    bbox_inches="tight"
)
logging.info("[11D] Completed Diffusion Map")

# Graph
sc.tl.draw_graph(adata_balanced)
sc.pl.draw_graph(
    adata_balanced,
    color="sample",
    size=1,
    show=False
) 
plt.savefig(
    preprocessing_qc_dir / "dimensionality_reduction_graph.jpeg",
    dpi=300,
    bbox_inches="tight"
)
logging.info("[11E] Completed Graph")


logging.info("[11] Compelted Dimensionality Reduction")

quit()

#---------- Cell Cycle Scoring ---------

# to be continued. 

#---------- Clustering ---------------

sc.tl.leiden(adata_balanced, flavor="igraph", n_iterations=2)
sc.pl.umap(adata_balanced, color=["leiden"], show=False)
plt.savefig(
    clustering_dir / "umap_leiden.jpeg",
    dpi=300,
    bbox_inches="tight"
)

logging.info("Cluster Counts:\n%s", adata_balanced.obs['leiden'].value_counts())

logging.info("[12] Compelted Clustering")

#---------- Re-Assess QC ---------------

sc.pl.umap(adata_balanced, color=['region', 'patient', 'total_counts'], show=False)
plt.savefig(
    clustering_dir / "umap_region_counts.jpeg",
    dpi=300,
    bbox_inches="tight"
)

sc.pl.umap(adata_balanced, color=['log1p_total_counts', 'pct_counts_mt'], show=False)
plt.savefig(
    clustering_dir / "umap_log_pct_mt.jpeg",
    dpi=300,
    bbox_inches="tight"
)

sc.pl.umap(
    adata_balanced,
    color=["leiden", "log1p_total_counts", "pct_counts_mt", "log1p_n_genes_by_counts"],
    wspace=0.5,
    ncols=2,
    show=False
)
plt.savefig(
    clustering_dir / "umap_cell_filtering.jpeg",
    dpi=300,
    bbox_inches="tight"
)

logging.info("[13] Completed Re-Assess Quality Control ")

#---------- Marker Genes & Cluster Annotation ---------------

sc.tl.rank_genes_groups(adata_balanced, groupby='leiden', key_added='rank_genes_leiden')

sc.pl.rank_genes_groups(adata_balanced, key='rank_genes_leiden', groups=['0','1','2'], fontsize=12, show=False)
plt.savefig(
    clustering_dir / "rank_genes_groups_1.jpeg",
    dpi=300,
    bbox_inches="tight"
)
sc.pl.rank_genes_groups(adata_balanced, key='rank_genes_leiden', groups=['3','4','5'], fontsize=12, show=False)
plt.savefig(
    clustering_dir / "rank_genes_groups_2.jpeg",
    dpi=300,
    bbox_inches="tight"
)
sc.pl.rank_genes_groups(adata_balanced, key='rank_genes_leiden', groups=['6', '7', '8'], fontsize=12, show=False)
plt.savefig(
    clustering_dir / "rank_genes_groups_3.jpeg",
    dpi=300,
    bbox_inches="tight"
)

cell_annotation = sc.tl.marker_gene_overlap(adata_balanced, marker_genes, key='rank_genes_leiden')
logging.info("Cell Annotation\n%s", cell_annotation)

cell_annotation_norm = sc.tl.marker_gene_overlap(adata_balanced, marker_genes, key='rank_genes_leiden', normalize='reference')
sns.heatmap(cell_annotation_norm, cbar=False, annot=True)
plt.savefig(
    clustering_dir / "cell_annotation_heatmap.jpeg",
    dpi=300,
    bbox_inches="tight"
)

'''
# Saving the Data in .h5 file

with h5py.File(save_path, 'w') as f_normalized:
    f_normalized.create_dataset('X', data=adata_balanced.X, compression="gzip", compression_opts=9)
    y = np.array(adata_balanced.obs_names, dtype='S')
    f_normalized.create_dataset('Y', data=y, compression="gzip", compression_opts=9)

'''