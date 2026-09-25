import logging
import time
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

logging.basicConfig(filename=f'../log/{timestamp}-preprocessing.log', filemode='w', 
                    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#------------------------------------

#----------- FILE NAMES -------------

sc_matrix_path = "../data/GSE225600_sc_matrix.mtx/matrix.mtx"
sc_features_path = "../data/GSE225600_sc_features.tsv/GSE225600_sc_features.tsv"
sc_barcodes_path = "../data/GSE225600_sc_barcodes.tsv/GSE225600_sc_barcodes.tsv"
save_path = '../data/GSE225600_sc_matrix.mtx/matrix_2000.h5'

preprocessing_dir = Path("../results/preprocessing") / timestamp
preprocessing_dir.mkdir(parents=True, exist_ok=True)

clustering_dir = Path("../results/clustering") / timestamp
clustering_dir.mkdir(parents=True, exist_ok=True)

#------------------------------------

#----------- FUNCTIONS --------------

def is_outlier(adata, metric: str, nmads: int):
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
        np.median(M) + nmads * median_abs_deviation(M) < M
    )
    return outlier

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

logging.info("[0] Starting Preprocessing") 

# Load Data
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

logging.info("[1] Completed Importing the data and creating AnnData object") 

logging.info("-----Data Summary-----")    
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

# Annotate Data
barcodes.rename(columns={0:'barcode'}, inplace=True)
barcodes.set_index('barcode', inplace=True)
adata.obs = barcodes
adata.obs_names = adata.obs.index.astype(str)                                                   # set barcode as the obs index
adata.obs_names.name = "barcode"                                                                 
barcode_info = adata.obs_names.to_series().str.extract(r"-(?P<region>[LT])(?P<patient>\d+)$")
adata.obs['sample'] = ("P" + barcode_info["patient"] + "_" + barcode_info["region"]).astype("category")
adata.obs['patient'] = barcode_info["patient"].astype("category")
adata.obs['region'] = barcode_info["region"].astype("category")

features.rename(columns={0:'id', 1:'gene_symbol'}, inplace=True)
features.set_index('id', inplace=True)
adata.var = features
adata.var_names = adata.var.index.astype(str) 
adata.var_names.name = "gene_symbol"    
adata.var_names_make_unique()

logging.info("[2] Completed Assigning Cell and Gene Names")
logging.info("obs_names name: %s", adata.obs_names.name)
logging.info("var_names name: %s", adata.var_names.name)
logging.info("First barcodes: %s", adata.obs_names[:5].tolist())
logging.info("First genes: %s", adata.var_names[:5].tolist())
logging.info("obs columns: %s", adata.obs.columns.tolist())
logging.info("var columns: %s", adata.var.columns.tolist())
logging.info("Patient Counts:\n%s", adata.obs['patient'].value_counts())
logging.info("Region Counts:\n%s", adata.obs['region'].value_counts())
logging.info("AnnData Object:\n%s", adata)

# Mitochondrial Genes
adata.var["mt"] = adata.var_names.str.startswith("MT-")
# Ribosomal Genes
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
# Hemoglobin Genes
adata.var["hb"] = adata.var_names.str.contains(r"^HB[ABDEGMQZ]\d*(?!\w)")

sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, percent_top=[20], log1p=True)

# n_genes_by_counts : number of genes expressed in the count matrix 
# total_counts : total counts per cell
# pct_counts_mt : percentage of counts in mitochondrial genes

logging.info("[3] Completed Quality Control Metrics Computation")

adata_tumor = adata[adata.obs["region"] == "T"].copy()
adata_lymph = adata[adata.obs["region"] == "L"].copy()

logging.info("QC metrics:\n%s", adata)
logging.info("Overall QC Statistics\n%s",adata.obs[["total_counts", "n_genes_by_counts", "pct_counts_mt", "total_counts_ribo"]].describe(percentiles=[.01, .5, .99]))
logging.info("Tumor QC Statistics\n%s", adata_tumor.obs[["total_counts", "n_genes_by_counts", "pct_counts_mt", "total_counts_ribo"]].describe(percentiles=[.01, .5, .99]))
logging.info("Lymph QC Statistics\n%s", adata_lymph.obs[["total_counts", "n_genes_by_counts", "pct_counts_mt", "total_counts_ribo"]].describe(percentiles=[.01, .5, .99]))

#---------- Sample Quality Plots ----------

fig, ax = plt.subplots(figsize=(10, 7))
sns.histplot(
    data=adata.obs,
    x="total_counts",
    bins=100,
    log_scale=True,
    alpha=0.6,
#    ax=ax
)
#ax.set_xlabel("Count Depth", fontsize=lab_size)
#ax.set_ylabel("Frequency", fontsize=lab_size)
#ax.set_title(
#    "Distribution of Count Depth",
#    fontsize=lab_size,
#    fontweight="bold",
#    pad=15,
#)
fig.savefig(
    preprocessing_dir / "total_counts.jpeg",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

fig, ax = plt.subplots(figsize=(10, 7))
sns.histplot(
    data=adata.obs,
    x="total_counts",
    hue="region",
    bins=100,
    log_scale=True,
    alpha=0.6,
#    ax=ax,
)
#ax.set_xlabel("Count Depth", fontsize=lab_size)
#ax.set_ylabel("Frequency", fontsize=lab_size)
#ax.set_title(
#    "Distribution of Count Depth by Region",
#    fontsize=lab_size,
#    fontweight="bold",
#   pad=15,
#)
fig.savefig(
    preprocessing_dir / "total_counts_region.jpeg",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

#---------- Data Quality Plots ----------

fig, ax = plt.subplots(figsize=(10, 7))
sns.histplot(
    data=adata.obs,
    x="n_genes_by_counts",
    hue="region",
    bins=100,
    log_scale=True,
    alpha=0.6,
#    ax=ax
)
#ax.set_xlabel("Number of Genes", fontsize=lab_size)
#ax.set_ylabel("Frequency", fontsize=lab_size)
#ax.set_title(
#    "Distribution of Number of Genes",
#    fontsize=lab_size,
#    fontweight="bold",
#    pad=15,
#)
fig.savefig(
    preprocessing_dir / "gene_counts.jpeg",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

sc.pl.scatter(
    adata,
    x="total_counts",
    y="n_genes_by_counts",
    color="pct_counts_mt",
    show=False
)
plt.savefig(
    preprocessing_dir / "fraction_mt_counts.jpg",
    dpi=300,
    bbox_inches="tight"
)

sc.pl.scatter(
    adata[adata.obs['total_counts']<100000],
    x="total_counts",
    y="n_genes_by_counts",
    color="pct_counts_mt",
    show=False
)
plt.savefig(
    preprocessing_dir / "fraction_mt_counts_selection.jpg",
    dpi=300,
    bbox_inches="tight"
)

#---------- Sample Quality Plots ----------

sc.pl.violin(
    adata, 
    "total_counts",
    groupby='sample', 
    color=main_colors,  
    stripplot=False,
    cut=0, rotation=0, 
    log=True, 
    show=False
)
plt.savefig(
    preprocessing_dir / "total_counts_violin.jpg",
    dpi=300,
    bbox_inches="tight"
)

sc.pl.violin(
    adata, 
    "pct_counts_mt",
    groupby='sample', 
    color=main_colors,  
    stripplot=False,
    cut=0, rotation=0,  
    show=False
)
plt.savefig(
    preprocessing_dir / "pct_counts_mt_violin.jpg",
    dpi=300,
    bbox_inches="tight"
)

sc.pl.violin(
    adata_tumor, 
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    groupby='patient', 
    color=main_colors,  
    stripplot=False,
    cut=0, rotation=0, 
    show=False
)
plt.savefig(
    preprocessing_dir / "qc_metrics_violin_tumor.jpg",
    dpi=300,
    bbox_inches="tight"
)

sc.pl.violin(
    adata_lymph, 
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    groupby='patient',
    color=main_colors,  
    stripplot=False, 
    cut=0, rotation=0, 
    show=False
)
plt.savefig(
    preprocessing_dir / "qc_metrics_violin_lymph.jpg",
    dpi=300,
    bbox_inches="tight"
)

sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    groupby='sample',
    stripplot=False, 
    cut=0, rotation=0, 
    show=False
)
plt.savefig(
    preprocessing_dir / "qc_metrics_violin_sample.jpg",
    dpi=300,
    bbox_inches="tight"
)

logging.info("[4] Completed Quality Control")

#---------- Cells and Gene Filtering ----------

sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=200)

sc.pl.scatter(adata[adata.obs['total_counts']<100000], "total_counts", "n_genes_by_counts", color="pct_counts_mt", show=False)
#ax.axhline(y=200, color="red", linestyle="--", linewidth=1.5)
#ax.axvline(x=3, color="red", linestyle="--", linewidth=1.5)
plt.savefig(
    preprocessing_dir / "filtered_scatter.jpeg",
    dpi=300,
    bbox_inches="tight"
)

logging.info("[5] Completed Filtering Cells and Genes")

#---------- Outlier Detection ----------

# Outlier Detection
adata.obs["outlier_counts"] = (is_outlier(adata, "log1p_total_counts", 5))
adata.obs["outlier_genes"] = (is_outlier(adata, "log1p_n_genes_by_counts", 5))
adata.obs["outlier_top20"] = (is_outlier(adata, "pct_counts_in_top_20_genes", 5))
adata.obs["outlier"] = (
    is_outlier(adata, "log1p_total_counts", 5)
    | is_outlier(adata, "log1p_n_genes_by_counts", 5)
    | is_outlier(adata, "pct_counts_in_top_20_genes", 5)
)
adata.obs["mt_outlier"] = (is_outlier(adata, "pct_counts_mt", 3))
logging.info("Count Outliers: %d", adata.obs["outlier_counts"].sum())
logging.info("Gene Outliers: %d", adata.obs["outlier_genes"].sum())
logging.info("Top-20 Outliers: %d", adata.obs["outlier_top20"].sum())
logging.info("Mitochondrial Outliers: %d", adata.obs["mt_outlier"].sum())

#adata.obs["qc_outlier"] = (adata.obs["outlier"] | adata.obs["mt_outlier"])
logging.info("QC Outliers: %d", adata.obs["outlier"].sum())

adata.raw = adata.copy()
adata = adata[~adata.obs["outlier"]].copy()

logging.info(f"Remaining Cells: {adata.n_obs}")

sc.pl.scatter(adata[adata.obs['total_counts']<100000], "total_counts", "n_genes_by_counts", color="pct_counts_mt", show=False)
plt.savefig(
    preprocessing_dir / "outlier_scatter.jpeg",
    dpi=300,
    bbox_inches="tight"
)

sc.pl.violin(
    adata, 
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    groupby='sample',  
    stripplot=False,
    cut=0, rotation=0, 
    show=False
)
plt.savefig(
    preprocessing_dir / "outlier_qc_metrics_violin.jpg",
    dpi=300,
    bbox_inches="tight"
)

logging.info("[6] Completed Outlier Detection")

#---------- Doublet Detection ----------

sc.pp.scrublet(adata, batch_key="sample")

logging.info("Number of doublets detected: %d", adata.obs["predicted_doublet"].sum())

logging.info("[7] Completed Doublet Detection")

# Saving count data
adata.layers["counts"] = adata.X.copy()

#---------- Normalization ---------------

# Normalizing to median total counts and add Size Factor
sc.pp.normalize_total(adata)
adata.obs['size_factors'] = adata.obs.total_counts / np.median(adata.obs.total_counts)

# Logarithmize the data
sc.pp.log1p(adata)

logging.info("[8] Completed Normalization and Log Transformation")

#---------- Batch Correction ---------------

#sc.pp.combat(adata, key='sample')

#logging.info("[9] Completed Batch Correction")

#---------- Feature Selection ---------------

sc.pp.highly_variable_genes(adata, n_top_genes=4000, batch_key="sample")
logging.info('Number of highly variable genes: {:d}'.format(np.sum(adata.var['highly_variable'])))

sc.pl.highly_variable_genes(adata, show=False)
plt.savefig(
    preprocessing_dir / "highly_variable_genes.jpeg",
    dpi=300,
    bbox_inches="tight"
)

logging.info("[10] Completed Feature Selection")

#---------- Dimensionality Reduction ---------------

# PCA
sc.pp.pca(adata, n_comps=50, mask_var="highly_variable", svd_solver="arpack")
sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True, show=False)
plt.savefig(
    preprocessing_dir / "dimensionality_reduction_pca_variance.jpeg",
    dpi=300,
    bbox_inches="tight"
)

sc.pl.pca(
    adata,
    color="sample",
    size=1,
    show=False
)
plt.savefig(
    preprocessing_dir / "dimensionality_reduction_pca.jpeg",
    dpi=300,
    bbox_inches="tight"
)
logging.info("[11A] Completed PCA")

# UMAP
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(
    adata,
    color="sample",
    size=1,
    show=False
)
plt.savefig(
    preprocessing_dir / "dimensionality_reduction_umap.jpeg",
    dpi=300,
    bbox_inches="tight"
)
logging.info("[11B] Completed UMAP")

# TSNE
sc.tl.tsne(adata)
sc.pl.tsne(
    adata,
    color="sample",
    size=1,
    show=False
) 
plt.savefig(
    preprocessing_dir / "dimensionality_reduction_tsne.jpeg",
    dpi=300,
    bbox_inches="tight"
)
logging.info("[11C] Completed TSNE")

# Diffusion Map
sc.tl.diffmap(adata)
sc.pl.diffmap(
    adata,
    color="sample",
    size=1,
    show=False
) 
plt.savefig(
    preprocessing_dir / "dimensionality_reduction_diffusion_map.jpeg",
    dpi=300,
    bbox_inches="tight"
)
logging.info("[11D] Completed Diffusion Map")

# Graph
'''
sc.tl.draw_graph(adata)
sc.pl.draw_graph(
    adata,
    color="sample",
    size=1,
    show=False
) 
plt.savefig(
    preprocessing_dir / "dimensionality_reduction_graph.jpeg",
    dpi=300,
    bbox_inches="tight"
)
logging.info("[11E] Completed Graph")
'''

logging.info("[11] Compelted Dimensionality Reduction")

#---------- Cell Cycle Scoring ---------

# to be continued. 

#---------- Clustering ---------------

sc.tl.leiden(adata, flavor="igraph", n_iterations=2)
sc.pl.umap(adata, color=["leiden"], show=False)
plt.savefig(
    clustering_dir / "umap_leiden.jpeg",
    dpi=300,
    bbox_inches="tight"
)

logging.info("Cluster Counts:\n%s", adata.obs['leiden'].value_counts())

logging.info("[12] Compelted Clustering")

#---------- Re-Assess QC ---------------

sc.pl.umap(adata, color=['region', 'patient', 'total_counts'], show=False)
plt.savefig(
    clustering_dir / "umap_region_counts.jpeg",
    dpi=300,
    bbox_inches="tight"
)

sc.pl.umap(adata, color=['log1p_total_counts', 'pct_counts_mt'], show=False)
plt.savefig(
    clustering_dir / "umap_log_pct_mt.jpeg",
    dpi=300,
    bbox_inches="tight"
)

sc.pl.umap(
    adata,
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

sc.tl.rank_genes_groups(adata, groupby='leiden', key_added='rank_genes_leiden')

sc.pl.rank_genes_groups(adata, key='rank_genes_leiden', groups=['0','1','2'], fontsize=12, show=False)
plt.savefig(
    clustering_dir / "rank_genes_groups_1.jpeg",
    dpi=300,
    bbox_inches="tight"
)
sc.pl.rank_genes_groups(adata, key='rank_genes_leiden', groups=['3','4','5'], fontsize=12, show=False)
plt.savefig(
    clustering_dir / "rank_genes_groups_2.jpeg",
    dpi=300,
    bbox_inches="tight"
)
sc.pl.rank_genes_groups(adata, key='rank_genes_leiden', groups=['6', '7', '8'], fontsize=12, show=False)
plt.savefig(
    clustering_dir / "rank_genes_groups_3.jpeg",
    dpi=300,
    bbox_inches="tight"
)

cell_annotation = sc.tl.marker_gene_overlap(adata, marker_genes, key='rank_genes_leiden')
logging.info("Cell Annotation\n%s", cell_annotation)

cell_annotation_norm = sc.tl.marker_gene_overlap(adata, marker_genes, key='rank_genes_leiden', normalize='reference')
sns.heatmap(cell_annotation_norm, cbar=False, annot=True)
plt.savefig(
    clustering_dir / "cell_annotation_heatmap.jpeg",
    dpi=300,
    bbox_inches="tight"
)

'''
# Saving the Data in .h5 file

with h5py.File(save_path, 'w') as f_normalized:
    f_normalized.create_dataset('X', data=adata.X, compression="gzip", compression_opts=9)
    y = np.array(adata.obs_names, dtype='S')
    f_normalized.create_dataset('Y', data=y, compression="gzip", compression_opts=9)

'''