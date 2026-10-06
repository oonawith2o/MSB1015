import logging
import time
import os
import harmonypy as hm
import pandas as pd
import scanpy as sc
import numpy as np
import seaborn as sns
import ClustAssessPy as ca
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

logging.info("------------ Starting preprocessing ------------") 

#------------------------------------

#----------- FILE NAMES -------------

DATA_FOLDER="../data"
SC_EXP_FNAME = os.path.join(DATA_FOLDER, "processed_data/adata_clustered.h5")
SC_EXP_SAVE_FNAME = os.path.join(DATA_FOLDER, "processed_data/adata_preprocessed.h5")

preprocessing_qc_dir = Path("../results/preprocessing") / timestamp / "qc"
preprocessing_qc_dir.mkdir(parents=True, exist_ok=True)

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
SAMPLING_PCT = 0.3

MIN_GENES = 300
MIN_CELLS = 5
MT_CUTOFF = 30.0

QC_GROUPS = ["patient", "region", "sample"]

QC_METRICS = [
    "n_genes_by_counts",
    "total_counts",
    "pct_counts_mt",
]

NUMBER_OF_VARIABLE_GENES = 3000

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

adata = sc.read_h5ad(SC_EXP_FNAME)
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

#---------- HIGHLY VARIABLE GENES ---------------

logging.info("[5] Highly variable genes") 

sc.pp.highly_variable_genes(
    adata,
    n_top_genes=NUMBER_OF_VARIABLE_GENES,
    flavor="seurat_v3",
    batch_key="sample"
)

# get highly variable genes ranked
highly_var_genes_sorted = adata.var['highly_variable_rank'][~adata.var['highly_variable_rank'].isna()].sort_values().index.to_numpy().tolist()

# get most abundant genes ranked
most_abundant_genes_sorted = pd.Series(np.asarray(adata.X.sum(axis=0)).flatten(), index=adata.var_names).sort_values(ascending=False).head(NUMBER_OF_VARIABLE_GENES).index.tolist()

logging.info(
    "Highly variable genes \n%s",
    highly_var_genes_sorted
)

logging.info(
    "Most abundant genes \n%s",
    most_abundant_genes_sorted
)

sc.pl.highly_variable_genes(adata, show=False)
savefig(preprocessing_qc_dir / "highly_variable_genes.jpg")

#---------- NORMALIZATION ---------------

logging.info("[6] Normalization and log transformation") 

# saving count data
adata.layers["counts"] = adata.X.copy()

# normalizing to median total counts and add size factor
# - makes data comparable across cells and mitigate the influence of cell-specific biases
sc.pp.normalize_total(adata, target_sum=1e4)
adata.obs['size_factors'] = adata.obs.total_counts / np.median(adata.obs.total_counts)

# logarithmize the data
sc.pp.log1p(adata)

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
savefig(preprocessing_qc_dir / "pca_variance_ratio_pre_batch_correction.jpg")

sc.pl.pca_loadings(
    adata_balanced,
    components='1,2,3',
    show=False
)
savefig(preprocessing_qc_dir / "pca_loadings_pre_batch_correction.jpg")

sc.pl.pca(
    adata_balanced,
    color="sample",
    components=["1,2", "3,4"],
    ncols=2,
    show=False
)
savefig(preprocessing_qc_dir / "pca_by_sample_pre_batch_correction.jpg")

sc.pl.pca(
    adata_balanced,
    color=["pct_counts_mt", "n_genes_by_counts"],
    components=["1,2"],
    ncols=2,
    show=False,
)
savefig(preprocessing_qc_dir / "pca_by_qc.jpg")

sc.pl.pca(
    adata_balanced,
    color=["region", "patient"],
    components=["1,2"],
    show=False
)
savefig(preprocessing_qc_dir / "pca_by_metadata_pre_batch_correction.jpg")

# UMAP
sc.pp.neighbors(adata_balanced, n_neighbors=15, n_pcs=30)
sc.tl.umap(adata_balanced)
sc.pl.umap(
    adata_balanced,
    color="sample",
    size=1,
    show=False
)
savefig(preprocessing_qc_dir / "umap_by_sample_pre_batch_correction.jpeg")
sc.pl.umap(
    adata_balanced,
    color="region",
    size=1,
    show=False
)
savefig(preprocessing_qc_dir / "umap_by_region_pre_batch_correction.jpeg")

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

#sc.pp.neighbors(adata_balanced, n_neighbors=10, n_pcs=30)
#sc.tl.umap(adata_balanced)
#sc.pl.umap(adata_balanced, color = ['FCER1G','TYROBP', 'cell_label'], legend_loc = 'on data')


# ----------- METADATA/SAMPLE COMPOSITION ---------- #

logging.info("[11] Generating metadata QC plots")

for column in QC_GROUPS:

    counts = adata_balanced.obs[column].value_counts()

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
        preprocessing_qc_dir / f"{column}_barcode_counts.jpg"
    )

# ----------- QC METRICS BY GROUP ---------- #

logging.info("[112] Generating grouped QC plots")

for group in QC_GROUPS:

    # violin plots
    sc.pl.violin(
        adata_balanced,
        QC_METRICS,
        groupby=group,
        stripplot=False,
        multi_panel=True,
        rotation=45,
        show=False
    )
    plt.savefig(
        preprocessing_qc_dir / f"qc_violin_{group}.jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")

    # log-scale count distribution
    sc.pl.violin(
        adata_balanced,
        ["n_genes_by_counts", "total_counts"],
        groupby=group,
        stripplot=False,
        multi_panel=True,
        log=True,
        rotation=45,
        show=False
    )
    plt.savefig(
        preprocessing_qc_dir / f"qc_violin_{group}_log.jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")

# ----------- QC RELATIONSHIPS ---------- #

logging.info("[13] Generating QC relationship plots")

# counts vs detected genes
sc.pl.scatter(
    adata_balanced,
    x="total_counts",
    y="n_genes_by_counts",
    color="pct_counts_mt",
    alpha=0.6,
    size=2,
    show=False
)
savefig(
    preprocessing_qc_dir / "qc_scatter_counts_vs_genes.jpg"
)

#---------- DIMENSIONALITY REDUCTION ---------------

logging.info("[14] Dimensionality reduction")

'''
# PCA
sc.pp.pca(adata_balanced, n_comps=50, mask_var="highly_variable", svd_solver="arpack")
sc.pl.pca_variance_ratio(adata_balanced, n_pcs=50, log=True, show=False)
savefig(preprocessing_qc_dir / "dimensionality_reduction_pca_variance.jpg")

sc.pl.pca(
    adata_balanced,
    color="sample",
    size=1,
    show=False
)
savefig(preprocessing_qc_dir / "dimensionality_reduction_pca.jpg")
logging.info("[12A] Completed PCA")
'''

# UMAP
sc.pp.neighbors(adata_balanced, n_neighbors=15, n_pcs=30, use_rep="X_pca_harmony")
sc.tl.umap(adata_balanced)
sc.pl.umap(
    adata_balanced,
    color="sample",
    size=1,
    show=False
)
savefig(preprocessing_qc_dir / "umap_by_sample_post_batch_correction.jpeg")
sc.pl.umap(
    adata_balanced,
    color="region",
    size=1,
    show=False
)
savefig(preprocessing_qc_dir / "umap_by_region_post_batch_correction.jpeg")
logging.info("[12B] Completed UMAP")

'''
# TSNE
sc.tl.tsne(adata_balanced)
sc.pl.tsne(
    adata_balanced,
    color="sample",
    size=1,
    show=False
) 
savefig(preprocessing_qc_dir / "dimensionality_reduction_tsne.jpg")
logging.info("[12C] Completed TSNE")

# Diffusion Map
sc.tl.diffmap(adata_balanced)
sc.pl.diffmap(
    adata_balanced,
    color="sample",
    size=1,
    show=False
) 
savefig(preprocessing_qc_dir / "dimensionality_reduction_diffusion_map.jpg")
logging.info("[12D] Completed Diffusion Map")

# Graph
sc.tl.draw_graph(adata_balanced)
sc.pl.draw_graph(
    adata_balanced,
    color="sample",
    size=1,
    show=False
) 
savefig(preprocessing_qc_dir / "dimensionality_reduction_graph.jpg")
logging.info("[12E] Completed Graph")
'''

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

logging.info("[13] Saving adata object as .h5 file")

adata_balanced.write_h5ad(
    SC_EXP_SAVE_FNAME,
    compression="gzip",
)

logging.info("------------ Completed Preprocessing ------------") 