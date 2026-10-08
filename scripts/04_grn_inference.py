import time, logging
import os, glob, pickle
import h5py, anndata

import pandas as pd
import seaborn as sns
import numpy as np
import seaborn as sns
import scanpy as sc
import matplotlib.pyplot as plt

from arboreto.utils import load_tf_names
from arboreto.algo import grnboost2

from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
from pyscenic.utils import modules_from_adjacencies
from pyscenic.prune import prune2df
from pyscenic.transform import df2regulons
from pyscenic.aucell import aucell

timestamp = time.strftime('%Y%m%d')

#----------- DESIGN SETUP -----------

lab_size = 25
tick_size = 15

main_colors = ["#0E2841", "#4E95D9", "#CBCBCB", "#F0F0F0"]
accent_color = "#FFC000"

#------------------------------------

#----------- LOGGER SETUP -----------

logging.basicConfig(filename=f'../log/{timestamp}-04_grn_inference.log', filemode='w', 
                    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.category").setLevel(logging.WARNING)

logging.info("----------- Starting GRN Inference ------------") 

#------------------------------------

#----------- FILE NAMES -------------

DATA_FOLDER="../data"
RESULTS_FOLDER="../results"

SC_EXP_FNAME = os.path.join(DATA_FOLDER, "processed_data/adata_clustered.h5")
HS_TFS_FNAME = os.path.join(DATA_FOLDER, "pySCENIC/hs_hgnc_tfs.txt")

DATABASES_GLOB = os.path.join(DATA_FOLDER, "pySCENIC/hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather")
MOTIF_ANNOTATIONS_FNAME = os.path.join(DATA_FOLDER, "pySCENIC/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl")

ADJACENCIES_FNAME = os.path.join(RESULTS_FOLDER, "grn-inference", "adjacencies.tsv")
MODULES_FNAME = os.path.join(RESULTS_FOLDER, "grn-inference", "modules.p")
MOTIFS_FNAME = os.path.join(RESULTS_FOLDER, "grn-inference", "motifs.csv")
REGULONS_FNAME = os.path.join(RESULTS_FOLDER, "grn-inference", "regulons.p")
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

# ----------- DERIVE LIST OF TRANSCRIPTION FACTORS (TF) ---------- #

logging.info("[2] Loading TF list") 

tf_names = load_tf_names(HS_TFS_FNAME)

# ----------- LOAD RANKING DATABASES ---------- #

logging.info("[3] Loading ranking database") 

db_fnames = glob.glob(DATABASES_GLOB)
def name(fname):
    return os.path.splitext(os.path.basename(fname))[0]
dbs = [RankingDatabase(fname=fname, name=name(fname)) for fname in db_fnames]

logging.info(
    "Feather Ranking Database object \n%s",
    dbs
)

# ----------- PHASE 1 ---------- #

logging.info("[4] Phase 1 - GRN inference (GRNBoost2)") 

expression_data = pd.DataFrame(
    adata.X,
    index=adata.obs_names,    
    columns=adata.var_names  
)

adjacencies = grnboost2(expression_data=expression_data, tf_names=tf_names, verbose=True)

logging.info(
    "Adjacency matrix \n%s",
    adjacencies.head()
)

adjacencies.to_csv(ADJACENCIES_FNAME, index=False, sep='\t')
logging.info("Exported adjcancy matrix (%s)", ADJACENCIES_FNAME)

# visualize distribution of weights 
sns.histplot(
    np.log10(adjacencies["importance"]), 
    bins=50
)

# derive potential regulomes from these co-expression modules
modules = list(modules_from_adjacencies(adjacencies, expression_data))

with open(MODULES_FNAME, 'wb') as f:
    pickle.dump(modules, f)

#with open(MODULES_FNAME, 'rb') as f:
#    modules = pickle.load(f)

# ----------- PHASE 2 ---------- #

logging.info("[5] Phase 2 - Prune modules for targets (RcisTarget)")

# prune modules for targets with cis regulatory footprints

df = prune2df(dbs, modules, MOTIF_ANNOTATIONS_FNAME)

logging.info(
    "Module overview \n%s",
    df.head()
)

df.to_csv(MOTIFS_FNAME)
logging.info("Exported motif names (%s)", MOTIFS_FNAME)

# dataframe is converted to regulons
regulons = df2regulons(df)

with open(REGULONS_FNAME, 'wb') as f:
    pickle.dump(regulons, f)

#with open(REGULONS_FNAME, 'rb') as f:
#    regulons = pickle.load(f)

# ----------- PHASE 3 ---------- #

logging.info("[6] Phase 3 - Cellular regulon enrichment matrix (AUCell)")

# cellular regulon enrichment matrix (aka AUCell)

auc_mtx = aucell(adata.X, regulons, num_workers=1)

# ============================================================
# ----------------------- SAVE DATA --------------------------
# ============================================================

logging.info("------------ Completed GRN Inference ------------")
