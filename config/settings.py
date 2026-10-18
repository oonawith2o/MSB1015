import os
import logging
from pathlib import Path
from datetime import datetime

RUN_ID = os.getenv(
    "ANALYSIS_RUN_ID",
    datetime.now().strftime("%Y%m%d"),
)

# project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
RUN_DIR = RESULTS_DIR / "runs" / RUN_ID

QC_DIR = RUN_DIR / "qc"
PREPROCESSING_DIR = RUN_DIR / "preprocessing"
CLUSTERING_DIR = RUN_DIR / "clustering"
GRN_INFERENCE_DIR = RUN_DIR / "grn-inference"
LOGS_DIR = RUN_DIR / "logs"

CELL_TYPE = os.getenv("CELL_TYPE", "unknown_cell_type")
REGION = os.getenv("REGION", "unknown_region")

CELL_TYPE_DIR = GRN_INFERENCE_DIR / CELL_TYPE
REGION_DIR = CELL_TYPE_DIR / REGION
SUBNETWORK_FOLDER = REGION_DIR / "subnetwork"

MODULES_FNAME = REGION_DIR / "pyscenic_modules.p"
MOTIFS_FNAME = REGION_DIR / "pyscenic_motifs.p"
REGULONS_FNAME = REGION_DIR / "pyscenic_regulons.p"
AUCELL_FNAME = REGION_DIR / "pyscenic_aucell.csv"

if os.getenv("RUN_GRN_INFERENCE") == "1":
    for directory in [
        CELL_TYPE_DIR,
        REGION_DIR,
        SUBNETWORK_FOLDER,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    QC_DIR,
    PREPROCESSING_DIR,
    CLUSTERING_DIR,
    GRN_INFERENCE_DIR,
    LOGS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# input data 
SC_EXP_FNAME = (
    RAW_DATA_DIR
    / "GSE225600_sc_matrix.mtx"
    / "matrix.mtx"
)

SC_FEATURES_FNAME = (
    RAW_DATA_DIR
    / "GSE225600_sc_features.tsv"
    / "GSE225600_sc_features.tsv"
)

SC_BARCODES_FNAME = (
    RAW_DATA_DIR
    / "GSE225600_sc_barcodes.tsv"
    / "GSE225600_sc_barcodes.tsv"
)

PATIENT_INFO_FNAME = (
    RAW_DATA_DIR    
    / "supplementary_data"
    / "ADVS-10-2205395-s008.xlsx"
)

# pySCENIC reference resources
PYSCENIC_DATA_DIR = DATA_DIR / "pySCENIC"

HS_TFS_FNAME = PYSCENIC_DATA_DIR / "hs_hgnc_tfs.txt"

DATABASES_GLOB = (
    PYSCENIC_DATA_DIR
    / "hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather"
)

MOTIF_ANNOTATIONS_FNAME = (
    PYSCENIC_DATA_DIR
    / "motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl"
)

# anndata artifacts
SC_EXP_RAW_FNAME = PROCESSED_DATA_DIR / "adata_raw.h5"
SC_EXP_PREPROCESSED_FNAME = PROCESSED_DATA_DIR / "adata_preprocessed.h5"
SC_EXP_CLUSTERED_FNAME = PROCESSED_DATA_DIR / "adata_clustered.h5"
SC_EXP_CLUSTERED_FNAME_LEGACY = PROCESSED_DATA_DIR / "adata_clustered_legacy.h5"


# setup logger -----------------------------------------------

def setup_logging(script_name):

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOGS_DIR / f"{RUN_ID}-{script_name}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )

    # reduce noisy third-party logs
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.category").setLevel(logging.WARNING)

    logger = logging.getLogger(script_name)
    logger.info("----------- Starting %s ------------", script_name)

    return logger
