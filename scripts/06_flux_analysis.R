install.packages("BiocManager")
install.packages("remotes")

remotes::install_github("scverse/anndataR")
packageVersion("anndataR")
library(anndataR)
library(SingleCellExperiment)

h5ad_path <- file.path(
  getwd(),
  "data",
  "processed_data",
  "adata_clustered.h5"
)

stopifnot(file.exists(h5ad_path))

sce <- read_h5ad(h5ad_path, as = "SingleCellExperiment")

