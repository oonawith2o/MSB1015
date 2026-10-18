import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

COMPONENT_PAIRS = [(0, 1), (2, 3), (0, 3), (1, 2)]

QC_GROUPS = ["patient", "region", "sample"]

COUNT_METRICS = ["total_counts", "n_genes_by_counts"]

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
    "#0072B2", 
    "#E69F00", 
    "#009E73", 
    "#CC79A7", 
    "#56B4E9", 
    "#D55E00", 
    "#4D4D4D", 
    "#332288", 
    "#88CCEE", 
    "#44AA99", 
    "#117733", 
    "#999933",  
    "#DDCC77", 
    "#AA4499", 
    "#882255",  
]

sns.set_theme(
    style="whitegrid",
    context="talk",
    font="DejaVu Sans",
    font_scale=1.05,
    palette=COLOR_PALETTE_2,
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

def savefig(fig, path):
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

def style_pca_figure(fig, adata_pca):
    fig.set_size_inches((18,5))

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
            fontsize=14,
            labelpad=8,
        )
        ax.set_ylabel(
            f"PC{pc_y + 1} ({variance[pc_y]:.2f}%)",
            fontsize=14,
            labelpad=8,
        )

        ax.tick_params(axis="both", labelsize=9)
        ax.grid(axis="both", alpha=0.35, linewidth=0.6)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    return pca_axes

def plot_qc_violinplots(adata, output_dir, suffix=None):
    for group in QC_GROUPS:
        fig, axes = plt.subplots(
            1,
            len(QC_OBS_METRICS),
            figsize=(5 * len(QC_OBS_METRICS), 4),
            squeeze=False,
        )
        axes = axes.ravel()

        for ax, metric in zip(axes, QC_OBS_METRICS):
            plot_data = adata.obs[[group, metric]].dropna().copy()

            if metric in COUNT_METRICS:
                upper = plot_data[metric].quantile(0.99)
                plot_data = plot_data[plot_data[metric] <= upper]

            sns.violinplot(
                data=plot_data,
                x=group,
                y=metric,
                color=MAIN_COLOR,
                linewidth=1,
                linecolor="k",
                inner=None,
                legend=False,
                ax=ax,
            )

            if metric in COUNT_METRICS:
                ax.set_yscale("log")
                ax.set_ylim(bottom=1)

            ax.set_xlabel("")
            ax.set_ylabel(
                QC_METRICS_NAMES.get(metric, "Value"),
                fontsize=14,
                labelpad=10,
            )
            ax.tick_params(
                axis="x",
                bottom=True,
                labelsize=14,
            )
            ax.tick_params(axis="y", labelsize=14)
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

        filename = f"qc_metrics_violin_{group}{suffix}.jpg"
        savefig(fig, output_dir / filename)