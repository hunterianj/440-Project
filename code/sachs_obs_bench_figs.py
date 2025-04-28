import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import utils
import dataUtils
import pandas as pd
import numpy as np

# Load the benchmark results
with open('benchmarks/benchmark_results_obs.json', 'r') as f:
    ccpg_results = json.load(f)
with open('benchmarks/benchmark_results_obs_pc.json', 'r') as f:
    pc_results = json.load(f)

# Load the ground-truth graph
ground_truth_graph = dataUtils.loadSachsGroundTruth()

# get node names
log_obs_df, _ = dataUtils.loadSachsObservational()
node_names = list(log_obs_df.columns)

# Setup the figure
n_graphs = len(ccpg_results) + 1
n_cols = 4
n_rows = (n_graphs + n_cols - 1) // n_cols
fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))

# Flatten in case axs is 2D
axs = axs.flatten()

# Label list
labels = [chr(ord('a') + i) for i in range(n_graphs)]

# Plot ground-truth
_, _ = utils.plot_graph_2(ground_truth_graph, ax=axs[0])
axs[0].set_title(f"({labels[0]}) Ground Truth")
# axs[0].set_facecolor("#f0f0f0")
# axs[0].add_patch(
#     patches.Rectangle(
#         (0, 0), 1, 1,
#         transform=axs[0].transAxes,
#         facecolor="#f0f0f0",
#         zorder=-1
#     )
# )
axs[0].add_patch(
    patches.FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="round,pad=0.02,rounding_size=0.25",
        transform=axs[0].transAxes,
        facecolor="#f0f0f0",
        edgecolor="none",
        zorder=-1
    )
)
axs[0].axis('off')

# Plot each CCPG results
for i, res in enumerate(ccpg_results):
    components = res['result_details']['components']
    edges = res['result_details']['edges']
    # Create the graph from components and edges
    graph = utils.get_ccpg_graph(components, edges, node_names=node_names)
    _, _ = utils.plot_graph_2(graph, ax=axs[i+1])
    label = f"({labels[i+1]}) {res['data_label']}-{res['ci_test_name']}"
    axs[i+1].set_title(label)
    axs[i+1].set_facecolor("#f0f0f0")
    axs[i+1].add_patch(
        patches.FancyBboxPatch(
            (0, 0), 1, 1,
            boxstyle="round,pad=0.02,rounding_size=0.25",
            transform=axs[i+1].transAxes,
            facecolor="#f0f0f0",
            edgecolor="none",
            zorder=-1
        )
    )
    axs[i+1].axis('off')

# Remove any empty subplots
for j in range(n_graphs, len(axs)):
    axs[j].axis('off')

for ax in axs:
    for spine in ax.spines.values():
        spine.set_edgecolor('lightgray')
        spine.set_linewidth(1)

fig.tight_layout()
# plt.show()
filename = f"figs/observational_ccpg_graphs.png"
plt.savefig(filename, bbox_inches="tight", pad_inches=0.1)


# bar graphs
def flatten_results(results, algorithm):
    return [
        {
            "Algorithm": algorithm,
            "Dataset": r["data_label"],
            "CI Test": r["ci_test_name"],
            "SHD": r["result_details"]["shd"],
            "S Metric": r["result_details"]["s_metric"],
            "C Metric": r["result_details"]["c_metric"],
            "SC Metric": r["result_details"]["sc_metric"]
        }
        for r in results
    ]

all_data = flatten_results(ccpg_results, "CCPG") + flatten_results(pc_results, "PC")
df = pd.DataFrame(all_data)

datasets = df['Dataset'].unique()

ccpg_color = '#3399ff'  # Blue
pc_color = '#ff9933'    # Orange

def plot_metric_1(df, metric, ylabel, fileName):
    fig, axes = plt.subplots(len(datasets), 1, figsize=(14, 6 * len(datasets)), sharey=False)

    for i, dataset in enumerate(datasets):
        subset = df[df['Dataset'] == dataset]
        ci_tests = subset['CI Test'].unique()

        indices = np.arange(len(ci_tests))
        bar_width = 0.35

        ccpg = subset[subset['Algorithm'] == 'CCPG'].set_index('CI Test')
        pc = subset[subset['Algorithm'] == 'PC'].set_index('CI Test')

        axes[i].bar(indices - bar_width/2, ccpg[metric].reindex(ci_tests), width=bar_width, label='CCPG', color=ccpg_color)
        axes[i].bar(indices + bar_width/2, pc[metric].reindex(ci_tests), width=bar_width, label='PC', color=pc_color)

        axes[i].set_title(f"{dataset.capitalize()} Data")
        axes[i].set_xticks(indices)
        axes[i].set_xticklabels([ci_test_labels.get(t, t).capitalize() for t in ci_tests], rotation=45, ha='right')
        axes[i].set_ylabel(ylabel)
        axes[i].legend()

    # plt.tight_layout()
    fig.subplots_adjust(hspace=0.4, left=0.1, right=0.95, bottom=0.15, top=0.9)
    # plt.show()
    plt.savefig(fileName, bbox_inches="tight", pad_inches=0.1)

ci_test_labels = {
    'fisherz': 'FisherZ',
    'rcit': 'RCIT',
    'gaussbic': 'Gauss-BIC',
    'chisq': 'Chi-Sq',
    'gsq': 'G-Sq',
    'kci': 'KCI'
}

plot_metric_1(df, 'SHD', 'Structural Hamming Distance (SHD)', "figs/observational_ccpg_v_pc_bar_graphs_shd.png")
plot_metric_1(df, 'S Metric', 'S-Score', "figs/observational_ccpg_v_pc_bar_graphs_sscore.png")
plot_metric_1(df, 'C Metric', 'C-Score', "figs/observational_ccpg_v_pc_bar_graphs_cscore.png")
plot_metric_1(df, 'SC Metric', 'SC-Score', "figs/observational_ccpg_v_pc_bar_graphs_scscore.png")
