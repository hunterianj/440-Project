import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import utils
import dataUtils

# Load the benchmark results
with open('benchmark_results_obs.json', 'r') as f:
    results = json.load(f)

# Load the ground-truth graph
ground_truth_graph = dataUtils.loadSachsGroundTruth()

# get node names
log_obs_df, _ = dataUtils.loadSachsObservational()
node_names = list(log_obs_df.columns)

# Setup the figure
n_graphs = len(results) + 1
n_cols = 5
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
        boxstyle="round,pad=0.02,rounding_size=0.05",
        transform=axs[0].transAxes,
        facecolor="#f0f0f0",
        edgecolor="none",
        zorder=-1
    )
)
axs[0].axis('off')

# Plot each CCPG results
for i, res in enumerate(results):
    components = res['result_details']['components']
    edges = res['result_details']['edges']
    # Create the graph from components and edges
    graph = utils.get_ccpg_graph(components, edges, node_names=node_names)
    _, _ = utils.plot_graph_2(graph, ax=axs[i+1])
    label = f"({labels[i+1]}) {res['data_label']}-{res['ci_test_name']}"
    axs[i+1].set_title(label)
    axs[i+1].set_facecolor("#f0f0f0")
    axs[i+1].axis('off')

# Remove any empty subplots
for j in range(n_graphs, len(axs)):
    axs[j].axis('off')

for ax in axs:
    for spine in ax.spines.values():
        spine.set_edgecolor('lightgray')
        spine.set_linewidth(1)

fig.tight_layout()
plt.show()
