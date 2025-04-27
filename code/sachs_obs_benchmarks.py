import dataUtils, utils
from ccpg import ccpg
from causallearn.graph.SHD import SHD
from causallearn.graph.SCM import SCM
import numpy as np

sachs_ground_truth = dataUtils.loadSachsGroundTruth()
utils.plot_graph(sachs_ground_truth, "sachs_ground_truth")

discrete_obs_dataframe = dataUtils.loadSachsObservationalDiscrete()
discrete_obs_data = discrete_obs_dataframe.values
discrete_obs_data_names = list(discrete_obs_dataframe.columns)

log_obs_df, continuous_log_df = dataUtils.loadSachsObservational()
continuous_names = list(log_obs_df.columns)
log_obs_data = log_obs_df.values
continuous_obs_data = continuous_log_df.values

# fisherz CI tests on continuous data
alphas = [0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0001]
bestSHD = np.inf
bestAlpha = 0.4
bestResult = {}
for alpha in alphas:
    result = utils.benchmark_ccpg_against_ground_truth(
        data=continuous_obs_data,
        ground_truth_graph=sachs_ground_truth,
        node_names=continuous_names,
        ci_test_name="fisherz",
        alpha=alpha,
        verbose=True
    )
    if result["shd"] < bestSHD:
        bestSHD = result["shd"]
        bestAlpha = alpha
        bestResult = result

print(f"continuous-fisherz: best alpha found = {bestAlpha} with results: {bestResult}")

# fisherz CI tests on log-continuous data
alphas = [0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0001]
bestSHD = np.inf
bestAlpha = 0.4
bestResult = {}
for alpha in alphas:
    result = utils.benchmark_ccpg_against_ground_truth(
        data=log_obs_data,
        ground_truth_graph=sachs_ground_truth,
        node_names=continuous_names,
        ci_test_name="fisherz",
        alpha=alpha,
        verbose=True
    )
    if result["shd"] < bestSHD:
        bestSHD = result["shd"]
        bestAlpha = alpha
        bestResult = result

print(f"log-cont-fisherz: best alpha found = {bestAlpha} with results: {bestResult}")