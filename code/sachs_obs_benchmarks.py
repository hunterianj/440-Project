import dataUtils, utils
from ccpg import ccpg
from causallearn.graph.SHD import SHD
from causallearn.graph.SCM import SCM
import numpy as np
import ci_tests
import json
import os

ci_tests.register_ci_tests()

verbose = True
sachs_ground_truth = dataUtils.loadSachsGroundTruth()
utils.plot_graph(sachs_ground_truth, "sachs_ground_truth")

discrete_obs_dataframe = dataUtils.loadSachsObservationalDiscrete()
discrete_obs_data = discrete_obs_dataframe.values
# discrete_obs_data_names = list(discrete_obs_dataframe.columns)
# discrete_obs_data_names = [name.lower() for name in discrete_obs_data_names]

log_obs_df, continuous_log_df = dataUtils.loadSachsObservational()
continuous_names = list(log_obs_df.columns)
discrete_obs_data_names = continuous_names # plc is plcg in the discrete dataset
log_obs_data = log_obs_df.values
continuous_obs_data = continuous_log_df.values

results_filename = "benchmark_results_obs.json"

# Create file if it doesn't exist
if not os.path.exists(results_filename):
    with open(results_filename, "w") as f:
        json.dump([], f)

def save_result(data):
    """Append the latest benchmark result to file immediately."""
    with open(results_filename, "r+") as f:
        try:
            existing_data = json.load(f)
        except json.JSONDecodeError:
            existing_data = []
        existing_data.append(data)
        f.seek(0)
        json.dump(existing_data, f, indent=2)
        f.truncate()

def run_benchmark_suite(data,
                        data_label,
                        node_names,
                        ci_test_name,
                        params,
                        param_name,
                        ci_test_kwargs = None):
    bestSHD = np.inf
    bestParam = None
    bestResult = None
    print(f"\n--- Benchmarking: {data_label} + {ci_test_name} ---")

    for param in params:
        result = utils.benchmark_ccpg_against_ground_truth(
            data=data,
            ground_truth_graph=sachs_ground_truth,
            node_names=node_names,
            ci_test_name=ci_test_name,
            alpha=param,
            verbose=verbose,
            ci_test_kwargs=ci_test_kwargs,
        )

        # record = result.copy()
        # record["graph"] = None
        # record["components"] = [list(comp) for comp in record["components"]]
        # record["edges"] = list(record["edges"])
        # result_record = {
        #     "data_label": data_label,
        #     "ci_test_name": ci_test_name,
        #     "param_name": param_name,
        #     "param_value": bestParam,
        #     "shd": record["shd"],
        #     "result_details": record
        # }
        # save_result(result_record)

        if result["shd"] < bestSHD:
            bestSHD = result["shd"]
            bestParam = param
            bestResult = result

    print(f"Best {param_name} for {data_label} + {ci_test_name}: {bestParam} with SHD = {bestSHD}")
    print(f"Result: {bestResult}")

    result = bestResult.copy()
    result["graph"] = None
    result["edges"] = list(result["edges"])
    result["components"] = [list(comp) for comp in result["components"]]
    result_record = {
        "data_label": data_label,
        "ci_test_name": ci_test_name,
        "param_name": param_name,
        "param_value": bestParam,
        "shd": bestResult["shd"],
        "result_details": result
    }
    save_result(result_record)
    return bestParam, bestResult

# Parameters for tuning
alphas = [0.1, 0.05, 0.01, 0.001, 0.0001]
bic_thresholds = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]

# fisherz CI test on continuous data
run_benchmark_suite(
    data=continuous_obs_data,
    data_label="continuous",
    node_names=continuous_names,
    ci_test_name="fisherz",
    params=alphas,
    param_name="alpha"
)

# fisherz CI test on log-transformed data
run_benchmark_suite(
    data=log_obs_data,
    data_label="log-continuous",
    node_names=continuous_names,
    ci_test_name="fisherz",
    params=alphas,
    param_name="alpha"
)

# fisherz CI test on discretized data
run_benchmark_suite(
    data=discrete_obs_data,
    data_label="discrete",
    node_names=discrete_obs_data_names,
    ci_test_name="fisherz",
    params=alphas,
    param_name="alpha"
)

# ci_test_kwargs = {
#     "K": 5,
#     "J": 4,
#     # "alpha": 500,
#     # "use_gp": True,
# }
# # FastKCI CI test on continuous data
# run_benchmark_suite(
#     data=continuous_obs_data,
#     data_label="continuous",
#     node_names=continuous_names,
#     ci_test_name="fastkci",
#     params=alphas,
#     param_name="alpha",
#     ci_test_kwargs=ci_test_kwargs
# )
#
# # FastKCI CI test on log-transformed data
# run_benchmark_suite(
#     data=log_obs_data,
#     data_label="log-continuous",
#     node_names=continuous_names,
#     ci_test_name="fastkci",
#     params=alphas,
#     param_name="alpha",
#     ci_test_kwargs=ci_test_kwargs
# )

ci_test_kwargs = {
    # "kernelX": "Linear",
    # "kernelY": "Linear",
    # "kernelZ": "Gaussian",
    "est_width": "median",
    "approx": True,
    "null_ss": 5000
}
# KCI CI test on continuous data
run_benchmark_suite(
    data=continuous_obs_data,
    data_label="continuous",
    node_names=continuous_names,
    ci_test_name="kci",
    params=alphas,
    param_name="alpha",
    ci_test_kwargs=ci_test_kwargs
)

# KCI CI test on log-transformed data
run_benchmark_suite(
    data=log_obs_data,
    data_label="log-continuous",
    node_names=continuous_names,
    ci_test_name="kci",
    params=alphas,
    param_name="alpha",
    ci_test_kwargs=ci_test_kwargs
)

# this isn't too bad for speed, could go for some extensive hyper-parameter tuning
ci_test_kwargs = {
    # "approx": "chi2",
    # "approx": "gamma",
    # "approx": "hbe",
    # "approx": "lpd4", # default
    # "approx": "perm",
    "num_f": 200, # default = 100
    "num_f2": 20 # default = 5
}
# RCIT CI test on continuous data
run_benchmark_suite(
    data=continuous_obs_data,
    data_label="continuous",
    node_names=continuous_names,
    ci_test_name="rcit",
    params=alphas,
    param_name="alpha",
    ci_test_kwargs=ci_test_kwargs
)

# Gaussian BIC custom CI test on continuous data
run_benchmark_suite(
    data=continuous_obs_data,
    data_label="continuous",
    node_names=continuous_names,
    ci_test_name="gaussbic",
    params=bic_thresholds,
    param_name="threshold"
)

# Gaussian BIC custom CI test on log-transformed data
run_benchmark_suite(
    data=log_obs_data,
    data_label="log-continuous",
    node_names=continuous_names,
    ci_test_name="gaussbic",
    params=bic_thresholds,
    param_name="threshold"
)

# Chi-Square CI test on discretized data
run_benchmark_suite(
    data=discrete_obs_data,
    data_label="discrete",
    node_names=discrete_obs_data_names,
    ci_test_name="chisq",
    params=alphas,
    param_name="alpha"
)

# G^2 CI test on discretized data
run_benchmark_suite(
    data=discrete_obs_data,
    data_label="discrete",
    node_names=discrete_obs_data_names,
    ci_test_name="gsq",
    params=alphas,
    param_name="alpha"
)
